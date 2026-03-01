"""
Phase 6: Question Answering — Streamlit App
============================================
Interactive QA interface using DistilBERT fine-tuned on SQuAD v1.1.
Paste a context paragraph, ask a question, and see the extracted answer highlighted.
"""

import os
import sys
import re

import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QA with Transformers",
    page_icon="🤖",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .answer-box {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1B5E20;
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        padding: 20px 25px;
        border-radius: 12px;
        border-left: 8px solid #4CAF50;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #FFD54F;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 700;
        box-shadow: 0 2px 4px rgba(255, 213, 79, 0.4);
    }
    .context-box {
        line-height: 1.8;
        font-size: 1.1rem;
        background: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    h1 {
        color: #2c3e50;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "distilbert-squad-finetuned")
HF_MODEL = "distilbert/distilbert-base-uncased-distilled-squad"


# ── Model Loading (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading QA model …")
def load_model():
    """Load tokenizer + model once and cache across reruns."""
    if os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR)
    else:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        model = AutoModelForQuestionAnswering.from_pretrained(HF_MODEL)
        os.makedirs(MODEL_DIR, exist_ok=True)
        tokenizer.save_pretrained(MODEL_DIR)
        model.save_pretrained(MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


# ── Inference ────────────────────────────────────────────────────────────────
def predict_answer(context: str, question: str, tokenizer, model, device, max_length: int = 384):
    """Run QA inference and return (answer_text, confidence, start_char, end_char)."""
    inputs = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=max_length,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )

    offset_mapping = inputs.pop("offset_mapping").numpy()
    inputs.pop("overflow_to_sample_mapping", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    best_answer = ""
    best_score = float("-inf")
    best_start_char = 0
    best_end_char = 0

    for chunk_idx in range(outputs.start_logits.shape[0]):
        start_logits = outputs.start_logits[chunk_idx].cpu().numpy()
        end_logits = outputs.end_logits[chunk_idx].cpu().numpy()

        start_idx = int(np.argmax(start_logits))
        end_idx = int(np.argmax(end_logits))
        if end_idx < start_idx:
            end_idx = start_idx

        score = float(start_logits[start_idx] + end_logits[end_idx])

        if score > best_score:
            best_score = score
            offsets = offset_mapping[chunk_idx]
            if start_idx < len(offsets) and end_idx < len(offsets):
                s = int(offsets[start_idx][0])
                e = int(offsets[end_idx][1])
                if s is not None and e is not None:
                    best_answer = context[s:e]
                    best_start_char = s
                    best_end_char = e

    return best_answer.strip(), best_score, best_start_char, best_end_char


# ── Highlight helper ─────────────────────────────────────────────────────────
def highlight_answer(context: str, start: int, end: int) -> str:
    """Return HTML with the answer span highlighted."""
    before = context[:start]
    answer = context[start:end]
    after = context[end:]
    return (
        f'<div class="context-box">'
        f'{_escape(before)}'
        f'<mark class="highlight">{_escape(answer)}</mark>'
        f'{_escape(after)}'
        f'</div>'
    )

def _escape(text: str) -> str:
    """Escape HTML special characters, preserve newlines."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("\n", "<br>")
    return text


# ── Sample data ──────────────────────────────────────────────────────────────
SAMPLES = [
    {
        "title": "🏈 Super Bowl 50",
        "context": (
            "Super Bowl 50 was an American football game to determine the champion of the "
            "National Football League (NFL) for the 2015 season. The American Football Conference "
            "(AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion "
            "Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on "
            "February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California."
        ),
        "question": "Which NFL team represented the AFC at Super Bowl 50?",
    },
    {
        "title": "🌍 Amazon Rainforest",
        "context": (
            "The Amazon rainforest, alternatively the Amazon jungle, also known in English as Amazonia, "
            "is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon "
            "basin of South America. This basin encompasses 7,000,000 km2, of which 5,500,000 km2 are "
            "covered by the rainforest. This region includes territory belonging to nine nations and 3,344 "
            "formally acknowledged indigenous territories."
        ),
        "question": "How many square kilometers of the Amazon basin are covered by the rainforest?",
    },
    {
        "title": "🧬 DNA Structure",
        "context": (
            "In 1953, James Watson and Francis Crick published a paper describing the double helix "
            "structure of DNA. Their model was based on X-ray diffraction data collected by Rosalind "
            "Franklin and Raymond Gosling. The discovery of DNA's structure was one of the most important "
            "scientific achievements of the 20th century and earned Watson and Crick the Nobel Prize in "
            "Physiology or Medicine in 1962."
        ),
        "question": "Who collected the X-ray diffraction data used by Watson and Crick?",
    },
    {
        "title": "🚀 Apollo 11",
        "context": (
            "Apollo 11 was the spaceflight that first landed humans on the Moon. Commander Neil Armstrong "
            "and lunar module pilot Buzz Aldrin formed the American crew that landed the Apollo Lunar "
            "Module Eagle on July 20, 1969. Armstrong became the first person to step onto the lunar "
            "surface six hours and 39 minutes later on July 21."
        ),
        "question": "When did the Apollo Lunar Module Eagle land on the Moon?",
    }
]


# ── UI ───────────────────────────────────────────────────────────────────────
def main():
    # Load model
    tokenizer, model, device = load_model()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center;'>
                <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="80" style="margin-bottom:10px;">
                <h2 style='margin-top:0;'>QA Transformers</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.markdown("### 🤖 Model Info")
        st.markdown(f"**Model**: DistilBERT-SQuAD")
        param_count = sum(p.numel() for p in model.parameters())
        st.markdown(f"**Parameters**: `{param_count:,}`")
        st.markdown(f"**Compute**: `{device}`")
        st.markdown("---")
        st.markdown("### ✨ Try a Sample")
        for i, sample in enumerate(SAMPLES):
            if st.button(sample["title"], key=f"sample_{i}", use_container_width=True):
                st.session_state["context"] = sample["context"]
                st.session_state["question"] = sample["question"]
                
        st.markdown("---")
        with st.expander("ℹ️ How it works"):
            st.info(
                "This app uses a **DistilBERT** model fine-tuned on the SQuAD dataset. "
                "When you ask a question, the model reads the context paragraph and "
                "extracts the most likely text span that answers your question."
            )

    # ── Header ──
    st.markdown(
        "<h1 style='text-align:center;'>🤖 Ask the AI: Question Answering</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#5f6368; font-size:1.2rem;'>"
        "Provide a paragraph of text, and then ask me any question about it!</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Inputs ──
    col1, col2 = st.columns([1.5, 1], gap="large")

    with col1:
        st.markdown("### 📄 Context Paragraph")
        context = st.text_area(
            "Context",
            value=st.session_state.get("context", ""),
            height=280,
            placeholder="Paste your context paragraph here …",
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("### ❓ Your Question")
        question = st.text_area(
            "Question",
            value=st.session_state.get("question", ""),
            height=120,
            placeholder="What would you like to know?",
            label_visibility="collapsed"
        )
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("✨  Find Answer", type="primary", use_container_width=True)

    # ── Inference & Display ──
    if run_btn:
        if not context.strip():
            st.warning("Please provide a context paragraph first.")
            return
        if not question.strip():
            st.warning("Please enter a question to ask.")
            return

        with st.spinner("🧠 Analyzing text..."):
            answer, confidence, start, end = predict_answer(
                context, question, tokenizer, model, device
            )

        if not answer:
            st.error("😕 The model could not find an answer in the given context.")
            return

        st.balloons()
        st.markdown("<hr style='border:1px dashed #ccc; margin: 2rem 0;'>", unsafe_allow_html=True)

        # Answer card
        st.markdown("<h2 style='text-align:center; color:#1B5E20;'>🎯 Here's what I found:</h2>", unsafe_allow_html=True)
        
        ans_col, conf_col = st.columns([3, 1])
        with ans_col:
            st.markdown(
                f"<div class='answer-box'>{answer}</div>",
                unsafe_allow_html=True,
            )
        with conf_col:
            # Normalize confidence to roughly 0-1 range (heuristic: sigmoid of score/10)
            norm_conf = 1 / (1 + np.exp(-confidence / 10))
            st.metric("Model Confidence", f"{norm_conf:.1%}")
            st.progress(float(norm_conf))

        # Highlighted context
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📖 Answer Extracted From Context:")
        st.markdown(highlight_answer(context, start, end), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
