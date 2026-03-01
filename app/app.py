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
    page_icon="🔍",
    layout="wide",
)

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
        f'<div style="line-height:1.8; font-size:1.05rem;">'
        f'{_escape(before)}'
        f'<mark style="background-color:#FFD54F; padding:2px 4px; border-radius:4px; font-weight:600;">'
        f'{_escape(answer)}</mark>'
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
]


# ── UI ───────────────────────────────────────────────────────────────────────
def main():
    # Load model
    tokenizer, model, device = load_model()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## 🔍 QA with Transformers")
        st.markdown("---")
        st.markdown("### Model Info")
        st.markdown(f"**Model**: DistilBERT-SQuAD")
        param_count = sum(p.numel() for p in model.parameters())
        st.markdown(f"**Parameters**: {param_count:,}")
        st.markdown(f"**Device**: `{device}`")
        st.markdown("---")
        st.markdown("### Try a Sample")
        for i, sample in enumerate(SAMPLES):
            if st.button(sample["title"], key=f"sample_{i}", use_container_width=True):
                st.session_state["context"] = sample["context"]
                st.session_state["question"] = sample["question"]

    # ── Header ──
    st.markdown(
        "<h1 style='text-align:center;'>🔍 Question Answering with Transformers</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:gray;'>"
        "Paste a paragraph below. Ask a question. The model will extract the answer.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Inputs ──
    col1, col2 = st.columns([2, 1])

    with col1:
        context = st.text_area(
            "📄 Context Paragraph",
            value=st.session_state.get("context", ""),
            height=250,
            placeholder="Paste your context paragraph here …",
        )

    with col2:
        question = st.text_input(
            "❓ Your Question",
            value=st.session_state.get("question", ""),
            placeholder="Type your question …",
        )
        st.markdown("")
        run_btn = st.button("🚀  Get Answer", type="primary", use_container_width=True)

    # ── Inference & Display ──
    if run_btn:
        if not context.strip():
            st.warning("Please provide a context paragraph.")
            return
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Thinking …"):
            answer, confidence, start, end = predict_answer(
                context, question, tokenizer, model, device
            )

        if not answer:
            st.error("The model could not find an answer in the given context.")
            return

        st.markdown("---")

        # Answer card
        ans_col, conf_col = st.columns([3, 1])
        with ans_col:
            st.markdown("### 💡 Answer")
            st.markdown(
                f"<div style='font-size:1.5rem; font-weight:700; color:#1B5E20; "
                f"background:#E8F5E9; padding:16px 20px; border-radius:10px; "
                f"border-left:5px solid #4CAF50;'>{answer}</div>",
                unsafe_allow_html=True,
            )
        with conf_col:
            st.markdown("### 📊 Confidence")
            # Normalize confidence to roughly 0-1 range (heuristic: sigmoid of score/10)
            norm_conf = 1 / (1 + np.exp(-confidence / 10))
            st.metric("Score", f"{norm_conf:.1%}")
            st.progress(float(norm_conf))

        # Highlighted context
        st.markdown("### 📖 Context with Highlighted Answer")
        st.markdown(highlight_answer(context, start, end), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
