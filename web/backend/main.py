import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# ── Paths & Config ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models", "distilbert-squad-finetuned")
HF_MODEL = "distilbert/distilbert-base-uncased-distilled-squad"

app = FastAPI(
    title="QA with Transformers API",
    description="A FastAPI backend serving a DistilBERT Question Answering model.",
    version="1.0.0"
)

# ── CORS Middleware ──────────────────────────────────────────────────────────
# Allow all origins for local development (Vite frontend on anything)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# ── Global Model State ───────────────────────────────────────────────────────
model = None
tokenizer = None
device = None

# ── Pydantic Models ──────────────────────────────────────────────────────────
class QARequest(BaseModel):
    context: str
    question: str

class QAResponse(BaseModel):
    answer: str
    confidence: float
    start_char: int
    end_char: int

# ── Startup/Shutdown ─────────────────────────────────────────────────────────
@app.on_event("startup")
def load_model():
    """Load model once when the FastAPI server starts up."""
    global model, tokenizer, device
    print("Loading QA model...", flush=True)
    try:
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
        print(f"Model loaded successfully on {device}!", flush=True)
    except Exception as e:
        print(f"Failed to load model: {e}", flush=True)

# ── Inference Engine ─────────────────────────────────────────────────────────
def predict_answer_span(context: str, question: str):
    """Run prediction. Reuse robust logic from evaluation phase."""
    inputs = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=384,
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

    num_chunks = outputs.start_logits.shape[0]
    for chunk_idx in range(num_chunks):
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
                if s is not None and e is not None and e > s:
                    best_answer = context[s:e]
                    best_start_char = s
                    best_end_char = e

    return best_answer.strip(), best_score, best_start_char, best_end_char


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "QA Model Backend API is running. Use POST /predict to run inference."}

@app.post("/predict", response_model=QAResponse)
def predict(request: QARequest):
    """
    Given a context paragraph and a question, returns the extracted answer span, 
    confidence score (raw logit sum), and the char start and end indices.
    """
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model is still loading or failed to load. Please try again.")
    if not request.context.strip() or not request.question.strip():
        raise HTTPException(status_code=400, detail="Context and question cannot be empty.")

    try:
        answer, confidence, start, end = predict_answer_span(request.context, request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return QAResponse(
        answer=answer,
        confidence=confidence,
        start_char=start,
        end_char=end
    )
