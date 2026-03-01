# Phase V: Evaluation & Error Analysis Documentation

## Overview

Phase V evaluates the fine-tuned DistilBERT model on SQuAD v1.1 using standard QA metrics (Exact Match and F1 Score) and performs comprehensive error analysis to identify performance patterns and failure modes.

All evaluation work is contained in `notebooks/03_evaluation.ipynb`, with supporting source modules in `src/`.

---

## Prerequisites

Before running Phase V, ensure the following are in place:

1. **Virtual environment** activated (`localenv/`)
2. **Dependencies installed** (from `requirements.txt`):
   - `transformers`, `datasets`, `torch`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `tqdm`
3. **Phases I–III completed** (tokenizer, preprocessing pipeline ready)

---

## Step 1: Download the Pre-Trained Model

Since we use the Hugging Face Hub model `distilbert/distilbert-base-uncased-distilled-squad` (already fine-tuned on SQuAD v1.1), we download and save it locally:

```bash
cd "Question Answering with Transformers_NLP"
localenv/Scripts/python.exe src/download_model.py
```

This runs `src/download_model.py`, which:
- Downloads the model and tokenizer from `distilbert/distilbert-base-uncased-distilled-squad`
- Saves them to `models/distilbert-squad-finetuned/`

**Key code** (`src/download_model.py`):
```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

HF_MODEL_NAME = "distilbert/distilbert-base-uncased-distilled-squad"
LOCAL_MODEL_DIR = "models/distilbert-squad-finetuned"

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(HF_MODEL_NAME)
tokenizer.save_pretrained(LOCAL_MODEL_DIR)
model.save_pretrained(LOCAL_MODEL_DIR)
```

### Model Details

| Property | Value |
|----------|-------|
| **Model** | `distilbert-base-uncased-distilled-squad` |
| **Parameters** | 66,364,418 (~66M) |
| **Architecture** | DistilBERT (~40% smaller than BERT-base) |
| **Pre-training** | Distilled from BERT-base, fine-tuned on SQuAD v1.1 |
| **Local Path** | `models/distilbert-squad-finetuned/` |

---

## Step 2: Load Validation Data

The notebook loads the SQuAD v1.1 validation split (10,570 examples total) and selects a 500-example subset for speed:

```python
from datasets import load_dataset

SAMPLE_SIZE = 500
dataset = load_dataset("squad", split="validation")
dataset = dataset.select(range(min(SAMPLE_SIZE, len(dataset))))
```

To evaluate on the **full** validation set, set `SAMPLE_SIZE = None`.

---

## Step 3: Run Inference

For each example, we tokenize the question + context, run the model, and extract the best answer span using offset mapping to map token positions back to the original context characters:

```python
def predict_answer(context, question, tokenizer, model, device, max_length=384):
    inputs = tokenizer(
        question, context,
        truncation="only_second",
        max_length=max_length,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt"
    )

    # Remove offset_mapping and overflow_to_sample_mapping before model forward pass
    offset_mapping = inputs.pop("offset_mapping").numpy()
    inputs.pop("overflow_to_sample_mapping", None)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Select best answer across all chunks (for long contexts)
    best_answer = ""
    best_score = float('-inf')
    for chunk_idx in range(outputs.start_logits.shape[0]):
        start_idx = np.argmax(outputs.start_logits[chunk_idx].cpu().numpy())
        end_idx = np.argmax(outputs.end_logits[chunk_idx].cpu().numpy())
        if end_idx < start_idx:
            end_idx = start_idx
        score = float(start_logits[start_idx] + end_logits[end_idx])
        if score > best_score:
            best_score = score
            offsets = offset_mapping[chunk_idx]
            start_char = offsets[start_idx][0]
            end_char = offsets[end_idx][1]
            best_answer = context[int(start_char):int(end_char)]

    return best_answer.strip(), best_score
```

**Important notes**:
- `offset_mapping` and `overflow_to_sample_mapping` must be **removed** from inputs before passing to the model — DistilBERT does not accept these keys.
- For long contexts that exceed `max_length`, the tokenizer generates multiple chunks via sliding window (`stride=128`). We take the chunk with the highest combined start+end logit score.
- The confidence score is defined as `start_logits[best_start] + end_logits[best_end]`.

---

## Step 4: Compute Metrics

We use two standard QA metrics from `src/evaluation.py`:

### Exact Match (EM)
Checks whether the normalized prediction exactly equals the normalized reference:
```python
def compute_exact_match(a_pred, a_gold):
    return float(normalize_text(a_pred) == normalize_text(a_gold))
```

### F1 Score
Token-level precision/recall between prediction and reference:
```python
def compute_f1(a_pred, a_gold):
    pred_tokens = normalize_text(a_pred).split()
    gold_tokens = normalize_text(a_gold).split()
    common_tokens = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common_tokens.values())
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
```

### Normalization
Both metrics first normalize text by:
1. Converting to lowercase
2. Removing punctuation
3. Removing articles (a, an, the)
4. Collapsing whitespace

### Multi-Reference Scoring
SQuAD provides multiple valid answers per question. We take the **maximum** EM and F1 across all reference answers:
```python
metrics = compute_metrics_multiple_refs(predictions, references)
# references is List[List[str]] — multiple valid answers per example
```

---

## Step 5: Error Analysis

The `ErrorAnalyzer` class in `src/error_analysis.py` performs four types of analysis:

### 5.1 Performance by Question Type
Classifies questions by their first word (what, who, when, where, how, which, etc.) and computes per-type EM/F1.

### 5.2 Performance by Answer Length
Groups reference answers into length bins (1-2 words, 3-5, 6-10, 11-20, 21-50, 50+) and measures performance per bin.

### 5.3 Common Error Patterns
Categorizes errors into:
- **Empty predictions**: Model returned empty string
- **Partial matches**: F1 > 0.5 but EM = 0 (overlapping but imprecise span)
- **Wrong entity**: 0.1 < F1 ≤ 0.5 (some token overlap)
- **Completely wrong**: F1 ≤ 0.1 (no meaningful overlap)

### 5.4 Failure Case Analysis
Returns the N worst examples sorted by F1 score, showing the question, prediction, reference, and context for manual inspection.

---

## Evaluation Results

### Overall Metrics (500-sample validation subset)

| Metric | Score |
|--------|-------|
| **Exact Match (EM)** | **84.4%** |
| **F1 Score** | **88.0%** |
| Perfect matches | 422 / 500 |
| Failures | 78 / 500 |

### F1 Score Statistics

| Statistic | Value |
|-----------|-------|
| Mean | 0.8795 |
| Median | 1.0000 |
| Std Dev | 0.3037 |
| Min | 0.0000 |
| Max | 1.0000 |

### Performance by Question Type

| Question Type | Count | EM Score | F1 Score |
|---------------|-------|----------|----------|
| What | 170 | 0.841 | 0.882 |
| How | 103 | 0.893 | 0.895 |
| Who | 99 | 0.798 | 0.823 |
| Which | 37 | 0.892 | 0.924 |
| Other | 33 | 0.727 | 0.863 |
| When | 27 | **0.963** | **0.988** |
| In | 16 | 0.812 | 0.824 |
| Where | 15 | 0.800 | 0.917 |

### Performance by Answer Length

| Length Bin | Count | EM | F1 | Avg Ref Len | Avg Pred Len |
|------------|-------|------|------|-------------|--------------|
| 1-2 words | 395 | 0.856 | 0.883 | 1.5 | 2.3 |
| 3-5 words | 100 | 0.810 | 0.867 | 3.3 | 3.2 |
| 6-10 words | 4 | 0.500 | 0.810 | 7.0 | 7.0 |
| 11-20 words | 1 | 1.000 | 1.000 | 16.0 | 16.0 |

### Error Patterns (% of all errors)

| Pattern | Percentage |
|---------|------------|
| Completely Wrong | 56.4% |
| Partial Matches | 26.9% |
| Wrong Entity | 12.8% |
| Empty Predictions | 3.8% |

### Confidence Statistics

| Metric | Value |
|--------|-------|
| Mean confidence (correct predictions) | 16.50 |
| Mean confidence (incorrect predictions) | 13.30 |

---

## Key Findings

1. **"When" questions are easiest** (EM=96.3%, F1=98.8%) — temporal answers have highly constrained answer spaces.
2. **"Who" and "Other" question types are hardest** — these often require deeper semantic understanding.
3. **Short answers (1-2 words) dominate** the dataset (79% of examples) and achieve the highest EM scores.
4. **F1 degrades more gracefully** than EM with increasing answer length, confirming that partial span overlap is captured.
5. **Majority of errors are complete misidentifications** (56.4%), not partial span errors — the model either gets it right or picks a completely wrong region.
6. **Model confidence positively correlates** with correctness — correct predictions average 16.50 confidence vs 13.30 for incorrect.

---

## How to Reproduce

### Quick Run (Notebook)

1. Open `notebooks/03_evaluation.ipynb`
2. Run all cells in order
3. Outputs are saved automatically to `plots/` and `docs/`

### Quick Run (Command Line)

```bash
cd "Question Answering with Transformers_NLP"

# 1. Download model (if not already done)
localenv/Scripts/python.exe src/download_model.py

# 2. Run evaluation notebook
localenv/Scripts/python.exe -m jupyter nbconvert \
  --to notebook --execute \
  --ExecutePreprocessor.timeout=600 \
  notebooks/03_evaluation.ipynb \
  --output 03_evaluation.ipynb
```

### Adjusting Sample Size

In the notebook, change `SAMPLE_SIZE` in the data loading cell:
```python
SAMPLE_SIZE = 500   # Current: 500 examples (~4 min on CPU)
SAMPLE_SIZE = None  # Full validation set: 10,570 examples (~50 min on CPU)
```

### Using a Different Model

To evaluate a different model, change `MODEL_DIR` (or `HF_MODEL`) in the model loading cell:
```python
MODEL_DIR = os.path.join('..', 'models', 'your-custom-model')
# or download a different HF model:
HF_MODEL = "deepset/bert-base-uncased-squad2"
```

---

## Generated Artifacts

### Visualizations (`plots/`)
- `score_distributions.png` — EM bar chart + F1 histogram with mean/median lines
- `question_type_performance.png` — EM/F1 grouped bar chart by question type + distribution
- `performance_by_length.png` — EM/F1 grouped bar chart by answer length + count distribution
- `error_patterns.png` — Error pattern pie chart
- `confidence_analysis.png` — Confidence vs F1 scatter plot + confidence distribution by correctness

### Reports (`docs/`)
- `error_analysis_report.txt` — Full text error analysis report
- `evaluation_results.json` — Structured JSON with all metrics, per-type scores, and error patterns

### Source Code (`src/`)
- `evaluation.py` — `normalize_text()`, `compute_f1()`, `compute_exact_match()`, `compute_metrics()`, `compute_metrics_multiple_refs()`
- `error_analysis.py` — `ErrorAnalyzer` class with methods: `get_failure_examples()`, `analyze_answer_length_performance()`, `analyze_question_types()`, `analyze_confidence_patterns()`, `analyze_common_errors()`, `generate_error_report()`, `visualize_performance()`
- `model_evaluator.py` — `QAEvaluator` class (alternative pipeline for programmatic evaluation)
- `download_model.py` — `download_and_save_model()` utility

### Notebook
- `notebooks/03_evaluation.ipynb` — 10 sections covering setup through conclusions

---

## Bug Fixes Applied During This Phase

### `src/evaluation.py`
- **Lines 87, 118**: Multi-line `assert` statements used bare newlines, causing `SyntaxError`. Fixed by collapsing to single lines.

### `src/error_analysis.py`
- **Line 92**: `analyze_answer_length_performance()` iterated `self.references` (a `List[List[str]]`) but called `.split()` directly on each element. Fixed to index `refs[0].split()`.
- **Lines 266, 273**: `generate_error_report()` accessed `stats['f1']` but the dict key from `analyze_answer_length_performance()` and `analyze_question_types()` is `'f1_score'`. Fixed to `stats['f1_score']`.

---

## Conclusion

Phase V successfully demonstrates that the DistilBERT model achieves strong performance on SQuAD v1.1:
- **EM of 84.4%** and **F1 of 88.0%** on a 500-sample validation subset
- These results are consistent with published benchmarks for `distilbert-base-uncased-distilled-squad` (expected EM ~77-79%, F1 ~86-87% on the full validation set; our subset may differ slightly)
- Error analysis reveals clear patterns: temporal questions are easiest, short answers perform best, and most errors are complete misidentifications rather than partial span errors
- The evaluation pipeline is fully reproducible via the notebook or command line

---

**Next Phase**: [Phase VI: Deployment (Streamlit App)](phase6_documentation.md)
