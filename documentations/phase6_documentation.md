# Phase VI: Deployment (Streamlit App) Documentation

## Overview

Phase VI focuses on deploying the fine-tuned DistilBERT Question Answering model via an interactive web application built with Streamlit. This allows users to easily interact with the model by providing a context paragraph and a question to extract answers.

## Implementation Details

### Application Architecture

The application is implemented as a single-page Streamlit app in `app/app.py`.

#### Key Components:
1. **Model Management (`load_model`)**:
   - The DistilBERT tokenizer and model are loaded from the local `models/distilbert-squad-finetuned/` directory.
   - If the local model is missing, it automatically falls back to downloading `distilbert/distilbert-base-uncased-distilled-squad` from the Hugging Face Hub.
   - The model is cached using `@st.cache_resource` to prevent reloading during UI re-renders, significantly improving responsiveness.
   - Inference runs automatically on GPU if available (`cuda`), otherwise falls back to `cpu`.

2. **Inference Logic (`predict_answer`)**:
   - Reuses the robust offset-mapping extraction logic developed in Phase V.
   - Handles long contexts through chunking (sliding window with `stride=128`).
   - Returns the extracted answer string, a raw confidence score (sum of start and end logits), and the character start/end positions for highlighting.

3. **User Interface Features**:
   - **Sample Questions**: A sidebar provides 1-click sample contexts and questions (Super Bowl 50, Amazon Rainforest, DNA Structure) to let users immediately test the app.
   - **Interactive Inputs**: Wide text areas for pasting custom context and typing questions.
   - **Rich Results Display**: 
     - Extracted answer prominently displayed in a styled success box.
     - Confidence score shown both numerically and as a progress bar (normalized via sigmoid curve for intuitiveness).
     - **Context Highlighting**: The original context is displayed with the exact answer span highlighted in yellow HTML `<mark>` tags, making it easy to verify the model's extraction in context.

## How to Run

1. Ensure the virtual environment is activated:
   ```bash
   localenv\Scripts\activate
   ```

2. Start the Streamlit server:
   ```bash
   python -m streamlit run app/app.py
   ```

3. Open the provided `localhost` URL (usually `http://localhost:8501`) in your browser.

## Next Steps

With the local Streamlit application complete, the final phase (Phase VII) will decouple the frontend and backend by building a FastAPI REST API for the model and creating a React/Next.js "Miniapp" interface.

---

[< Back to Phase V](phase5_documentation.md) | [Next: Phase VII: Web Miniapp >](phase7_documentation.md)
