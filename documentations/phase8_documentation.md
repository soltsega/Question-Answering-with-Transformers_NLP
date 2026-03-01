# Phase VIII: Final Documentation & Project Sign-Off

## Overview

Phase VIII wraps up the **Question Answering with Transformers** project. This phase involved conducting a final code cleanup, polishing the repository's main `README.md`, and writing this final architectural summary of the resulting system. 

By successfully pushing the final UI advancements and code refactoring to the `main` branch, the live **Streamlit Community Cloud** application automatically fetched the latest context and redeployed seamlessly.

## Project Summary

The objective of this project was to build a state-of-the-art NLP pipeline capable of extracting precise answers from a provided context paragraph, mimicking reading comprehension. Over the course of 8 phases, we accomplished:

1. **Environment & Data Setup**: Validated dependencies and handled raw SQuAD v1.1 subsets.
2. **Model Selection**: Leveraged `distilbert-base-uncased-distilled-squad` as a lightweight, performant extraction model (~66M parameters).
3. **Robust Evaluation**: Evaluated the model against a 500-sample validation subset.
   - **Exact Match (EM)**: 84.4%
   - **F1 Score**: 88.0%
4. **Error Analysis**: Confirmed high model precision on temporal ("When") questions (96% EM) and identified that the majority of failure patterns were complete misses rather than partial overlaps.
5. **Interactive UI (Streamlit)**: Built a primary web app in `app/app.py` allowing users to paste contexts and dynamically test questions with rich HTML span highlighting. Deployed live!
6. **Decoupled Miniapp (React + FastAPI)**: To demonstrate scalable architecture, the backend inference logic was decoupled into a FastAPI service (`web/backend`), and paired with a modern Vite + React frontend (`web/frontend`) styled with premium CSS and animated interactions.

## Deployment Architectures

This repository supports **two distinct inference architectures**:

### 1. The Monolith (Streamlit)
- **Codebase**: `app/app.py`
- **Use Case**: Rapid prototyping, data science presentations, instantaneous cloud deployment.
- **Status**: **LIVE** on Streamlit Community Cloud. 

### 2. The Decoupled Miniapp (FastAPI + React)
- **Codebase**: `web/backend/` and `web/frontend/`
- **Use Case**: Production-ready scalable SPA deployed behind load balancers.
- **Status**: Run locally. The `FastAPI` instance serves via `uvicorn` and handles Python inference logic asynchronously, accessed via REST (`POST /predict`) by the `Vite/React` frontend which handles all DOM manipulations.

## Conclusion

This project successfully proves that combining high-quality pre-trained transformer weights (DistilBERT) with precise span mapping via offset tokens yields incredibly accurate Extractive QA engines. The model generalizes well, runs inference in milliseconds (even on CPUs), and can be packaged elegantly for web deployment.

---
**Project Status:** COMPLETE 🏁
