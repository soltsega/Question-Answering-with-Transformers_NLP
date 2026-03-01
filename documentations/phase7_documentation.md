# Phase VII: Web Miniapp (React + FastAPI) Documentation

## Overview

Phase VII separates the Question Answering model into a robust backend REST API built with **FastAPI**, and pairs it with a beautiful, interactive Single Page Application (SPA) frontend built using **React** and **Vite**.

This decoupled architecture represents standard production patterns, allowing the backend to scale independently of the frontend UI.

## Implementation Details

### 1. API Backend (`web/backend/main.py`)
- **FastAPI**: Provides high-performance async endpoints for serving the inference.
- **Model Caching**: The heavy DistilBERT model is loaded into memory once during the `startup` event, ensuring fast responses.
- **CORS Handling**: Cross-Origin Resource Sharing is enabled via `CORSMiddleware`, permitting the frontend (`localhost:5173`) to make fetch requests.
- **Endpoint `POST /predict`**: Accepts structured `QARequest` JSON containing context and question. Reuses the accurate offset-mapping extraction algorithm to return a `QAResponse` JSON with the extracted answer, normalized confidence, and boundary indices.

### 2. React Frontend (`web/frontend/`)
- **Vite Setup**: Utilized the lightning-fast Vite bundler to scaffold the React architecture.
- **Interactive UI (`App.jsx`)**: 
  - Dynamic React hooks (`useState`) manage user inputs, API results, loading spinners, and error blocks.
  - Asynchronous `fetch` calls are made to the local FastAPI backend.
  - Implements dynamic string manipulation to wrap the exact answer span with HTML `<mark>` tags, rendering highlighted results.
- **Premium Aesthetics (`index.css`)**:
  - Implements a modern color palette, custom CSS variables, flexbox structures, and stylistic hover micro-animations to elevate the "wow" factor of the application interface.
  - Includes a sleek confidence meter represented as an animated progress bar.

## How to Run

Because this is a decoupled architecture, you must run both the backend and frontend simultaneously in separate terminals.

### Terminal 1: Start the Backend
```bash
# From the project root with your virtual environment activated
cd web/backend
uvicorn main:app --reload --port 8000
```
*The API will be available at [http://localhost:8000/docs](http://localhost:8000/docs)*

### Terminal 2: Start the Frontend
```bash
# From the project root
cd web/frontend
npm install   # Only needed the first time
npm run dev
```
*The React application will be available at [http://localhost:5173](http://localhost:5173)*

## Next Steps

With the web application successfully decoupled and deployed locally with a premium UI, the system is fully operational. Phase 8 (Final Documentation) will encapsulate the entire repository's accomplishments into a definitive final report.

---

[< Back to Phase VI](phase6_documentation.md) | [Next: Phase VIII: Final Documentation >](phase8_documentation.md)
