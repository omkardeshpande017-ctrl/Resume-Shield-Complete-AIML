# Resume Shield — Complete AI Resume Scanner + Analyzer

A working MERN-style full-stack application with React frontend, FastAPI backend, MongoDB, PDF/DOCX parsing, local NLP/ML scoring, ATS analysis, JD matching, integrity checks, scam checks, improvement suggestions, dashboard/history, and resume generation.

## Requirements
- macOS/Linux
- Python 3.11+ (3.13 supported)
- Node.js 18+
- Docker Desktop

## Terminal 1 — MongoDB
```bash
cd ~/Downloads/resume-shield-complete
docker compose up -d mongo
```

## Terminal 2 — Backend
```bash
cd ~/Downloads/resume-shield-complete/backend
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend: http://localhost:8000
Swagger: http://localhost:8000/docs

## Terminal 3 — Frontend
```bash
cd ~/Downloads/resume-shield-complete/frontend
npm install
npm run dev
```

Website: http://localhost:5173

## First use
1. Register an account.
2. Upload a PDF or DOCX resume.
3. Open the analysis dashboard.
4. Paste a job description and run JD Match.
5. Run Integrity and Scam Shield checks.
6. Review recommendations and generate an ATS-friendly resume.

The ML layer works locally without an API key. It uses TF-IDF/cosine similarity, curated skill extraction, deterministic ATS scoring, and explainable rules. Optional LLM integration can be added later.
