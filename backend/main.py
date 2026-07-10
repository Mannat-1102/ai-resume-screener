"""
FastAPI backend for the AI Resume Screener.

Endpoints:
  POST /score/text   -> score using pasted resume + JD text
  POST /score/files  -> score using uploaded resume + JD files (.pdf, .docx, .txt)
  GET  /health       -> simple health check

Run with:
  uvicorn main:app --reload --port 8000
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from matcher import score_resume, extract_text

app = FastAPI(title="AI Resume Screener API")

# Allow the Streamlit frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextScoreRequest(BaseModel):
    resume_text: str
    jd_text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score/text")
def score_text(payload: TextScoreRequest):
    result = score_resume(payload.resume_text, payload.jd_text)
    return result


@app.post("/score/files")
def score_files(resume_file: UploadFile = File(...), jd_file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmp:
        resume_path = Path(tmp) / resume_file.filename
        jd_path = Path(tmp) / jd_file.filename

        with open(resume_path, "wb") as f:
            shutil.copyfileobj(resume_file.file, f)
        with open(jd_path, "wb") as f:
            shutil.copyfileobj(jd_file.file, f)

        resume_text = extract_text(str(resume_path))
        jd_text = extract_text(str(jd_path))

    result = score_resume(resume_text, jd_text)
    return result
