# AI Resume Screener

An end-to-end NLP tool that scores how well a resume matches a job description —
built to demonstrate a full ML/analytics pipeline: text extraction, feature
engineering, semantic embeddings, explainable scoring, and a deployed interface.

## Why this project

Most "resume matcher" tutorials do naive keyword counting. This one goes further:

- **Semantic similarity via sentence embeddings** (not just keyword overlap) —
  understands that "led a team" and "managed direct reports" are related ideas
- **Sub-scores with explanations**, not one opaque number — skills, semantic
  similarity, experience, and education are scored and reported separately
- **A basic fairness safeguard** — emails, phone numbers, graduation years, and
  addresses are stripped before scoring, so results aren't influenced by
  incidental personal details unrelated to qualifications
- **Real architecture** — a FastAPI backend serving predictions, and a
  Streamlit frontend for interactive use, mirroring how ML tools are actually
  deployed for internal use at companies

## Architecture

```
resume_screener/
├── backend/
│   ├── main.py           # FastAPI app: /score/text and /score/files endpoints
│   ├── matcher.py         # Core logic: parsing, redaction, skills, embeddings, scoring
│   └── requirements.txt
├── frontend/
│   ├── app.py             # Streamlit UI, calls the backend API
│   └── requirements.txt
├── sample_data/
│   ├── sample_resume.txt
│   └── sample_jd.txt
└── README.md
```

## How scoring works

| Component | Weight | What it measures |
|---|---|---|
| Skills match | 40% | Overlap of skills (from a curated bank) found in resume vs. JD |
| Semantic similarity | 35% | Cosine similarity between sentence embeddings of the full texts |
| Experience match | 15% | Years of experience mentioned vs. years implied by the JD |
| Education match | 10% | Degree level mentioned vs. degree level implied by the JD |

The embedding model is `all-MiniLM-L6-v2` via `sentence-transformers` — small,
fast, and good enough for this use case without needing a GPU.

## Setup

Requires Python 3.9+.

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The first run will download the embedding model (~90MB) — this requires
internet access and takes a minute or two. After that it's cached locally.

Leave this running. You can test it directly at `http://localhost:8000/docs`
(FastAPI's auto-generated interactive docs).

### 2. Frontend

In a **second terminal**:

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

This opens the UI in your browser (usually `http://localhost:8501`). Paste in
the sample resume/JD from `sample_data/`, or upload your own `.pdf`, `.docx`,
or `.txt` files.

## Example result

Using `sample_data/sample_resume.txt` against `sample_data/sample_jd.txt`
produces something like:

```
Overall Match Score: 72.3%
  Skills Match:          46.2%
  Semantic Similarity:   ~80%   (varies slightly by embedding model version)
  Experience Match:      100%
  Education Match:       100%

Matched skills: numpy, pandas, python, sql, statistics, tableau
Missing skills: aws, data analysis, gcp, machine learning, nlp, power bi
```

## Validating it against human judgment (recommended next step)

To turn this from "a tool that runs" into "a tool with a measured result" —
the detail that stands out most on a resume — gather 20-30 resume/JD pairs,
have 2-3 people rank them independently, and compute the correlation between
your tool's `overall_score` and the human rankings.

This repo includes a ready-made script for this:

```bash
cd resume_screener  # run from project root, not from scripts/
pip install scipy
```

1. Copy `scripts/validation_data_template.csv` and fill in rows of
   `resume_path,jd_path,human_score` — paths relative to the project root,
   `human_score` a 0-100 rating a person gave that pair (average 2-3 raters
   per pair if you can)
2. Run:
   ```bash
   python scripts/validate.py scripts/your_validation_data.csv
   ```
3. It prints a Spearman correlation and a ready-to-use resume sentence, e.g.:
   > "Achieved a Spearman correlation of 0.74 with human recruiter rankings
   > across 24 resume-job pairs."

Aim for at least ~15-20 pairs — fewer than that and the correlation number
is too noisy to mean much.


## Deployment (get a live link, not just a repo)

This is the step that makes the project *demoable* in an interview instead of
just readable as code.

### 1. Push to GitHub first

```bash
cd resume_screener
git remote add origin https://github.com/YOUR_USERNAME/ai-resume-screener.git
git branch -M main
git push -u origin main
```

### 2. Deploy the backend (Render, free tier)

1. Go to [render.com](https://render.com) → New → Blueprint → connect your repo
   (it will detect `render.yaml` automatically and configure everything)
2. Alternatively, without the blueprint: New → Web Service → connect repo →
   set **root directory** to `backend`, **build command** to
   `pip install -r requirements.txt`, **start command** to
   `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Deploy. Note the resulting URL, e.g. `https://resume-screener-api.onrender.com`
4. Sanity check it: visit `https://resume-screener-api.onrender.com/health` —
   should return `{"status": "ok"}`

Note: Render's free tier spins down after inactivity, so the first request
after idle time takes ~30-50 seconds to wake up. Worth mentioning if you demo
it live.

### 3. Deploy the frontend (Streamlit Community Cloud, free)

1. Go to [share.streamlit.io](https://share.streamlit.io) → New app → connect
   your repo
2. Set **main file path** to `frontend/app.py`
3. Under **Advanced settings → Secrets**, add:
   ```
   BACKEND_URL = "https://resume-screener-api.onrender.com"
   ```
4. Deploy. You'll get a public URL like
   `https://your-app-name.streamlit.app` — this is the link to put on your
   resume/LinkedIn.

You'll need to set the `BACKEND_URL` environment variable to read from
Streamlit secrets — this is already wired up in `frontend/app.py` via
`os.environ.get("BACKEND_URL", "http://localhost:8000")`, and Streamlit Cloud
automatically injects values from the Secrets panel as environment variables.



- Swap the curated skills bank for a proper skill-extraction model (e.g. spaCy NER)
- Add SHAP-style explanations for *why* the semantic score landed where it did
- Deploy the backend + frontend (Render/Railway for the API, Streamlit
  Community Cloud for the UI) so the project has a live link, not just a repo
- Expand the fairness check into a proper audit — e.g. testing whether
  resumes with employment gaps score systematically lower

## Resume bullet

> Built and deployed an NLP-based resume-to-job-description matching tool
> using sentence embeddings, FastAPI, and Streamlit; produced explainable
> sub-scores (skills, semantic similarity, experience, education) and a
> fairness safeguard that redacts PII-like signals prior to scoring.
