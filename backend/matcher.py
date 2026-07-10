"""
Core logic for the AI Resume Screener.

Pipeline:
1. Extract raw text from resume/JD (pdf, docx, or plain text)
2. Redact PII-ish signals before scoring (fairness safeguard)
3. Extract candidate "skills" via keyword matching against a curated skills bank
4. Compute a semantic similarity score via sentence embeddings
5. Compute sub-scores (skills, semantic, experience, education)
6. Combine into an overall score + human-readable explanation
"""

import re
from functools import lru_cache

# -----------------------------
# 1. Text extraction
# -----------------------------

def extract_text_from_pdf(file_path: str) -> str:
    import pdfplumber
    text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
    return "\n".join(text)


def extract_text_from_docx(file_path: str) -> str:
    import docx
    doc = docx.Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text(file_path: str) -> str:
    """Dispatch based on file extension. Falls back to plain read for .txt."""
    lower = file_path.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif lower.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


# -----------------------------
# 2. Fairness safeguard: redact PII-ish signals before scoring
# -----------------------------
# Rationale: names, ages, graduation years, and addresses can introduce bias
# unrelated to qualifications. We strip obvious patterns before the text
# ever reaches the scoring model. This is a lightweight heuristic, not a
# guarantee of full anonymization — worth stating plainly in interviews.

PII_PATTERNS = [
    (re.compile(r"\b\d{4,}\s?[-–]\s?\d{4,}\b"), " "),          # date ranges like 1990-1994 (grad years)
    (re.compile(r"\b(19|20)\d{2}\b"), " "),                      # standalone years (birth/grad year signals)
    (re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"), " "),    # phone numbers
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), " "),             # emails
    (re.compile(r"\b\d{1,5}\s+\w+(\s\w+){0,3}\s(street|st|ave|avenue|road|rd|blvd)\b", re.I), " "),  # street address
]


def redact_pii(text: str) -> str:
    redacted = text
    for pattern, replacement in PII_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


# -----------------------------
# 3. Skills extraction (curated keyword bank + simple matching)
# -----------------------------

SKILLS_BANK = [
    "python", "r", "sql", "java", "javascript", "c++", "scala",
    "pandas", "numpy", "scikit-learn", "sklearn", "pytorch", "tensorflow", "keras",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "data analysis", "data visualization", "statistics", "statistical modeling",
    "excel", "tableau", "power bi", "looker",
    "aws", "gcp", "azure", "docker", "kubernetes",
    "spark", "hadoop", "airflow", "etl",
    "regression", "classification", "clustering", "a/b testing",
    "git", "linux", "fastapi", "flask", "django",
    "communication", "leadership", "project management", "stakeholder management",
]


def extract_skills(text: str) -> set:
    text_lower = text.lower()
    found = set()
    for skill in SKILLS_BANK:
        escaped = re.escape(skill)
        # \b doesn't work reliably around non-word chars (c++, a/b testing),
        # so use explicit boundary checks: start/end of string or non-alphanumeric neighbor.
        pattern = r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found


# -----------------------------
# 4. Semantic similarity via sentence embeddings
# -----------------------------

@lru_cache(maxsize=1)
def get_model():
    """Load the embedding model once and cache it (first call downloads weights)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Returns cosine similarity scaled to 0-100."""
    from sentence_transformers import util
    model = get_model()
    emb_a = model.encode(text_a, convert_to_tensor=True)
    emb_b = model.encode(text_b, convert_to_tensor=True)
    score = util.cos_sim(emb_a, emb_b).item()
    # cos_sim is roughly -1..1; clamp and rescale to 0..100
    score = max(0.0, min(1.0, (score + 1) / 2))
    return round(score * 100, 1)


# -----------------------------
# 5. Experience & education heuristics
# -----------------------------

def extract_years_experience(text: str) -> int:
    """Finds patterns like '5 years of experience' and returns the max found (0 if none)."""
    matches = re.findall(r"(\d{1,2})\+?\s*(?:years|yrs)", text.lower())
    years = [int(m) for m in matches]
    return max(years) if years else 0


EDUCATION_LEVELS = {
    "phd": 4, "doctorate": 4,
    "master": 3, "msc": 3, "m.s.": 3, "mba": 3,
    "bachelor": 2, "bsc": 2, "b.s.": 2, "b.a.": 2,
    "associate": 1,
}


def extract_education_level(text: str) -> int:
    text_lower = text.lower()
    level = 0
    for keyword, rank in EDUCATION_LEVELS.items():
        if keyword in text_lower and rank > level:
            level = rank
    return level


# -----------------------------
# 6. Combine into overall score + explanation
# -----------------------------

WEIGHTS = {
    "skills": 0.40,
    "semantic": 0.35,
    "experience": 0.15,
    "education": 0.10,
}


def score_resume(resume_text: str, jd_text: str) -> dict:
    # Fairness step: redact PII-ish content before any scoring happens
    resume_clean = redact_pii(resume_text)
    jd_clean = redact_pii(jd_text)

    resume_skills = extract_skills(resume_clean)
    jd_skills = extract_skills(jd_clean)

    matched_skills = sorted(resume_skills & jd_skills)
    missing_skills = sorted(jd_skills - resume_skills)

    skills_score = (len(matched_skills) / len(jd_skills) * 100) if jd_skills else 0.0

    semantic_score = semantic_similarity(resume_clean, jd_clean)

    resume_years = extract_years_experience(resume_clean)
    jd_years_required = extract_years_experience(jd_clean)
    if jd_years_required == 0:
        experience_score = 100.0  # JD didn't specify a requirement
    else:
        experience_score = min(100.0, (resume_years / jd_years_required) * 100)

    resume_edu = extract_education_level(resume_clean)
    jd_edu = extract_education_level(jd_clean)
    if jd_edu == 0:
        education_score = 100.0
    else:
        education_score = 100.0 if resume_edu >= jd_edu else (resume_edu / jd_edu) * 100

    overall = (
        skills_score * WEIGHTS["skills"]
        + semantic_score * WEIGHTS["semantic"]
        + experience_score * WEIGHTS["experience"]
        + education_score * WEIGHTS["education"]
    )

    explanation = build_explanation(
        matched_skills, missing_skills, resume_years, jd_years_required,
        resume_edu, jd_edu, overall
    )

    return {
        "overall_score": round(overall, 1),
        "sub_scores": {
            "skills_match": round(skills_score, 1),
            "semantic_similarity": round(semantic_score, 1),
            "experience_match": round(experience_score, 1),
            "education_match": round(education_score, 1),
        },
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "resume_years_experience": resume_years,
        "jd_years_required": jd_years_required,
        "explanation": explanation,
    }


LEVEL_NAMES = {0: "unspecified", 1: "associate degree", 2: "bachelor's degree", 3: "master's degree", 4: "phd/doctorate"}


def build_explanation(matched, missing, resume_years, jd_years, resume_edu, jd_edu, overall) -> str:
    lines = []
    if overall >= 80:
        lines.append("Strong overall match.")
    elif overall >= 60:
        lines.append("Moderate match — worth a closer look.")
    else:
        lines.append("Weak match against this job description.")

    if matched:
        lines.append(f"Matched skills: {', '.join(matched)}.")
    else:
        lines.append("No direct skill overlap found from the curated skills bank.")

    if missing:
        lines.append(f"Missing skills the JD mentions: {', '.join(missing)}.")

    if jd_years > 0:
        lines.append(f"JD implies ~{jd_years}+ years experience; resume indicates ~{resume_years} years.")

    if jd_edu > 0:
        lines.append(f"JD implies {LEVEL_NAMES[jd_edu]}; resume indicates {LEVEL_NAMES[resume_edu]}.")

    return " ".join(lines)
