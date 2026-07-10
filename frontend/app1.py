"""
Streamlit UI for the AI Resume Screener.

Run with:
  streamlit run app.py

Expects the FastAPI backend to be running at BACKEND_URL (default localhost:8000).
"""

import streamlit as st
import requests
import os

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

st.title("📄 AI Resume Screener")
st.caption(
    "Paste or upload a resume and a job description to get a match score, "
    "sub-scores, and a plain-English explanation of the result."
)

with st.expander("ℹ️ How scoring works"):
    st.markdown(
        """
        - **Skills match (40%)** — overlap between skills mentioned in the resume and the JD, from a curated skills bank
        - **Semantic similarity (35%)** — sentence-embedding cosine similarity between the full resume and JD text
        - **Experience match (15%)** — years of experience mentioned in the resume vs. years implied by the JD
        - **Education match (10%)** — degree level mentioned in the resume vs. degree level implied by the JD

        Before scoring, obvious PII-like signals (emails, phone numbers, graduation years, addresses)
        are stripped from both documents — a lightweight fairness safeguard so scoring focuses on
        qualifications rather than incidental personal details.
        """
    )

input_mode = st.radio("Input method", ["Paste text", "Upload files"], horizontal=True)

col1, col2 = st.columns(2)

resume_text, jd_text = "", ""
resume_file, jd_file = None, None

if input_mode == "Paste text":
    with col1:
        resume_text = st.text_area("Resume text", height=300, placeholder="Paste resume text here...")
    with col2:
        jd_text = st.text_area("Job description text", height=300, placeholder="Paste job description here...")
else:
    with col1:
        resume_file = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"])
    with col2:
        jd_file = st.file_uploader("Upload job description", type=["pdf", "docx", "txt"])

if st.button("Score match", type="primary"):
    with st.spinner("Scoring... (if the backend has been idle, this can take up to a minute while it wakes up)"):
        try:
            if input_mode == "Paste text":
                if not resume_text.strip() or not jd_text.strip():
                    st.warning("Please provide both resume and job description text.")
                    st.stop()
                resp = requests.post(
                    f"{BACKEND_URL}/score/text",
                    json={"resume_text": resume_text, "jd_text": jd_text},
                    timeout=60,
                )
            else:
                if not resume_file or not jd_file:
                    st.warning("Please upload both files.")
                    st.stop()
                files = {
                    "resume_file": (resume_file.name, resume_file.getvalue()),
                    "jd_file": (jd_file.name, jd_file.getvalue()),
                }
                resp = requests.post(f"{BACKEND_URL}/score/files", files=files, timeout=60)

            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.ConnectionError:
            st.error(
                "Could not reach the backend. Make sure it's running: "
                "`uvicorn main:app --reload --port 8000` from the backend/ folder."
            )
            st.stop()
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.divider()

    overall = result["overall_score"]
    st.metric("Overall Match Score", f"{overall}%")
    st.progress(min(int(overall), 100))

    sub = result["sub_scores"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Skills Match", f"{sub['skills_match']}%")
    m2.metric("Semantic Similarity", f"{sub['semantic_similarity']}%")
    m3.metric("Experience Match", f"{sub['experience_match']}%")
    m4.metric("Education Match", f"{sub['education_match']}%")

    st.subheader("Explanation")
    st.write(result["explanation"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**✅ Matched skills**")
        if result["matched_skills"]:
            st.write(", ".join(result["matched_skills"]))
        else:
            st.write("None found")
    with c2:
        st.markdown("**⚠️ Missing skills (from JD)**")
        if result["missing_skills"]:
            st.write(", ".join(result["missing_skills"]))
        else:
            st.write("None — full coverage")
