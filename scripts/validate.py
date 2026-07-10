"""
Validate the resume screener against human judgment.

Usage:
    1. Fill in `validation_data.csv` with rows of:
         resume_path, jd_path, human_score
       where human_score is a 0-100 rating a person gave that resume/JD pair
       (get 2-3 people to rate independently and average their scores for
       more reliable numbers).
    2. Run:
         python validate.py validation_data.csv

    Requires scipy: pip install scipy
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from matcher import score_resume, extract_text  # noqa: E402


def load_pairs(csv_path: str):
    pairs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(
                (row["resume_path"].strip(), row["jd_path"].strip(), float(row["human_score"]))
            )
    return pairs


def main(csv_path: str):
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("This script needs scipy. Install it with: pip install scipy")
        sys.exit(1)

    pairs = load_pairs(csv_path)
    if len(pairs) < 5:
        print(f"Warning: only {len(pairs)} pairs found. Aim for 20-30 for a meaningful correlation.")

    tool_scores = []
    human_scores = []
    rows = []

    for resume_path, jd_path, human_score in pairs:
        resume_text = extract_text(resume_path)
        jd_text = extract_text(jd_path)
        result = score_resume(resume_text, jd_text)
        tool_score = result["overall_score"]

        tool_scores.append(tool_score)
        human_scores.append(human_score)
        rows.append((Path(resume_path).name, Path(jd_path).name, tool_score, human_score))

    corr, p_value = spearmanr(tool_scores, human_scores)

    print("\n--- Per-pair results ---")
    print(f"{'Resume':<25} {'JD':<25} {'Tool':>8} {'Human':>8}")
    for resume_name, jd_name, tool_score, human_score in rows:
        print(f"{resume_name:<25} {jd_name:<25} {tool_score:>8.1f} {human_score:>8.1f}")

    print("\n--- Summary ---")
    print(f"Pairs evaluated:        {len(pairs)}")
    print(f"Spearman correlation:   {corr:.3f}")
    print(f"P-value:                {p_value:.4f}")

    if corr >= 0.7:
        verdict = "Strong agreement with human judgment."
    elif corr >= 0.4:
        verdict = "Moderate agreement — reasonable, with room to improve (e.g. expand the skills bank)."
    else:
        verdict = "Weak agreement — worth revisiting weightings or skill coverage."
    print(f"Verdict:                {verdict}")

    print(
        "\nCite this in your resume/portfolio, e.g.:\n"
        f'  "Achieved a Spearman correlation of {corr:.2f} with human recruiter rankings '
        f'across {len(pairs)} resume-job pairs."'
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate.py <path_to_validation_csv>")
        sys.exit(1)
    main(sys.argv[1])
