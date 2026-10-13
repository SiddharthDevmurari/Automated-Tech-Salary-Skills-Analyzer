"""
Generate a SYNTHETIC tech job postings dataset for demoing the app.

The numbers produced here are randomly generated and are NOT real market
data. They only exist so the app can be tried without downloading a Kaggle
dataset first.

Usage:
    python generate_sample_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PATH = Path(__file__).parent / "data" / "sample_tech_jobs.csv"
ROW_COUNT = 800
SEED = 42

# Base (mid-level) salary in USD and the skill pool for every role.
# Skills are listed from most to least commonly requested.
ROLES = {
    "Data Scientist": (
        125_000,
        ["Python", "SQL", "Machine Learning", "Statistics", "Pandas", "TensorFlow", "Tableau", "AWS", "Spark"],
    ),
    "Data Analyst": (
        85_000,
        ["SQL", "Excel", "Python", "Tableau", "Power BI", "Statistics", "Pandas", "R"],
    ),
    "Machine Learning Engineer": (
        145_000,
        ["Python", "PyTorch", "TensorFlow", "Machine Learning", "AWS", "Docker", "Kubernetes", "SQL", "MLOps"],
    ),
    "Data Engineer": (
        130_000,
        ["SQL", "Python", "Spark", "AWS", "Airflow", "Kafka", "Snowflake", "Docker", "Scala"],
    ),
    "Software Engineer": (
        120_000,
        ["Java", "Python", "Git", "SQL", "Docker", "AWS", "JavaScript", "System Design", "C++"],
    ),
    "DevOps Engineer": (
        125_000,
        ["AWS", "Docker", "Kubernetes", "Terraform", "Linux", "CI/CD", "Python", "Jenkins", "Ansible"],
    ),
    "Frontend Developer": (
        100_000,
        ["JavaScript", "React", "HTML", "CSS", "TypeScript", "Git", "Next.js", "Redux", "Figma"],
    ),
    "Backend Developer": (
        115_000,
        ["Java", "SQL", "Python", "Node.js", "REST APIs", "Docker", "AWS", "MongoDB", "Git"],
    ),
}

# Experience level -> (salary multiplier, share of postings)
EXPERIENCE_LEVELS = {
    "Junior": (0.65, 0.30),
    "Mid": (1.00, 0.35),
    "Senior": (1.35, 0.25),
    "Lead": (1.70, 0.10),
}

LOCATIONS = ["United States", "India", "United Kingdom", "Germany", "Canada"]
WORK_MODES = ["On-site", "Hybrid", "Remote"]


def build_dataset(row_count: int, seed: int) -> pd.DataFrame:
    """Build the synthetic job postings DataFrame."""
    rng = np.random.default_rng(seed)
    levels = list(EXPERIENCE_LEVELS)
    level_shares = [share for _, share in EXPERIENCE_LEVELS.values()]

    rows = []
    for _ in range(row_count):
        job_title = rng.choice(list(ROLES))
        base_salary, skill_pool = ROLES[job_title]
        level = rng.choice(levels, p=level_shares)

        # Log-normal noise gives the right-skewed shape real salaries have.
        salary = base_salary * EXPERIENCE_LEVELS[level][0] * rng.lognormal(mean=0, sigma=0.15)

        # Earlier skills in the pool are picked more often than later ones.
        weights = np.linspace(1.0, 0.25, len(skill_pool))
        skills = rng.choice(
            skill_pool,
            size=rng.integers(3, 7),
            replace=False,
            p=weights / weights.sum(),
        )

        rows.append(
            {
                "job_title": job_title,
                "experience_level": level,
                "salary_in_usd": round(salary / 500) * 500,
                "key_skills": ", ".join(skills),
                "company_location": rng.choice(LOCATIONS),
                "work_mode": rng.choice(WORK_MODES),
            }
        )

    df = pd.DataFrame(rows)

    # Knock out a few values so the app's data cleaning step has work to do.
    df.loc[rng.random(row_count) < 0.03, "salary_in_usd"] = np.nan
    df.loc[rng.random(row_count) < 0.01, "job_title"] = np.nan
    df.loc[rng.random(row_count) < 0.02, "key_skills"] = np.nan
    return df


if __name__ == "__main__":
    dataset = build_dataset(ROW_COUNT, SEED)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(dataset)} rows to {OUTPUT_PATH}")
