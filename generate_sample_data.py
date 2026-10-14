"""
Generate SYNTHETIC tech job postings datasets for demoing the app.

The numbers produced here are randomly generated and are NOT real market
data. They only exist so the app can be tried without downloading a Kaggle
dataset first.

Four files are written to the data/ folder, each showing off a different
part of the app:

    sample_tech_jobs.csv              Clean data in the app's ideal layout.
    sample_ai_cloud_jobs.csv          Same layout, a different set of roles.
    sample_kaggle_style_salaries.csv  Laid out like the Kaggle "Data Science
                                      Salaries" dataset: coded experience
                                      levels, two salary columns, no skills.
    sample_messy_job_postings.csv     Deliberately untidy: odd column names,
                                      text salaries, mixed-case skills and
                                      missing values.

Usage:
    python generate_sample_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

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

# A second set of roles, for the AI / cloud dataset.
AI_CLOUD_ROLES = {
    "AI Engineer": (
        155_000,
        ["Python", "PyTorch", "LLMs", "Machine Learning", "LangChain", "TensorFlow", "Docker", "AWS", "Vector Databases"],
    ),
    "Cloud Engineer": (
        130_000,
        ["AWS", "Azure", "Terraform", "Kubernetes", "Docker", "Linux", "Python", "Networking", "GCP"],
    ),
    "Cybersecurity Analyst": (
        110_000,
        ["SIEM", "Network Security", "Linux", "Python", "Incident Response", "Penetration Testing", "Splunk", "Firewalls", "Risk Assessment"],
    ),
    "Full Stack Developer": (
        115_000,
        ["JavaScript", "React", "Node.js", "SQL", "Git", "TypeScript", "REST APIs", "MongoDB", "Docker"],
    ),
    "Mobile App Developer": (
        110_000,
        ["Kotlin", "Swift", "Flutter", "React Native", "Git", "REST APIs", "Firebase", "Java", "Dart"],
    ),
    "QA Automation Engineer": (
        90_000,
        ["Selenium", "Python", "Java", "API Testing", "Jira", "CI/CD", "Postman", "SQL", "Cypress"],
    ),
    "Business Intelligence Analyst": (
        95_000,
        ["SQL", "Power BI", "Tableau", "Excel", "Data Modeling", "Python", "DAX", "Snowflake", "Statistics"],
    ),
    "Site Reliability Engineer": (
        140_000,
        ["Linux", "Kubernetes", "AWS", "Python", "Prometheus", "Terraform", "Docker", "Grafana", "Go"],
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

# ---- Kaggle-style dataset --------------------------------------------------

# Roles and base (mid-level, US) salaries for the Kaggle-style dataset.
KAGGLE_ROLES = {
    "Data Scientist": 130_000,
    "Data Engineer": 135_000,
    "Data Analyst": 90_000,
    "Machine Learning Engineer": 150_000,
    "Analytics Engineer": 120_000,
    "Research Scientist": 160_000,
    "Data Architect": 155_000,
}

# Experience code -> (salary multiplier, share of postings)
KAGGLE_EXPERIENCE = {
    "EN": (0.60, 0.20),
    "MI": (0.90, 0.30),
    "SE": (1.25, 0.42),
    "EX": (1.70, 0.08),
}

# Country code -> (currency, units of that currency per USD, pay level
# relative to the US, share of postings). The exchange rates are fixed,
# illustrative values rather than live ones.
KAGGLE_COUNTRIES = {
    "US": ("USD", 1.00, 1.00, 0.55),
    "IN": ("INR", 83.0, 0.30, 0.15),
    "GB": ("GBP", 0.79, 0.75, 0.12),
    "DE": ("EUR", 0.92, 0.80, 0.10),
    "CA": ("CAD", 1.36, 0.85, 0.08),
}

# ---- Messy dataset ---------------------------------------------------------

# Different ways the same experience level gets written in real exports.
MESSY_EXPERIENCE_LABELS = {
    "Junior": ["Junior", "junior", "Entry-level", "EN", "entry level"],
    "Mid": ["Mid", "mid", "Mid-level", "MI"],
    "Senior": ["Senior", "senior", "SE", "SENIOR"],
    "Lead": ["Lead", "lead"],
}


def build_dataset(roles: dict, row_count: int, seed: int) -> pd.DataFrame:
    """Build a clean synthetic job postings DataFrame for the given roles."""
    rng = np.random.default_rng(seed)
    levels = list(EXPERIENCE_LEVELS)
    level_shares = [share for _, share in EXPERIENCE_LEVELS.values()]

    rows = []
    for _ in range(row_count):
        job_title = rng.choice(list(roles))
        base_salary, skill_pool = roles[job_title]
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


def build_kaggle_style_dataset(row_count: int, seed: int) -> pd.DataFrame:
    """Build a dataset with the column layout of Kaggle's "Data Science Salaries"."""
    rng = np.random.default_rng(seed)
    levels = list(KAGGLE_EXPERIENCE)
    level_shares = [share for _, share in KAGGLE_EXPERIENCE.values()]
    countries = list(KAGGLE_COUNTRIES)
    country_shares = [share for *_, share in KAGGLE_COUNTRIES.values()]

    rows = []
    for _ in range(row_count):
        job_title = rng.choice(list(KAGGLE_ROLES))
        level = rng.choice(levels, p=level_shares)
        country = rng.choice(countries, p=country_shares)
        currency, rate, pay_level, _ = KAGGLE_COUNTRIES[country]

        salary_usd = KAGGLE_ROLES[job_title] * KAGGLE_EXPERIENCE[level][0] * pay_level
        salary_usd = round(salary_usd * rng.lognormal(mean=0, sigma=0.18) / 500) * 500

        rows.append(
            {
                "work_year": rng.choice([2023, 2024, 2025, 2026], p=[0.15, 0.25, 0.30, 0.30]),
                "experience_level": level,
                "employment_type": rng.choice(["FT", "PT", "CT", "FL"], p=[0.93, 0.02, 0.03, 0.02]),
                "job_title": job_title,
                # Salary in the local currency - the app should ignore this
                # column and use salary_in_usd instead.
                "salary": round(salary_usd * rate / 100) * 100,
                "salary_currency": currency,
                "salary_in_usd": salary_usd,
                "employee_residence": country,
                "remote_ratio": rng.choice([0, 50, 100], p=[0.35, 0.20, 0.45]),
                "company_location": country,
                "company_size": rng.choice(["S", "M", "L"], p=[0.15, 0.55, 0.30]),
            }
        )
    return pd.DataFrame(rows)


def build_messy_dataset(row_count: int, seed: int) -> pd.DataFrame:
    """Build a deliberately untidy dataset to exercise the app's cleaning step."""
    rng = np.random.default_rng(seed)
    clean = build_dataset(ROLES, row_count, seed)

    def messy_salary(value):
        if pd.isna(value):
            return ""
        roll = rng.random()
        if roll < 0.04:
            return "Not disclosed"
        if roll < 0.06:
            return "0"
        if roll < 0.07:
            return "-1"
        # The same number, written the way different job boards write it.
        style = rng.choice(["plain", "dollar", "usd", "decimal"])
        if style == "plain":
            return f"{value:.0f}"
        if style == "dollar":
            return f"${value:,.0f}"
        if style == "usd":
            return f"USD {value:,.0f}"
        return f"{value:,.2f}"

    def messy_title(title):
        if pd.isna(title):
            return ""
        return rng.choice([title, title, title, f" {title}", f"{title}  "])

    def messy_skills(skills):
        if pd.isna(skills):
            return ""
        names = []
        for name in skills.split(", "):
            roll = rng.random()
            if roll < 0.12:
                name = name.lower()
            elif roll < 0.17:
                name = name.upper()
            names.append(name)
        # Inconsistent separators and the odd trailing comma.
        text = rng.choice([", ", ",", " , "]).join(names)
        return text + "," if rng.random() < 0.05 else text

    return pd.DataFrame(
        {
            "Job Title": clean["job_title"].map(messy_title),
            "Experience": clean["experience_level"].map(lambda level: rng.choice(MESSY_EXPERIENCE_LABELS[level])),
            "Annual Salary": clean["salary_in_usd"].map(messy_salary),
            "Skills": clean["key_skills"].map(messy_skills),
            "Location": clean["company_location"],
        }
    )


if __name__ == "__main__":
    datasets = {
        "sample_tech_jobs.csv": build_dataset(ROLES, row_count=800, seed=42),
        "sample_ai_cloud_jobs.csv": build_dataset(AI_CLOUD_ROLES, row_count=1200, seed=7),
        "sample_kaggle_style_salaries.csv": build_kaggle_style_dataset(row_count=600, seed=21),
        "sample_messy_job_postings.csv": build_messy_dataset(row_count=400, seed=99),
    }

    DATA_DIR.mkdir(exist_ok=True)
    for file_name, dataset in datasets.items():
        dataset.to_csv(DATA_DIR / file_name, index=False)
        print(f"Wrote {len(dataset):>5} rows to {DATA_DIR / file_name}")
