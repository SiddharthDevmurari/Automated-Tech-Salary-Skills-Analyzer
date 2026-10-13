"""
Automated Tech Salary & Skills Analyzer
=======================================

A Streamlit web app that:
    1. Accepts a CSV of tech job postings (salaries + skills).
    2. Cleans the data with Pandas.
    3. Computes descriptive statistics with Pandas and SciPy.
    4. Visualises salary distributions and skill demand with Seaborn/Matplotlib.
    5. Uses an LLM (through LangChain LCEL) to write a "Career Action Plan"
       grounded in the computed statistics.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from matplotlib.ticker import FuncFormatter
from scipy import stats

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SAMPLE_DATA_PATH = Path(__file__).parent / "data" / "sample_tech_jobs.csv"

# Kaggle datasets name their columns differently, so each column the app
# needs is matched against a list of common alternatives (in priority order).
COLUMN_ALIASES = {
    "job_title": ["job_title", "title", "job", "role", "position", "designation"],
    "salary_in_usd": ["salary_in_usd", "salary_usd", "salary", "annual_salary", "avg_salary"],
    "experience_level": ["experience_level", "experience", "seniority", "level"],
    "key_skills": ["key_skills", "skills", "required_skills", "skill", "tech_stack"],
}
REQUIRED_COLUMNS = ["job_title", "salary_in_usd"]

# Short codes used by popular Kaggle salary datasets -> readable labels.
EXPERIENCE_LABELS = {
    "en": "Junior",
    "entry": "Junior",
    "entry-level": "Junior",
    "entry level": "Junior",
    "junior": "Junior",
    "mi": "Mid",
    "mid": "Mid",
    "mid-level": "Mid",
    "mid level": "Mid",
    "se": "Senior",
    "senior": "Senior",
    "ex": "Executive",
    "executive": "Executive",
}
# Order of the boxes on the experience-level boxplot.
EXPERIENCE_ORDER = ["Junior", "Mid", "Senior", "Lead", "Executive"]

# Groq exposes an OpenAI-compatible API, so one LangChain class covers both.
LLM_PROVIDERS = {
    "OpenAI": {"base_url": None, "default_model": "gpt-4o-mini"},
    "Groq": {"base_url": "https://api.groq.com/openai/v1", "default_model": "llama-3.3-70b-versatile"},
}

CAREER_PROMPT = PromptTemplate.from_template(
    "You are an expert tech career advisor. Based on the following real market data "
    "for the role of {job_title} (Average Salary: {mean_salary}, Top Skills: {top_skills}), "
    "write a short, actionable 3-step Career Action Plan for a student aiming to get this job. "
    "Be encouraging and reference the data provided.\n\n"
    "Additional market data: Median Salary: {median_salary}, Maximum Salary: {max_salary}, "
    "Salary Standard Deviation: {std_salary}, Job postings analysed: {posting_count}."
)

TOP_SKILL_COUNT = 5


# --------------------------------------------------------------------------
# Step 2: Data loading & cleaning (Pandas)
# --------------------------------------------------------------------------

def load_csv(source) -> pd.DataFrame:
    """Read a CSV (uploaded file or path) into a DataFrame.

    Raises ValueError with a user-friendly message if the file is unusable.
    """
    try:
        try:
            df = pd.read_csv(source)
        except UnicodeDecodeError:
            # Some Kaggle exports are not UTF-8 encoded; retry with Latin-1.
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, encoding="latin-1")
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded file is empty.")
    except pd.errors.ParserError:
        raise ValueError("The file could not be parsed. Please upload a valid, comma-separated CSV.")
    except FileNotFoundError:
        raise ValueError(f"Dataset not found: {source}")

    if df.empty:
        raise ValueError("The uploaded file has column headers but no rows.")
    return df


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the dataset's columns to the names the app expects."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[\s\-]+", "_", regex=True)

    # The first alias found wins, so "salary_in_usd" is preferred over a
    # local-currency "salary" column when a dataset has both.
    rename_map = {}
    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = target
                break
    df = df.rename(columns=rename_map)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Required column(s) missing: {', '.join(missing)}. "
            f"Columns found in the file: {', '.join(df.columns)}."
        )
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Handle missing values and tidy up the columns used in the analysis.

    Returns the cleaned DataFrame and a small report of what was removed.
    """
    df = standardise_columns(df)
    rows_before = len(df)

    # Salaries sometimes arrive as text such as "$120,000" - make them numeric.
    if not pd.api.types.is_numeric_dtype(df["salary_in_usd"]):
        df["salary_in_usd"] = df["salary_in_usd"].astype(str).str.replace(r"[^\d.\-]", "", regex=True)
    df["salary_in_usd"] = pd.to_numeric(df["salary_in_usd"], errors="coerce")

    df["job_title"] = df["job_title"].astype("string").str.strip()
    df["job_title"] = df["job_title"].replace("", pd.NA)

    # Drop rows where the salary or the job title is missing.
    missing_salary = int(df["salary_in_usd"].isna().sum())
    missing_title = int(df["job_title"].isna().sum())
    df = df.dropna(subset=["salary_in_usd", "job_title"])

    # A salary of zero or less is a data-entry error, not a real posting.
    invalid_salary = int((df["salary_in_usd"] <= 0).sum())
    df = df[df["salary_in_usd"] > 0]

    # Experience level is optional: fill the gaps and expand short codes.
    if "experience_level" in df.columns:
        levels = df["experience_level"].fillna("Unknown").astype(str).str.strip()
        df["experience_level"] = levels.str.lower().map(EXPERIENCE_LABELS).fillna(levels.str.title())
    else:
        df["experience_level"] = "Unknown"

    report = {
        "rows_before": rows_before,
        "rows_after": len(df),
        "missing_salary": missing_salary,
        "missing_title": missing_title,
        "invalid_salary": invalid_salary,
    }
    return df.reset_index(drop=True), report


def filter_by_role(df: pd.DataFrame, job_title: str) -> pd.DataFrame:
    """Keep only the selected job title and add a normalised salary column."""
    role_df = df[df["job_title"] == job_title].copy()

    # Min-max normalisation: rescale salaries to the 0-1 range for this role.
    salary_min = role_df["salary_in_usd"].min()
    salary_range = role_df["salary_in_usd"].max() - salary_min
    if salary_range > 0:
        role_df["salary_normalised"] = (role_df["salary_in_usd"] - salary_min) / salary_range
    else:
        role_df["salary_normalised"] = 0.0
    return role_df


# --------------------------------------------------------------------------
# Step 3: Statistical analysis (Pandas & SciPy)
# --------------------------------------------------------------------------

def compute_salary_statistics(role_df: pd.DataFrame) -> dict:
    """Measures of central tendency and dispersion for the role's salaries."""
    salaries = role_df["salary_in_usd"]
    count = len(salaries)

    return {
        "count": count,
        # Central tendency
        "mean": float(salaries.mean()),
        "median": float(salaries.median()),
        "min": float(salaries.min()),
        "max": float(salaries.max()),
        # Dispersion (sample statistics need at least two observations)
        "std": float(stats.tstd(salaries)) if count > 1 else 0.0,
        "variance": float(stats.tvar(salaries)) if count > 1 else 0.0,
        "iqr": float(stats.iqr(salaries)),
        # Shape of the distribution
        "skewness": float(stats.skew(salaries)) if count > 2 else 0.0,
    }


def get_top_skills(role_df: pd.DataFrame, top_n: int = TOP_SKILL_COUNT) -> pd.Series:
    """Count the most frequently requested skills for the role.

    Returns a Series indexed by skill name with the posting count as values.
    """
    if "key_skills" not in role_df.columns:
        return pd.Series(dtype=int)

    # "Python, SQL, AWS" -> one row per skill
    skills = role_df["key_skills"].dropna().astype(str).str.split(",").explode().str.strip()
    skills = skills[skills != ""]
    if skills.empty:
        return pd.Series(dtype=int)

    # Count case-insensitively ("python" == "Python") but keep the most
    # common spelling for display.
    grouped = skills.groupby(skills.str.lower())
    counts = grouped.size()
    display_names = grouped.agg(lambda names: names.mode().iat[0])
    counts.index = counts.index.map(display_names)

    return counts.sort_values(ascending=False).head(top_n)


# --------------------------------------------------------------------------
# Step 4: Data visualisation (Seaborn & Matplotlib)
# --------------------------------------------------------------------------

def format_usd(value: float) -> str:
    """Format a number as a whole-dollar amount, e.g. $123,456."""
    return f"${value:,.0f}"


def plot_salary_boxplot(role_df: pd.DataFrame, job_title: str) -> plt.Figure:
    """Graph 1: salary distribution across experience levels (Seaborn boxplot)."""
    present = role_df["experience_level"].unique()
    order = [level for level in EXPERIENCE_ORDER if level in present]
    order += sorted(level for level in present if level not in EXPERIENCE_ORDER)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(
        data=role_df,
        x="experience_level",
        y="salary_in_usd",
        order=order,
        hue="experience_level",
        palette="Set2",
        legend=False,
        ax=ax,
    )
    ax.set_title(f"Salary Distribution by Experience Level\n{job_title}")
    ax.set_xlabel("Experience Level")
    ax.set_ylabel("Salary (USD)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value / 1000:,.0f}k"))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_top_skills(top_skills: pd.Series, job_title: str) -> plt.Figure:
    """Graph 2: frequency of the most requested skills (Matplotlib bar chart)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(top_skills.index, top_skills.values, color=sns.color_palette("Set2", len(top_skills)))
    ax.bar_label(bars, padding=3)
    ax.set_title(f"Top {len(top_skills)} Requested Skills\n{job_title}")
    ax.set_xlabel("Skill")
    ax.set_ylabel("Number of Job Postings")
    ax.tick_params(axis="x", rotation=20)
    ax.margins(y=0.12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Step 5: LLM integration (LangChain LCEL)
# --------------------------------------------------------------------------

def build_llm_inputs(job_title: str, salary_stats: dict, top_skills: pd.Series) -> dict:
    """Compile the computed statistics into the dictionary fed to the prompt."""
    return {
        "job_title": job_title,
        "mean_salary": format_usd(salary_stats["mean"]),
        "median_salary": format_usd(salary_stats["median"]),
        "max_salary": format_usd(salary_stats["max"]),
        "std_salary": format_usd(salary_stats["std"]),
        "posting_count": salary_stats["count"],
        "top_skills": ", ".join(top_skills.index) if not top_skills.empty else "not available in the dataset",
    }


def generate_career_plan(llm_inputs: dict, provider: str, model: str, api_key: str) -> str:
    """Run the prompt -> LLM -> text chain and return the Career Action Plan."""
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=LLM_PROVIDERS[provider]["base_url"],
        temperature=0.7,
    )
    # LCEL: the output of each step is piped into the next one.
    chain = CAREER_PROMPT | llm | StrOutputParser()
    return chain.invoke(llm_inputs)


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

def render_sidebar() -> dict:
    """Step 1: dataset upload and LLM settings."""
    st.sidebar.header("1. Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload a job postings CSV", type=["csv"])
    use_sample = st.sidebar.checkbox(
        "Use the bundled sample dataset",
        value=False,
        disabled=uploaded_file is not None or not SAMPLE_DATA_PATH.exists(),
        help="A synthetic dataset shipped with the app for demo purposes.",
    )

    st.sidebar.header("2. LLM Settings")
    provider = st.sidebar.selectbox("LLM provider", list(LLM_PROVIDERS))
    api_key = st.sidebar.text_input(
        f"{provider} API key",
        type="password",
        help="Used only for this session to generate the Career Action Plan. It is never stored.",
    )
    model = st.sidebar.text_input("Model name", value=LLM_PROVIDERS[provider]["default_model"], key=f"model_{provider}")

    return {
        "uploaded_file": uploaded_file,
        "use_sample": use_sample and uploaded_file is None,
        "provider": provider,
        "api_key": api_key.strip(),
        "model": model.strip(),
    }


def render_welcome() -> None:
    """Shown until a dataset has been provided."""
    st.info("Upload a CSV from the sidebar (or tick **Use the bundled sample dataset**) to begin.")
    st.subheader("Expected dataset structure")
    st.markdown(
        "| Column | Required | Example |\n"
        "|---|---|---|\n"
        "| `job_title` | Yes | Data Scientist |\n"
        "| `salary_in_usd` | Yes | 125000 |\n"
        "| `experience_level` | Optional | Junior, Mid, Senior |\n"
        "| `key_skills` | Optional | Python, SQL, AWS |\n"
    )
    st.caption(
        'Any Kaggle "Data Science Salaries" or "Tech Job Postings" dataset works. '
        "Common alternative column names (e.g. `salary`, `skills`) are detected automatically."
    )


def render_statistics(salary_stats: dict) -> None:
    """Salary metrics laid out in columns."""
    st.subheader("Salary Statistics")
    columns = st.columns(5)
    columns[0].metric("Mean", format_usd(salary_stats["mean"]))
    columns[1].metric("Median", format_usd(salary_stats["median"]))
    columns[2].metric("Minimum", format_usd(salary_stats["min"]))
    columns[3].metric("Maximum", format_usd(salary_stats["max"]))
    columns[4].metric("Std. Deviation", format_usd(salary_stats["std"]))

    st.caption(
        f"Based on **{salary_stats['count']:,}** job postings  |  "
        f"Variance: {salary_stats['variance']:,.0f}  |  "
        f"Interquartile range: {format_usd(salary_stats['iqr'])}  |  "
        f"Skewness: {salary_stats['skewness']:.2f}"
    )
    if salary_stats["count"] < 2:
        st.warning("Only one posting matches this role, so dispersion measures cannot be estimated.")


def render_career_plan(job_title: str, salary_stats: dict, top_skills: pd.Series, settings: dict) -> None:
    """Career Action Plan section: button, LLM call and output."""
    st.subheader("AI Career Action Plan")
    st.write("Generate a personalised 3-step plan grounded in the statistics above.")

    plans = st.session_state.setdefault("career_plans", {})

    if st.button("Generate Career Action Plan", type="primary"):
        if not settings["api_key"]:
            st.warning(f"Enter your {settings['provider']} API key in the sidebar first.")
        elif not settings["model"]:
            st.warning("Enter a model name in the sidebar first.")
        else:
            llm_inputs = build_llm_inputs(job_title, salary_stats, top_skills)
            try:
                with st.spinner("Asking the career advisor..."):
                    plans[job_title] = generate_career_plan(
                        llm_inputs, settings["provider"], settings["model"], settings["api_key"]
                    )
            except Exception as error:  # network, authentication, bad model name, ...
                st.error(f"The LLM request failed: {error}")

    if job_title in plans:
        st.success(plans[job_title])


def main() -> None:
    st.set_page_config(page_title="Tech Salary & Skills Analyzer", page_icon="📊", layout="wide")
    st.title("📊 Automated Tech Salary & Skills Analyzer")
    st.caption("Upload job postings, explore salary statistics and skill demand, and get an AI-written career plan.")

    settings = render_sidebar()

    # ---- Load the dataset ------------------------------------------------
    if settings["uploaded_file"] is not None:
        source = settings["uploaded_file"]
    elif settings["use_sample"]:
        source = SAMPLE_DATA_PATH
    else:
        render_welcome()
        return

    try:
        raw_df = load_csv(source)
        df, report = clean_data(raw_df)
    except ValueError as error:
        st.error(str(error))
        return
    except Exception as error:
        st.error(f"Something went wrong while reading the file: {error}")
        return

    if df.empty:
        st.error("No usable rows are left after cleaning. Check the salary and job title columns.")
        return

    # ---- Job title selection ---------------------------------------------
    st.sidebar.header("3. Job Title")
    # Most common titles first, so the dropdown opens on well-populated roles.
    job_titles = df["job_title"].value_counts().index.tolist()
    job_title = st.sidebar.selectbox("Select a job title to analyse", job_titles)

    role_df = filter_by_role(df, job_title)

    # ---- Data cleaning summary -------------------------------------------
    with st.expander("Data cleaning summary"):
        columns = st.columns(4)
        columns[0].metric("Rows uploaded", f"{report['rows_before']:,}")
        columns[1].metric("Rows after cleaning", f"{report['rows_after']:,}")
        columns[2].metric("Missing salary", f"{report['missing_salary']:,}")
        columns[3].metric("Missing job title", f"{report['missing_title']:,}")
        if report["invalid_salary"]:
            st.caption(f"{report['invalid_salary']:,} row(s) with a zero or negative salary were also removed.")
        st.write(f"Cleaned rows for **{job_title}**:")
        st.dataframe(role_df.head(50))

    st.header(job_title)

    # ---- Statistics --------------------------------------------------------
    salary_stats = compute_salary_statistics(role_df)
    top_skills = get_top_skills(role_df)
    render_statistics(salary_stats)

    # ---- Visualisations ----------------------------------------------------
    st.subheader("Visualisations")
    left, right = st.columns(2)

    with left:
        fig = plot_salary_boxplot(role_df, job_title)
        st.pyplot(fig)
        plt.close(fig)

    with right:
        if top_skills.empty:
            st.info("No skills column was found in this dataset, so skill demand cannot be charted.")
        else:
            fig = plot_top_skills(top_skills, job_title)
            st.pyplot(fig)
            plt.close(fig)

    if not top_skills.empty:
        st.subheader(f"Top {len(top_skills)} Requested Skills")
        skills_table = top_skills.rename("Job Postings").rename_axis("Skill").reset_index()
        skills_table["Share of Postings"] = (skills_table["Job Postings"] / salary_stats["count"]).map("{:.0%}".format)
        skills_table.index = np.arange(1, len(skills_table) + 1)
        st.table(skills_table)

    # ---- Career Action Plan ------------------------------------------------
    st.divider()
    render_career_plan(job_title, salary_stats, top_skills, settings)


if __name__ == "__main__":
    main()
