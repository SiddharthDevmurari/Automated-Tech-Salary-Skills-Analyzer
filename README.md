# Automated Tech Salary & Skills Analyzer

An interactive Streamlit web app that turns a CSV of tech job postings into salary statistics, skill-demand charts, and an AI-written **Career Action Plan**.

Upload a dataset, pick a job title, and the app will:

1. **Clean the data** with Pandas (missing values, text salaries, inconsistent experience labels).
2. **Compute descriptive statistics** with Pandas and SciPy (mean, median, min, max, standard deviation, variance, IQR, skewness).
3. **Visualise** the salary distribution per experience level (Seaborn boxplot) and the top 5 requested skills (Matplotlib bar chart).
4. **Generate a 3-step Career Action Plan** with an LLM through LangChain, grounded in the computed statistics.

## Tech Stack

| Purpose | Library |
|---|---|
| Frontend UI | `streamlit` |
| Data manipulation | `pandas`, `numpy` |
| Statistics | `scipy` |
| Visualisation | `matplotlib`, `seaborn` |
| LLM framework | `langchain-core`, `langchain-openai` (LCEL chain) |

## Getting Started

Requires Python 3.10 or newer.

```bash
git clone https://github.com/SiddharthDevmurari/Automated-Tech-Salary-Skills-Analyzer.git
cd Automated-Tech-Salary-Skills-Analyzer

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>.

## Using the App

1. **Dataset** - upload a `.csv` in the sidebar, or tick *Use the bundled sample dataset*.
2. **LLM settings** - choose OpenAI or Groq, paste your API key, and adjust the model name if you want a different one. The key is only needed for the Career Action Plan; all statistics and charts work without it.
3. **Job title** - choose the role to analyse from the dropdown.
4. Click **Generate Career Action Plan**.

API keys: [OpenAI](https://platform.openai.com/api-keys) or [Groq](https://console.groq.com/keys) (Groq has a free tier). The key is held in memory for the session only and is never written to disk.

## Dataset Structure

Any Kaggle "Data Science Salaries" or "Tech Job Postings" dataset works. The CSV should contain:

| Column | Required | Example |
|---|---|---|
| `job_title` | Yes | Data Scientist |
| `salary_in_usd` | Yes | 125000 |
| `experience_level` | Optional | Junior, Mid, Senior (or Kaggle codes `EN`, `MI`, `SE`, `EX`) |
| `key_skills` | Optional | Python, SQL, AWS |

Common alternative column names such as `salary`, `title`, `experience` and `skills` are detected automatically. If the dataset has no skills column, the skills chart is skipped and the rest of the analysis still runs.

### Sample dataset

`data/sample_tech_jobs.csv` is a **synthetic** dataset (800 rows, 8 roles) produced by `generate_sample_data.py`. Its numbers are randomly generated for demo purposes and are not real market data. Regenerate it with:

```bash
python generate_sample_data.py
```

## How It Works

| Step | What happens | Where |
|---|---|---|
| 1. Setup | File uploader, API key input and job title dropdown in the sidebar | `render_sidebar()` |
| 2. Cleaning | Standardise column names, convert salaries to numbers, drop rows with a missing salary or job title, filter by role, min-max normalise salaries | `clean_data()`, `filter_by_role()` |
| 3. Statistics | Central tendency and dispersion of salaries; top 5 skills by frequency | `compute_salary_statistics()`, `get_top_skills()` |
| 4. Visualisation | `sns.boxplot` of salary by experience level; bar chart of top skills, rendered with `st.pyplot()` | `plot_salary_boxplot()`, `plot_top_skills()` |
| 5. LLM | Statistics are compiled into a dictionary and passed through a `PromptTemplate \| ChatOpenAI \| StrOutputParser` chain | `generate_career_plan()` |

## Syllabus Alignment

Built for the GTU course **Python for Data Science (BE05000231)**:

- **Data preparation (Pandas)** - handling missing values, filtering by job role, normalising salary data.
- **Descriptive statistics (Pandas / SciPy)** - measures of central tendency and dispersion.
- **Data visualisation (Matplotlib / Seaborn)** - boxplots and bar charts.
- **Advanced integration (LangChain)** - connecting a classic data science pipeline to an LLM.

## Project Structure

```
.
├── app.py                     # The Streamlit application
├── generate_sample_data.py    # Builds the synthetic sample dataset
├── data/
│   └── sample_tech_jobs.csv   # Synthetic demo dataset
├── requirements.txt
└── README.md
```
