# Automated Tech Salary & Skills Analyzer

An interactive Streamlit web app that turns a CSV of tech job postings into salary statistics, skill-demand charts, and an AI-written **Career Action Plan** - and checks your own **resume / CV** against what the market asks for.

Upload a dataset, pick a job title, and the app will:

1. **Clean the data** with Pandas (missing values, text salaries, inconsistent experience labels).
2. **Compute descriptive statistics** with Pandas and SciPy (mean, median, min, max, standard deviation, variance, IQR, skewness).
3. **Visualise** the salary distribution per experience level (Seaborn boxplot) and the top 5 requested skills (Matplotlib bar chart).
4. **Generate a 3-step Career Action Plan** with an LLM through LangChain, grounded in the computed statistics.

Upload your resume as well, and it will also:

5. **Analyse your resume** against the role: a skill match score, the in-demand skills you have, the ones to learn, the ones this role rarely asks for, and a checklist of standard sections.
6. **Let you chat with the AI about your resume** - what to improve, what to learn next and what is unnecessary - with one-click sample questions.

## Tech Stack

| Purpose | Library |
|---|---|
| Frontend UI | `streamlit` |
| Data manipulation | `pandas`, `numpy` |
| Statistics | `scipy` |
| Visualisation | `matplotlib`, `seaborn` |
| LLM framework | `langchain-core`, `langchain-openai` (LCEL chain) |
| Resume parsing | `pypdf`, `python-docx` |

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

1. **Job postings** - upload a `.csv` in the sidebar, or click **Try the sample dataset** on the welcome screen (the same as ticking *Use the bundled sample dataset*).
2. **Resume / CV** (optional) - upload a PDF, DOCX or TXT file.
3. **AI settings** - choose OpenAI or Groq, paste your API key, and pick a model from the dropdown (Groq offers `qwen/qwen3.8-27b` by default, plus `openai/gpt-oss-120b` and `openai/gpt-oss-20b`; OpenAI offers `gpt-4o-mini`). You can also type the name of any other chat model your key can use. The key is only needed for the Career Action Plan and the resume chat; all statistics, charts and the resume analysis work without it.
4. **Job title** - choose the role to analyse from the dropdown.

The results are split into four tabs:

| Tab | What it shows |
|---|---|
| **Market analysis** | Salary tiles (mean, median, minimum, maximum, standard deviation), a plain-language note on the spread, the two charts (each with a *View as table* option) and the **Generate career action plan** button |
| **Resume analysis** | Skill match score with a verdict, a chart of the role's most requested skills marked as on your resume or missing, colour-coded skills you have / should learn / the role rarely asks for, and a section and contact checklist |
| **Resume chat** | A chat with the AI about your resume. Click a sample question to send it straight away, or type your own |
| **Data** | What the cleaning step removed, and the cleaned rows for the selected role |

### Look and feel

The app uses a flat dashboard style: a navy sidebar, a pale mint page, white cards with coloured title bars, round icon badges, and a three-colour palette of navy `#282735`, coral `#f9876f` and green `#9be198`.

- The base colours and the heading font are set in `.streamlit/config.toml`, which Streamlit reads automatically when the app is started from the project folder. The same palette is defined once at the top of the chart section in `app.py` and reused by the charts and the page's CSS.
- Headings use the condensed **Oswald** font, loaded from Google Fonts. Without an internet connection the app falls back to the default sans-serif font and otherwise looks the same.

API keys: [OpenAI](https://platform.openai.com/api-keys) or [Groq](https://console.groq.com/keys) (Groq has a free tier). The key is held in memory for the session only and is never written to disk.

### Resume notes

- The resume analysis runs on your computer. The resume text is sent to the LLM provider you chose only when you use the chat.
- Skills are matched by name against the skills that appear in the dataset, so a skill written differently in your resume (for example "REST API" vs "REST APIs") may be missed. A dataset without a skills column gives a checklist-only analysis.
- Scanned, image-only PDFs contain no text and cannot be read.
- The chat restarts when you change the resume or the job title.

## Dataset Structure

Any Kaggle "Data Science Salaries" or "Tech Job Postings" dataset works. The CSV should contain:

| Column | Required | Example |
|---|---|---|
| `job_title` | Yes | Data Scientist |
| `salary_in_usd` | Yes | 125000 |
| `experience_level` | Optional | Junior, Mid, Senior (or Kaggle codes `EN`, `MI`, `SE`, `EX`) |
| `key_skills` | Optional | Python, SQL, AWS |

Common alternative column names such as `salary`, `title`, `experience` and `skills` are detected automatically. If the dataset has no skills column, the skills chart is skipped and the rest of the analysis still runs.

## Sample Files

Everything in `data/` is **synthetic**: the datasets are randomly generated and the resumes describe fictional people, companies and colleges. They are for testing and demos, not real market data.

### Sample datasets

| File | Rows | What it demonstrates |
|---|---|---|
| `sample_tech_jobs.csv` | 800 | Clean data in the app's ideal layout, 8 roles (Data Scientist, Data Analyst, ML Engineer, Data Engineer, Software Engineer, DevOps Engineer, Frontend Developer, Backend Developer). This is the one behind *Use the bundled sample dataset*. |
| `sample_ai_cloud_jobs.csv` | 1,200 | Same layout, 8 different roles (AI Engineer, Cloud Engineer, Cybersecurity Analyst, Full Stack Developer, Mobile App Developer, QA Automation Engineer, Business Intelligence Analyst, Site Reliability Engineer). |
| `sample_kaggle_style_salaries.csv` | 600 | The column layout of Kaggle's "Data Science Salaries" dataset: experience codes (`EN`, `MI`, `SE`, `EX`) expanded to labels, `salary_in_usd` chosen over the local-currency `salary` column, and the graceful fallback when there is no skills column. |
| `sample_messy_job_postings.csv` | 400 | Data cleaning: columns named `Job Title` / `Annual Salary` / `Experience` / `Skills`, salaries written as `$79,500`, `USD 116,000` or `Not disclosed`, zero and negative salaries, padded job titles, mixed-case skills and inconsistent experience labels. Open the **Data** tab to see what was removed. |

Regenerate all four with:

```bash
python generate_sample_data.py
```

### Sample resumes

In `data/resumes/`, covering all three supported formats:

| File | Profile | Try it with |
|---|---|---|
| `resume_fresher_data_analyst.pdf` | Final-year student, no work experience yet | *Data Analyst* (strong match) or *Data Scientist* (clear skills to learn) in `sample_tech_jobs.csv` |
| `resume_frontend_developer_career_switch.docx` | 3 years in frontend, moving into data | *Frontend Developer* (full match) vs *Data Scientist* (low match, many skills the role rarely asks for) in `sample_tech_jobs.csv` |
| `resume_senior_data_engineer.pdf` | 7 years, strong and well written | *Data Engineer* in `sample_tech_jobs.csv` (full match) |
| `resume_devops_cloud_engineer.docx` | 4 years in DevOps; skills are in a table | *DevOps Engineer* in `sample_tech_jobs.csv`, or *Cloud Engineer* / *Site Reliability Engineer* in `sample_ai_cloud_jobs.csv` |
| `resume_ml_engineer.pdf` | 2.5 years in machine learning, recent LLM work | *AI Engineer* in `sample_ai_cloud_jobs.csv` or *Machine Learning Engineer* in `sample_tech_jobs.csv` |
| `resume_needs_improvement.txt` | Deliberately weak: personal details, vague claims, long hobbies list, no links | Any role. Best for the chat prompts *What should I improve in my resume?* and *What is unnecessary in my resume and can be removed?* |

## How It Works

| Step | What happens | Where |
|---|---|---|
| 1. Setup | File uploader, API key input and job title dropdown in the sidebar | `render_sidebar()` |
| 2. Cleaning | Standardise column names, convert salaries to numbers, drop rows with a missing salary or job title, filter by role, min-max normalise salaries | `clean_data()`, `filter_by_role()` |
| 3. Statistics | Central tendency and dispersion of salaries; top 5 skills by frequency | `compute_salary_statistics()`, `get_top_skills()` |
| 4. Visualisation | `sns.boxplot` of salary by experience level; Matplotlib bar chart (`ax.bar`) of top skills; both rendered with `st.pyplot()` in the app's navy / coral / green palette | `plot_salary_boxplot()`, `plot_top_skills()` |
| 5. LLM | Statistics are compiled into a dictionary and passed through a `PromptTemplate \| ChatOpenAI \| StrOutputParser` chain | `generate_career_plan()` |
| 6. Resume analysis | Extract the resume text, find which dataset skills it mentions, and score it against the role's 10 most requested skills (weighted by share of postings) | `extract_resume_text()`, `analyse_resume()`, `plot_resume_skill_match()` |
| 7. Resume chat | Market data, the skill comparison and the resume go into the system message of a `ChatPromptTemplate`; earlier turns are replayed through a `MessagesPlaceholder` and the reply is streamed | `build_chat_context()`, `stream_resume_answer()` |

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
├── .streamlit/
│   └── config.toml            # Theme: colours and heading font
├── generate_sample_data.py    # Builds the synthetic sample datasets
├── data/
│   ├── sample_tech_jobs.csv
│   ├── sample_ai_cloud_jobs.csv
│   ├── sample_kaggle_style_salaries.csv
│   ├── sample_messy_job_postings.csv
│   └── resumes/               # Six fictional sample resumes (PDF, DOCX, TXT)
├── requirements.txt
└── README.md
```
