# Project Specification: Automated Tech Salary & Skills Analyzer

## 1. Project Overview
This project is an interactive web application built with Streamlit. It allows users to upload a dataset of tech job postings (containing salaries and skills), performs automated data cleaning and statistical analysis, visualizes the salary distributions, and uses a Large Language Model (via LangChain) to generate a personalized "Career Action Plan" based on the statistical findings.

## 2. Academic Syllabus Alignment (GTU BE05000231)
This project strictly adheres to the "Python for Data Science" syllabus requirements:
*   **Data Preparation (Pandas):** Handling missing values, filtering by job roles, and normalizing salary data.
*   **Descriptive Statistics (Pandas/SciPy):** Calculating measures of central tendency (mean, median) and dispersion (variance/standard deviation) for salaries based on specific skills.
*   **Data Visualization (Matplotlib/Seaborn):** Creating boxplots and bar charts to visually represent salary distributions and skill frequencies.
*   **Advanced Integration (LangChain):** Bridging traditional data science pipelines with modern AI agentic workflows.

## 3. Tech Stack
*   **Frontend UI:** `streamlit`
*   **Data Manipulation:** `pandas`, `numpy`
*   **Statistics:** `scipy`
*   **Visualization:** `matplotlib`, `seaborn`
*   **LLM Framework:** `langchain`, `langchain-openai` (or `langchain-community` for local LLMs like Ollama)

## 4. Recommended Dataset Structure
You can use any Kaggle dataset related to "Data Science Salaries" or "Tech Job Postings". The CSV should ideally contain:
*   `job_title` (e.g., Data Scientist, Software Engineer)
*   `salary_in_usd` (Numeric)
*   `experience_level` (e.g., Junior, Mid, Senior)
*   `key_skills` (Comma-separated strings, e.g., "Python, SQL, AWS")

## 5. Application Architecture & Flow

### Step 1: Streamlit Sidebar & Setup
*   Provide a file uploader widget in the sidebar to accept a `.csv` file.
*   Provide a text input for the LLM API Key (e.g., OpenAI API Key or Groq API Key).
*   Add a dropdown to select a specific `job_title` to analyze.

### Step 2: Data Cleaning & Processing (Pandas)
*   Read the uploaded CSV into a Pandas DataFrame.
*   Drop rows where `salary` or `job_title` is missing.
*   Filter the DataFrame based on the user's selected job title from the sidebar.

### Step 3: Statistical Analysis (SciPy & Pandas)
*   Calculate the **Mean**, **Median**, **Minimum**, and **Maximum** salary for the selected role.
*   Calculate the **Standard Deviation** of the salary to show market variance.
*   Extract and count the top 5 most frequently requested skills for that role.

### Step 4: Data Visualization (Seaborn/Matplotlib into Streamlit)
*   **Graph 1:** A Seaborn Boxplot (`sns.boxplot`) showing the salary distribution across different experience levels for the selected role. 
*   **Graph 2:** A Matplotlib Bar Chart (`plt.bar`) showing the top 5 requested skills and their frequency.
*   Use `st.pyplot()` to render these graphs in the Streamlit web app.

### Step 5: LLM Integration (LangChain)
*   Compile the calculated statistics (Mean Salary, Max Salary, Top 5 Skills) into a Python dictionary or JSON string.
*   Create a LangChain `PromptTemplate` with the following instruction:
    * *"You are an expert tech career advisor. Based on the following real market data for the role of {job_title} (Average Salary: {mean_salary}, Top Skills: {top_skills}), write a short, actionable 3-step Career Action Plan for a student aiming to get this job. Be encouraging and reference the data provided."*
*   Pass the data into the LLM chain and display the generated text in the Streamlit app using `st.success()` or `st.markdown()`.

---

## 6. Instructions for Claude (AI Coding Assistant)
**Dear Claude:** Please read the specifications above and generate a complete, single-file Python script (`app.py`) using Streamlit that implements this exact workflow. 
1. Ensure the code is well-commented and includes error handling for the CSV upload.
2. Use `langchain_core.prompts` and standard LangChain LCEL (chaining) for the LLM generation. 
3. Make the UI look clean using Streamlit columns and headers.
4. Keep the Pandas and Seaborn code straightforward to satisfy the academic requirements mentioned in Section 2.