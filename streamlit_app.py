import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from streamlit_option_menu import option_menu
from pathlib import Path
import pandas as pd
from src.logging_utils import get_logger
from src.configuration import PATHS
from src.ml_attrition import train_attrition_model, train_attrition_model_from_db, AttritionConfig
from src.ml_performance_regression import train_performance_model, train_performance_model_from_db, PerfConfig
from src.sentiment_analysis import train_sentiment, train_sentiment_from_db
from src.eda import eda_overview, eda_overview_db
from src.transformers_skill_matching import match_resume_to_jd, match_all_resumes_to_jds
from src.interview_summarization import summarize_all_interview_transcripts_from_uri
from src.time_series_forecasting import load_workload_series
from src.dl_cnn_fraud import train_cnn, TrainConfig
from src.llm_chatbot import gpt_summarize, gpt_chatbot_db
import matplotlib.pyplot as plt
import seaborn as sns
from src.db_utils import read_sql_df
from src.sentiment_analysis import train_sentiment_from_db, SENTIMENT_SQL


# --- Set Page Config and Add Banner Image ---
st.set_page_config(page_title="AI-Powered Talent Intelligence & Workforce Optimization Suite (T-IQ)", layout="wide")


# --- Custom Header and Theme ---
st.markdown(
    """
    <style>
    .main-title {
        font-size:2.5rem;
        font-weight:700;
        color:#1a237e;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.2rem 0.5rem 1.2rem 0.5rem;
        border-radius: 0.5rem;
        text-align:center;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(26,35,126,0.08);
        letter-spacing: 1px;
    }
    .stApp {
        background-color: #f5f7fa;
    }
    .section-header {
        font-size:1.5rem;
        font-weight:600;
        color:#1565c0;
        margin-top:1.5rem;
        margin-bottom:0.5rem;
        display:flex;
        align-items:center;
        gap:0.5rem;
    }
    .section-icon { font-size:2rem; margin-right:0.5rem; vertical-align:middle; filter: drop-shadow(0 1px 2px #1976d233); }
    .icon-eda { color:#1976d2 !important; }
    .icon-hist { color:#388e3c !important; }
    .icon-box { color:#fbc02d !important; }
    .icon-scatter { color:#d32f2f !important; }
    .icon-bar { color:#7b1fa2 !important; }
    .icon-attrition { color:#ff7043 !important; }
    .icon-risk { color:#0288d1 !important; }
    .icon-pie { color:#c2185b !important; }
    .icon-heat { color:#f57c00 !important; }
    .icon-perf { color:#388e3c !important; }
    .icon-predict { color:#1976d2 !important; }
    .icon-sentiment { color:#fbc02d !important; }
    .icon-resume { color:#7b1fa2 !important; }
    .icon-interview { color:#0288d1 !important; }
    .icon-time { color:#1976d2 !important; }
    .icon-fraud { color:#d32f2f !important; }
    .icon-chatbot { color:#388e3c !important; }
    </style>
    <div class="main-title" style="display:flex;align-items:center;justify-content:center;gap:1rem;">
        <i class="bi bi-people-fill" style="font-size:2.2rem;color:#1976d2;margin-right:0.7rem;"></i>
        Talent Intelligence Suite (T-IQ)
    </div>
    """,
    unsafe_allow_html=True
)

logger = get_logger("streamlit_app")

st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">', unsafe_allow_html=True)


with st.sidebar:
    option = option_menu(
        "Main Menu",
        [
            "EDA",
            "Attrition Prediction",
            "Performance Regression",
            "Sentiment Analysis",
            "Resume-JD Matching",
            "Interview Summarization",
            "Time Series Forecasting",
            "CNN Fraud Detection",
            "LLM Chatbot"
            #,"LLM Summarization"
        ],
        icons=["bar-chart", "person-x", "graph-up-arrow", "emoji-smile", "file-earmark-person", "file-earmark-text", "clock-history", "shield-lock", "robot", "file-earmark-richtext"],
        menu_icon="cast",
        default_index=0,
    )

if option == "EDA":
    st.markdown('<div class="section-header"><i class="bi bi-bar-chart-fill section-icon icon-eda"></i>Exploratory Data Analysis</div>', unsafe_allow_html=True)
    df = eda_overview_db()
    logger.info(f"Loaded {len(df)} rows from DB for EDA")
    #st.dataframe(df)
    #st.success("EDA report generated")

    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.markdown('<div class="section-header"><i class="bi bi-graph-up-arrow section-icon icon-hist"></i>Histograms</div>', unsafe_allow_html=True)
    hist_col = st.selectbox("Select column for histogram", num_cols, key="hist")
    if hist_col:
        fig, ax = plt.subplots()
        sns.histplot(df[hist_col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Histogram of {hist_col}")
        st.pyplot(fig)

    st.markdown('<div class="section-header"><i class="bi bi-box-seam section-icon icon-box"></i>Box Plots</div>', unsafe_allow_html=True)
    box_col = st.selectbox("Select column for box plot", num_cols, key="box")
    group_col = st.selectbox("Group by (optional)", [None] + cat_cols, key="box_group")
    if box_col:
        fig, ax = plt.subplots()
        if group_col and group_col in cat_cols:
            sns.boxplot(x=df[group_col], y=df[box_col], ax=ax)
            ax.set_title(f"Box plot of {box_col} by {group_col}")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        else:
            sns.boxplot(y=df[box_col], ax=ax)
            ax.set_title(f"Box plot of {box_col}")
        st.pyplot(fig)

    st.markdown('<div class="section-header"><i class="bi bi-circle-fill section-icon icon-scatter"></i>Scatter Plots</div>', unsafe_allow_html=True)
    scatter_x = st.selectbox("X axis", num_cols, key="scatter_x")
    scatter_y = st.selectbox("Y axis", num_cols, key="scatter_y")
    hue_col = st.selectbox("Color by (optional)", [None] + cat_cols, key="scatter_hue")
    if scatter_x and scatter_y:
        fig, ax = plt.subplots()
        if hue_col and hue_col in cat_cols:
            sns.scatterplot(x=df[scatter_x], y=df[scatter_y], hue=df[hue_col], ax=ax)
            ax.set_title(f"Scatter plot: {scatter_x} vs {scatter_y} by {hue_col}")
        else:
            sns.scatterplot(x=df[scatter_x], y=df[scatter_y], ax=ax)
            ax.set_title(f"Scatter plot: {scatter_x} vs {scatter_y}")
        st.pyplot(fig)

    st.markdown('<div class="section-header"><i class="bi bi-bar-chart-steps section-icon icon-bar"></i>Stacked Bar Charts</div>', unsafe_allow_html=True)
    bar_cat1 = st.selectbox("Category 1", cat_cols, key="bar_cat1")
    bar_cat2 = st.selectbox("Category 2", cat_cols, key="bar_cat2")
    if bar_cat1 and bar_cat2 and bar_cat1 != bar_cat2:
        stacked_data = df.groupby([bar_cat1, bar_cat2]).size().unstack(fill_value=0)
        fig, ax = plt.subplots()
        colors = sns.color_palette('tab20', n_colors=stacked_data.shape[1])
        stacked_data.plot(kind='bar', stacked=True, ax=ax, color=colors)
        ax.set_title(f"Stacked Bar: {bar_cat1} vs {bar_cat2}")
        ax.set_ylabel("Count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

elif option == "Attrition Prediction":
    st.markdown('<div class="section-header"><i class="bi bi-person-x-fill section-icon icon-attrition"></i>Attrition Model Training & Insights</div>', unsafe_allow_html=True)
    
    cfg = AttritionConfig(
        cat_cols=["Department", "JobRole", "status", "location_code"],
        num_cols=["YearsAtCompany", "MonthlyIncome"],
        target_col="Attrition"
    )
    # Load data from DB (same SQL as in train_attrition_model_from_db)
    # Load data from DB (same SQL as in train_attrition_model_from_db)
    from src.db_utils import read_sql_df
    ATTRITION_SQL = """
    SELECT e.employee_id,
           d.name AS Department, jr.title AS JobRole, e.status,
           TIMESTAMPDIFF(YEAR, e.hire_date, CURRENT_DATE) AS YearsAtCompany,
           e.compensation_base AS MonthlyIncome,
           e.location_code,
           e.attrition_flag AS Attrition
    FROM employee e
    LEFT JOIN department d ON d.department_id = e.department_id
    LEFT JOIN job_role jr ON jr.job_role_id = e.job_role_id
    WHERE e.status IN ('Active','OnLeave','Terminated');
    """
    df = read_sql_df(ATTRITION_SQL)
    df["Attrition"] = df["Attrition"].astype(int)
    # Train model
    model = train_attrition_model_from_db(cfg)
    
    # Predict probabilities for plotting
    from src.data_cleaning import split_features_target
    X, _ = split_features_target(df, cfg.target_col)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[:, 1]
        df['Attrition_Prob'] = proba
    else:
        df['Attrition_Prob'] = model.predict(X)

    # Bar Chart: Attrition risk by department
    st.markdown('<div class="section-header"><i class="bi bi-people-fill section-icon icon-risk"></i>Attrition Risk by Department</div>', unsafe_allow_html=True)
    dept_risk = df.groupby('Department')['Attrition_Prob'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    dept_risk.plot(kind='bar', ax=ax, color=sns.color_palette('tab10', n_colors=len(dept_risk)))
    ax.set_ylabel('Avg. Attrition Probability')
    ax.set_title('Attrition Risk by Department')
    st.pyplot(fig)

    # Pie Chart: Percentage of employees at high/medium/low risk
    st.markdown('<div class="section-header"><i class="bi bi-pie-chart-fill section-icon icon-pie"></i>Employee Risk Distribution</div>', unsafe_allow_html=True)
    bins = [0, 0.33, 0.66, 1.0]
    labels = ['Low', 'Medium', 'High']
    df['Risk_Level'] = pd.cut(df['Attrition_Prob'], bins=bins, labels=labels, include_lowest=True)
    risk_counts = df['Risk_Level'].value_counts().reindex(labels, fill_value=0)
    fig2, ax2 = plt.subplots()
    ax2.pie(risk_counts, labels=labels, autopct='%1.1f%%', colors=['#8fd175', '#ffe066', '#ff686b'])
    ax2.set_title('Percentage of Employees by Attrition Risk')
    st.pyplot(fig2)

    # Heatmap: Attrition probability vs. features (e.g., age, salary)
    st.markdown('<div class="section-header"><i class="bi bi-grid-3x3-gap-fill section-icon icon-heat"></i>Attrition Probability Heatmap</div>', unsafe_allow_html=True)
    heatmap_data = df.copy()
    heatmap_data['YearsAtCompany_bin'] = pd.cut(heatmap_data['YearsAtCompany'], bins=6)
    heatmap_data['MonthlyIncome_bin'] = pd.cut(heatmap_data['MonthlyIncome'], bins=6)
    pivot = heatmap_data.pivot_table(index='YearsAtCompany_bin', columns='MonthlyIncome_bin', values='Attrition_Prob', aggfunc='mean')
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax3)
    ax3.set_title('Attrition Probability by Years at Company & Monthly Income')
    st.pyplot(fig3)

elif option == "Performance Regression":
    st.markdown('<div class="section-header"><i class="bi bi-graph-up-arrow section-icon icon-perf"></i>Performance Regression Model Training & Prediction</div>', unsafe_allow_html=True)
    cfg = PerfConfig(
        cat_cols=["Department", "JobRole"],
        num_cols=["MonthlyIncome", "YearsAtCompany"],
        target_col="PerformanceScore"
    )
    # Train model
    model = train_performance_model_from_db(cfg)
   
    st.markdown('<div class="section-header"><i class="bi bi-person-badge-fill section-icon icon-predict"></i>Predict Employee Performance</div>', unsafe_allow_html=True)
    # For demo, use manual input fields
    departments = ["Sales", "Engineering", "HR", "Finance", "Operations"]
    jobroles = ["Manager", "Analyst", "Developer", "Executive", "Assistant"]
    dept = st.selectbox("Department", departments)
    jobrole = st.selectbox("Job Role", jobroles)
    monthly_income = st.number_input("Monthly Income", min_value=0, value=5000)
    years_at_company = st.number_input("Years at Company", min_value=0, value=3)

    if st.button("Predict Performance"):
        import numpy as np
        # Prepare input as DataFrame
        input_df = pd.DataFrame({
            "Department": [dept],
            "JobRole": [jobrole],
            "MonthlyIncome": [monthly_income],
            "YearsAtCompany": [years_at_company]
        })
        # Predict
        try:
            pred = model.predict(input_df)[0]
            st.success(f"Predicted Performance Score: {pred:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif option == "Sentiment Analysis":
    st.markdown('<div class="section-header"><i class="bi bi-emoji-smile-fill section-icon icon-sentiment"></i>Sentiment Analysis Model Training</div>', unsafe_allow_html=True)
    train_sentiment_from_db()


elif option == "Resume-JD Matching":
    st.markdown('<div class="section-header"><i class="bi bi-file-earmark-person-fill section-icon icon-resume"></i>Resume-JD Matching</div>', unsafe_allow_html=True)
    if st.button("Run Resume-JD Matching"):
        results_df, _ = match_all_resumes_to_jds()
        with st.expander("Matching Results", expanded=True):
            st.dataframe(results_df)

elif option == "Interview Summarization":
    st.markdown('<div class="section-header"><i class="bi bi-file-earmark-text-fill section-icon icon-interview"></i>Interview Summarization</div>', unsafe_allow_html=True)
    # Enhanced: Join interview, application, candidate tables for richer dashboard
    INTERVIEW_SQL = '''
    SELECT i.interview_id, i.interview_type, i.scheduled_ts, i.duration_min, i.interviewers_json, i.recording_uri, i.transcript_uri,
           a.application_id, a.job_posting_id, c.candidate_id, c.first_name, c.last_name
    FROM interview i
    LEFT JOIN application a ON i.application_id = a.application_id
    LEFT JOIN candidate c ON a.candidate_id = c.candidate_id
    '''
    interview_df = read_sql_df(INTERVIEW_SQL)
    if interview_df.empty:
        st.warning("No interview data found.")
    else:
        # Show filter widgets
        interview_types = sorted(interview_df["interview_type"].dropna().unique().tolist())
        candidates = sorted(interview_df["first_name"].fillna('') + ' ' + interview_df["last_name"].fillna(''))
        selected_types = st.multiselect("Filter by Interview Type", interview_types, default=interview_types)
        selected_candidates = st.multiselect("Filter by Candidate", candidates, default=candidates)
        # Filter DataFrame
        filtered_df = interview_df[
            interview_df["interview_type"].isin(selected_types) &
            ((interview_df["first_name"].fillna('') + ' ' + interview_df["last_name"].fillna('')).isin(selected_candidates))
        ]
        # Display table with metadata and links
        def make_link(uri, label):
            if pd.isna(uri) or not uri:
                return ""
            return f'<a href="{uri}" target="_blank">{label}</a>'
        display_df = filtered_df.copy()
        display_df["Candidate"] = display_df["first_name"].fillna('') + ' ' + display_df["last_name"].fillna('')
        display_df["Transcript"] = display_df["transcript_uri"].apply(lambda x: make_link(x, "Transcript"))
        display_df["Recording"] = display_df["recording_uri"].apply(lambda x: make_link(x, "Recording"))
        show_cols = ["interview_id", "interview_type", "scheduled_ts", "duration_min", "interviewers_json", "Candidate", "job_posting_id", "Transcript", "Recording"]
        st.markdown("### Interview List")
        st.write("Select interviews to summarize:")
        st.dataframe(display_df[show_cols], use_container_width=True, hide_index=True)
        # Allow selection of interviews
        selected_ids = st.multiselect(
            "Select Interview IDs to Summarize",
            display_df["interview_id"].astype(str).tolist(),
            default=display_df["interview_id"].astype(str).tolist()
        )
        # Summarize selected interviews
        if st.button("Summarize Selected Interviews"):
            if selected_ids:
                # Optionally, pass interview_ids to summarization function
                from src.interview_summarization import summarize_all_interview_transcripts_from_uri
                # Filter to selected interviews and get their types (or pass IDs if function supports)
                selected_df = display_df[display_df["interview_id"].astype(str).isin(selected_ids)]
                selected_types = selected_df["interview_type"].unique().tolist()
                summary = summarize_all_interview_transcripts_from_uri(selected_types)
                with st.expander("Interview Summary", expanded=True):
                    st.markdown(summary if summary else "No summary available.")
            else:
                st.warning("Please select at least one interview.")

elif option == "Time Series Forecasting":
    st.markdown('<div class="section-header"><i class="bi bi-clock-history section-icon icon-time"></i>Time Series Forecasting</div>', unsafe_allow_html=True)
    start_dt = st.text_input("Start Date", "2025-01-01")
    end_dt = st.text_input("End Date", "2025-12-31")
    if st.button("Load Time Series"):
        series = load_workload_series(start_dt, end_dt)
        with st.expander("Time Series Data", expanded=True):
            st.dataframe(series.head())

elif option == "CNN Fraud Detection":
    st.markdown('<div class="section-header"><i class="bi bi-shield-lock-fill section-icon icon-fraud"></i>CNN Fraud Detection Training</div>', unsafe_allow_html=True)
    if st.button("Train CNN Fraud Model"):
        cfg = TrainConfig()
        train_cnn(cfg)
        #st.success("CNN fraud detection training complete.")

elif option == "LLM Chatbot":
    st.markdown('<div class="section-header"><i class="bi bi-robot section-icon icon-chatbot"></i>LLM Chatbot</div>', unsafe_allow_html=True)
    query = st.text_area("Enter your query:", "What is the leave policy?")
    if st.button("Ask Chatbot"):
        session_id = 1  # For demo purposes, use a fixed session_id
        response = gpt_chatbot_db(session_id, query)
        with st.expander("Chatbot Response", expanded=True):
            st.markdown(response)

#elif option == "LLM Summarization":
#   st.header("LLM Summarization")
#    text = st.text_area("Enter text to summarize:", "What is the leave policy?")
#    if st.button("Summarize"):
#        summary = gpt_summarize(text)
#        with st.expander("Summary", expanded=True):
#            st.markdown(summary)

# Add Bootstrap Icons CDN for icons to work
st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">', unsafe_allow_html=True)
