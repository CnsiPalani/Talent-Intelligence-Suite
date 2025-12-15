
"""
Quick dashboard for demo:
- EDA metrics
- Attrition risk scoring
- Resume-JD matching
Run: streamlit run src/dashboard_app.py
"""
import streamlit as st
import pandas as pd
from pathlib import Path
from .config import PATHS
from .transformers_skill_matching import match_resume_to_jd

st.set_page_config(page_title="T-IQ Dashboard", layout="wide")

st.title("Talent Intelligence & Workforce Optimization (T-IQ)")

st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload employee CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Data Preview")
    st.dataframe(df.head())
    st.write("Rows:", len(df))

st.sidebar.header("Resume ↔ JD Matching")
resume = st.sidebar.text_area("Resume Text")
jd = st.sidebar.text_area("Job Description Text")
if st.sidebar.button("Compute Match"):
    if resume and jd:
        score = match_resume_to_jd(resume, jd)
        st.metric("Resume–JD Similarity", f"{score:.3f}")
    else:
        st.warning("Provide both resume and JD text.")

st.sidebar.header("Reports")
reports = sorted(Path(PATHS.reports_dir).glob("*.png"))
for rp in reports:
    st.image(str(rp), caption=rp.name)
