# main.py (add --source option)
import argparse
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
from src.llm_chatbot import gpt_summarize, gpt_chatbot

logger = get_logger("main")

def run_attrition(source: str, input_csv: Path = None):
    cfg = AttritionConfig(
        cat_cols=["Department", "JobRole", "status", "location_code"],
        num_cols=["YearsAtCompany", "MonthlyIncome"],
        target_col="Attrition"
    )
    if source == "db":
        pipe = train_attrition_model_from_db(cfg)
    else:
        df = pd.read_csv(input_csv)
        pipe = train_attrition_model(df, cfg)
    Path(PATHS.models_dir).mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(pipe, Path(PATHS.models_dir) / "attrition_pipeline.joblib")
    logger.info("Saved attrition pipeline")

def run_performance(source: str, input_csv: Path = None):
    cfg = PerfConfig(
        cat_cols=["Department", "JobRole"],
        num_cols=["MonthlyIncome", "YearsAtCompany"],
        target_col="PerformanceScore"
    )
    if source == "db":
        metrics = train_performance_model_from_db(cfg)
    else:
        df = pd.read_csv(input_csv)
        metrics = train_performance_model(df, cfg)
    logger.info(f"Performance model metrics: {metrics}")

def run_sentiment(source: str, input_csv: Path = None):
    if source == "db":
        pipe = train_sentiment_from_db()
    else:
        df = pd.read_csv(input_csv)
        pipe = train_sentiment(df)
    import joblib, os
    os.makedirs(PATHS.models_dir, exist_ok=True)
    joblib.dump(pipe, Path(PATHS.models_dir) / "sentiment_pipeline.joblib")
    logger.info("Saved sentiment pipeline")

def run_eda(source: str, input_csv: Path = None):
    if source == "db":
        eda_overview_db()
    else:
        df = pd.read_csv(input_csv)
        eda_overview(df, report_name=input_csv.stem)

# Resume-JD matching
def run_resume_jd_match():
    results_df = match_all_resumes_to_jds()
    logger.info(f"Resume–JD matching results:\n{results_df}")
    print(f"Resume–JD matching results:\n{results_df}")

# Interview summarization
def run_interview_summarization():
    # If you want to summarize a single transcript, use summarize_transcript
    # If you have a function in src.interview_summarization for all transcripts, import and use it here.
    # For now, let's use summarize_transcript with a placeholder transcript or argument.
    summary = summarize_all_interview_transcripts_from_uri()
    logger.info(f"Interview summary: {summary}")
    print(f"Interview summary: {summary}")

# Time series forecasting
def run_time_series(start_dt: str, end_dt: str):
    series = load_workload_series(start_dt, end_dt)
    logger.info(f"Loaded time series: {series.head()}")
    print(series.head())

# CNN fraud detection
def run_cnn_fraud():
    cfg = TrainConfig()
    train_cnn(cfg)
    logger.info("CNN fraud detection training complete.")
    print("CNN fraud detection training complete.")


# LLM chatbot (direct GPT call)
def run_llm_chatbot(query: str):
    response = gpt_chatbot(query)
    logger.info(f"LLM chatbot response: {response}")
    print(f"LLM chatbot response: {response}")

# LLM summarization (direct GPT call)
def run_llm_summarize(text: str):
    summary = gpt_summarize(text)
    logger.info(f"LLM summary: {summary}")
    print(f"LLM summary: {summary}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["eda", "attrition", "performance", "sentiment", "resume_jd_match", "interview_summarization", "time_series", "cnn_fraud", "llm_chatbot", "llm_summarize"])
    parser.add_argument("--source", default="db", choices=["db", "csv"])
    parser.add_argument("--input", type=Path, help="Path to CSV if source=csv")
    parser.add_argument("--resume", type=str, help="Resume text for matching")
    parser.add_argument("--jd", type=str, help="Job description text for matching")
    parser.add_argument("--transcript", type=str, help="Interview transcript for summarization")
    parser.add_argument("--start_dt", type=str, help="Start date for time series",default="2025-01-01")
    parser.add_argument("--end_dt", type=str, help="End date for time series",default="2025-12-31")
    parser.add_argument("--img_dir", type=Path, help="Image directory for CNN fraud detection")
    parser.add_argument("--query", type=str, help="Query for LLM chatbot",default="What is the leave policy?")
    parser.add_argument("--text", type=str, help="Text for LLM summarization",default="What is the leave policy?")
    args = parser.parse_args()

    if args.task == "eda":
        logger.info("args.task 'eda' ")
        run_eda(args.source, args.input)
    elif args.task == "attrition":
        logger.info("args.task 'attrition' ")
        run_attrition(args.source, args.input)
    elif args.task == "performance":
        logger.info("args.task 'performance' ")
        run_performance(args.source, args.input)
    elif args.task == "sentiment":
        logger.info("args.task 'sentiment' ")
        run_sentiment(args.source, args.input)
    elif args.task == "resume_jd_match":
        run_resume_jd_match()
    elif args.task == "interview_summarization":
        run_interview_summarization()
    elif args.task == "time_series":
        run_time_series(args.start_dt, args.end_dt)
    elif args.task == "cnn_fraud":
        run_cnn_fraud()
    elif args.task == "llm_chatbot":
        run_llm_chatbot(args.query)
    elif args.task == "llm_summarize":
        if args.text:
            run_llm_summarize(args.text)
        else:
            print("--text argument required for llm_summarize task.")
