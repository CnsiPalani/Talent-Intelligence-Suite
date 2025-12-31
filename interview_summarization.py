from typing import List
from .db_utils import read_sql_df

# SQL to load transcript URIs from the interview table
INTERVIEW_URI_SQL = """
SELECT interview_id, transcript_uri
FROM interview
WHERE transcript_uri IS NOT NULL;
"""

def load_interview_uris_from_db(interview_types=None):
    """Load all interview transcript URIs from the database as a DataFrame. interview_types can be a list or None."""
    sql = """
    SELECT interview_id, interview_type, transcript_uri
    FROM interview
    WHERE transcript_uri IS NOT NULL
    """
    params = {}
    if interview_types:
        if not isinstance(interview_types, list):
            interview_types = [interview_types]
        sql += " AND interview_type IN :interview_types"
        params = {"interview_types": tuple(interview_types)}
    return read_sql_df(sql, params=params)

def chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    text = text.strip()
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]



def load_transcript_from_uri(uri: str) -> str:
    if uri.startswith("http://") or uri.startswith("https://"):
        import requests
        response = requests.get(uri)
        response.raise_for_status()
        return response.text
    elif uri.startswith("s3://"):
        import boto3
        from urllib.parse import urlparse
        s3 = boto3.client("s3")
        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        obj = s3.get_object(Bucket=bucket, Key=key, ExpectedBucketOwner="YOUR_BUCKET_OWNER_ID")
        return obj["Body"].read().decode("utf-8")
    else:
        with open(uri, "r", encoding="utf-8") as f:
            return f.read()

def summarize_transcript(transcript_text: str) -> str:
    """
    Dummy summarization function. Replace with actual summarization logic.
    """
    # For now, just return the first 200 characters as a placeholder summary
    return transcript_text[:200] + ("..." if len(transcript_text) > 200 else "")

# Summarize transcripts loaded from URIs, with optional interview_id filter
def summarize_all_interview_transcripts_from_uri(interview_type: str = None, interview_ids: List[int] = None):
    """
    Summarize transcripts for the given interview_ids. If interview_ids is None, summarize all.
    Optionally filter by interview_type.
    """
    df = load_interview_uris_from_db(interview_type)
    if interview_ids is not None:
        if not isinstance(interview_ids, list):
            interview_ids = [interview_ids]
        # Ensure both are int for correct filtering
        df["interview_id"] = df["interview_id"].astype(int)
        interview_ids = [int(i) for i in interview_ids]
        df = df[df["interview_id"].isin(interview_ids)]
    if df.empty:
        return "No transcripts found for the selected interviews."
    df["transcript_text"] = df["transcript_uri"].apply(load_transcript_from_uri)
    df["summary"] = df["transcript_text"].apply(summarize_transcript)
    # Format as markdown for display
    summary_md = ""
    for _, row in df.iterrows():
        summary_md += f"**Interview ID:** {row['interview_id']}\n\n"
        summary_md += f"{row['summary']}\n\n---\n"
    return summary_md