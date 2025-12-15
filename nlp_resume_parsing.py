
"""
Resume parsing:
- Clean text
- Extract simple fields via regex/heuristics (placeholder)
- Vectorize for downstream tasks
"""
import re
from typing import Dict, List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .data_cleaning import basic_text_clean
from .logging_utils import get_logger

logger = get_logger("nlp_resume")

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\s\-]{7,}\d")

def extract_fields(text: str) -> Dict[str, str]:
    clean = basic_text_clean(text)
    email = (EMAIL_RE.search(text) or [None]).group(0) if EMAIL_RE.search(text) else None
    phone = (PHONE_RE.search(text) or [None]).group(0) if PHONE_RE.search(text) else None
    return {"clean_text": clean, "email": email, "phone": phone}

def build_tfidf(corpus: List[str], max_features: int = 5000) -> TfidfVectorizer:
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    vec.fit(corpus)
    logger.info(f"TF-IDF fitted with vocab size {len(vec.vocabulary_)}")
    return vec

def transform_texts(texts: List[str], vec: TfidfVectorizer):
    return vec.transform(texts)
