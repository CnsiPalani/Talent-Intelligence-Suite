
from .db_utils import read_sql_df
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional
import streamlit as st
import pandas as pd
RESUMES_SQL = "SELECT candidate_resume_id, resume_raw_text FROM candidate_resume WHERE parsed_at IS NOT NULL;"
JDS_SQL = """
SELECT jd.job_description_id, jd.jd_raw_text
FROM job_description jd
JOIN job_posting jp ON jp.job_posting_id = jd.job_posting_id
WHERE jp.status = 'Open';
"""

def load_resumes_from_db():
    return read_sql_df(RESUMES_SQL)

def load_jds_from_db():
    return read_sql_df(JDS_SQL)

# Simple resume-JD matching using TF-IDF cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def match_resume_to_jd(resume_text: str, jd_text: str) -> float:
     corpus = [resume_text, jd_text]
     vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
     tfidf = vec.fit_transform(corpus)
     sim = cosine_similarity(tfidf[0], tfidf[1])[0][0]
     return sim

# Match all resumes to all open job descriptions and return a DataFrame of scores

def match_all_resumes_to_jds():
    """
    Main method: Runs resume↔JD matching, all visualizations, and returns results.
    Args:
        threshold: cutoff for binary match matrix
        k: top-K for bar/leaderboard
        show_all: if True, show all visualizations
    Returns:
        df: DataFrame of all similarity scores
        leaderboards: dict of per-JD leaderboards
    """
    threshold=0.5
    k=5 
    show_all=True
    resumes_df = load_resumes_from_db()
    jds_df = load_jds_from_db()
    results = []
    for _, resume_row in resumes_df.iterrows():
        for _, jd_row in jds_df.iterrows():
            score = match_resume_to_jd(resume_row['resume_raw_text'], jd_row['jd_raw_text'])
            results.append({
                'candidate_resume_id': resume_row['candidate_resume_id'],
                'job_description_id': jd_row['job_description_id'],
                'similarity_score': score
            })
    df = pd.DataFrame(results)

    if show_all:
        # 1. Similarity Heatmap (matrix view)
        st.subheader("Similarity Heatmap (Matrix View)")
        plot_similarity_heatmap(df)

        # 2. Top-K bar chart (per resume, show only for first resume)
        st.subheader(f"Top-{k} JD Matches for a Sample Resume")
        first_resume = df['candidate_resume_id'].unique()[0]
        plot_topk_bar(df[df['candidate_resume_id'] == first_resume], k=k, by='resume')

        # 3. Top-K bar chart (per JD, show only for first JD)
        st.subheader(f"Top-{k} Resume Matches for a Sample JD")
        first_jd = df['job_description_id'].unique()[0]
        plot_topk_bar(df[df['job_description_id'] == first_jd], k=k, by='jd')

        # 4. Similarity score distribution
        st.subheader("Similarity Score Distribution")
        plot_similarity_distribution(df)

        # 5. Thresholded match matrix (binary view)
        st.subheader(f"Thresholded Match Matrix (cutoff={threshold})")
        plot_thresholded_matrix(df, threshold=threshold)

        # 6. Per-JD leaderboard (show only for first JD)
        st.subheader(f"Leaderboard: Top-{k} Resumes for a Sample JD")
        leaderboards = leaderboard_per_jd(df, k=k)
        if first_jd in leaderboards:
            st.dataframe(leaderboards[first_jd])

        # 7. UMAP embedding scatter (optional)
        st.subheader("UMAP Embedding (Resumes vs. JDs)")
        try:
            plot_umap_embeddings(df, list(resumes_df['resume_raw_text']), list(jds_df['jd_raw_text']))
        except Exception as e:
            st.info(f"UMAP plot skipped: {e}")
    else:
        leaderboards = leaderboard_per_jd(df, k=k)

    return df, leaderboards

# --- Visualization Utilities for Resume-JD Matching ---
def plot_similarity_heatmap(df, resumes=None, jds=None, figsize=(10,6)):
    """Matrix view: resumes as rows, JDs as columns."""
    pivot = df.pivot(index='candidate_resume_id', columns='job_description_id', values='similarity_score')
    if resumes is not None:
        pivot = pivot.loc[resumes]
    if jds is not None:
        pivot = pivot[jds]
    plt.figure(figsize=figsize)
    sns.heatmap(pivot, annot=False, cmap='YlGnBu')
    plt.title('Resume-JD Similarity Heatmap')
    plt.xlabel('Job Description ID')
    plt.ylabel('Resume ID')
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

def plot_topk_bar(df, k=5, by='resume'):
    """Bar chart: Top-K matches per resume or per JD."""
    if by == 'resume':
        for rid, group in df.groupby('candidate_resume_id'):
            topk = group.nlargest(k, 'similarity_score')
            plt.figure(figsize=(6,2))
            sns.barplot(x='job_description_id', y='similarity_score', data=topk, palette='Blues_d')
            plt.title(f'Top-{k} JD Matches for Resume {rid}')
            plt.ylim(0,1)
            plt.show()
            st.pyplot(plt)
    elif by == 'jd':
        for jid, group in df.groupby('job_description_id'):
            topk = group.nlargest(k, 'similarity_score')
            plt.figure(figsize=(6,2))
            sns.barplot(x='candidate_resume_id', y='similarity_score', data=topk, palette='Greens_d')
            plt.title(f'Top-{k} Resume Matches for JD {jid}')
            plt.ylim(0,1)
            plt.show()
            st.pyplot(plt)

def plot_similarity_distribution(df):
    """Histogram + KDE of all similarity scores."""
    plt.figure(figsize=(6,3))
    sns.histplot(df['similarity_score'], bins=30, kde=True, color='purple')
    plt.title('Similarity Score Distribution')
    plt.xlabel('Similarity Score')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

def plot_thresholded_matrix(df, threshold=0.5, figsize=(10,6)):
    """Binary matrix view at a chosen cutoff."""
    pivot = df.pivot(index='candidate_resume_id', columns='job_description_id', values='similarity_score')
    binary = (pivot >= threshold).astype(int)
    plt.figure(figsize=figsize)
    sns.heatmap(binary, annot=True, cmap='Greens', cbar=False)
    plt.title(f'Thresholded Match Matrix (cutoff={threshold})')
    plt.xlabel('Job Description ID')
    plt.ylabel('Resume ID')
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

def leaderboard_per_jd(df, k=5):
    """Per-JD leaderboard: top-K resumes for each JD, with mini bar visualization."""
    import pandas as pd
    leaderboards = {}
    for jid, group in df.groupby('job_description_id'):
        topk = group.nlargest(k, 'similarity_score')[['candidate_resume_id', 'similarity_score']]
        # Add mini bar (text-based)
        topk['bar'] = topk['similarity_score'].apply(lambda x: '█' * int(x*10))
        leaderboards[jid] = topk
    return leaderboards

def plot_umap_embeddings(df, resume_texts, jd_texts, n_neighbors=10, min_dist=0.1, random_state=42):
    """Optional: UMAP scatter for cluster intuition (requires umap-learn)."""
    try:
        import umap
        from sklearn.feature_extraction.text import TfidfVectorizer
        all_texts = resume_texts + jd_texts
        labels = ['Resume']*len(resume_texts) + ['JD']*len(jd_texts)
        vec = TfidfVectorizer(max_features=1000)
        X = vec.fit_transform(all_texts)
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
        emb = reducer.fit_transform(X.toarray())
        plt.figure(figsize=(8,5))
        sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=labels, palette=['blue','orange'], alpha=0.7)
        plt.title('UMAP Embedding: Resumes vs. JDs')
        plt.xlabel('UMAP-1')
        plt.ylabel('UMAP-2')
        plt.tight_layout()
        plt.show()
        st.pyplot(plt)
    except ImportError:
        print('UMAP not installed. Run: pip install umap-learn')
