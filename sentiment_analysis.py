from .db_utils import read_sql_df
import streamlit as st
def train_sentiment(df, text_col="review", label_col="sentiment"):
  # Dummy implementation: replace with actual ML pipeline
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.naive_bayes import MultinomialNB
  from sklearn.pipeline import Pipeline
  X = df[text_col]
  y = df[label_col]
  model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
  ], memory=None)
  model.fit(X, y)
  return model


SENTIMENT_SQL = """
SELECT
  engagement_event_id AS id,
  JSON_UNQUOTE(JSON_EXTRACT(payload_json, '$.q1')) AS review,
  CASE
    WHEN sentiment_score >= 0.2 THEN 'positive'
    WHEN sentiment_score <= -0.2 THEN 'negative'
    ELSE 'neutral'
  END AS sentiment
FROM engagement_event
WHERE event_type = 'SurveyResponse'
  AND JSON_EXTRACT(payload_json, '$.q1') IS NOT NULL;
"""

def train_sentiment_from_db():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import pandas as pd

    df = read_sql_df(SENTIMENT_SQL)
    df = df.dropna(subset=["review"])
    model = train_sentiment(df, text_col="review", label_col="sentiment")

    # Predict on training data (for demonstration; ideally use a test set)
    y_true = df["sentiment"]
    y_pred = model.predict(df["review"])

    # 1) Overall Metrics
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    print("\nAccuracy:", acc)
    print("\nClassification Report:")
    print(pd.DataFrame(report).transpose())

    # 2) Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=["positive", "neutral", "negative"])
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["positive", "neutral", "negative"], yticklabels=["positive", "neutral", "negative"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

    # 3) Distribution of Sentiments
    # Bar chart: counts per predicted class
    pred_counts = pd.Series(y_pred).value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
    pred_counts.plot(kind="bar", color=["#8fd175", "#ffe066", "#ff686b"])
    plt.title("Predicted Sentiment Distribution")
    plt.ylabel("Count")
    plt.xlabel("Sentiment")
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

    # Stacked bar: true vs. predicted per class
    cm_df = pd.DataFrame(cm, index=["True_Pos", "True_Neu", "True_Neg"], columns=["Pred_Pos", "Pred_Neu", "Pred_Neg"])
    fig, ax = plt.subplots()
    colors = ["#8fd175", "#ffe066", "#ff686b"]
    cm_df.plot(kind="bar", stacked=True, color=colors, ax=ax)
    ax.set_title("True vs. Predicted Sentiment (Stacked Bar)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig)

    # Pie chart: share of each predicted sentiment
    pred_counts.index = ["positive", "neutral", "negative"]
    fig, ax = plt.subplots()
    pred_counts.plot(kind="pie", labels=pred_counts.index, autopct="%1.1f%%", colors=["#8fd175", "#ffe066", "#ff686b"], ax=ax)
    ax.set_title("Predicted Sentiment Share")
    ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig)
