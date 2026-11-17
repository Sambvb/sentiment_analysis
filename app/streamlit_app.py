"""
app/streamlit_app.py
====================
WEB DEMO (use this to show the interviewer)

Run from the project root:
    streamlit run app/streamlit_app.py
A browser tab opens at http://localhost:8501

Tabs in the app:
    1. Single Text   - type any sentence, see sentiment + confidence bars
    2. Batch / CSV   - upload a CSV (or use the sample) -> table, charts, keywords
    3. About         - short explanation of how the system works
"""

import sys
from pathlib import Path

# Make "config" and "src" importable when Streamlit runs this file from app/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

import config
from src import insights
from src.predictor import SentimentPredictor

st.set_page_config(page_title="RoBERTa Sentiment Analysis", page_icon="💬", layout="wide")


@st.cache_resource   # load the model ONCE and reuse it (otherwise reloads on every click)
def load_predictor():
    return SentimentPredictor()


predictor = load_predictor()
EMOJI = {"Positive": "😊", "Neutral": "😐", "Negative": "😠"}

st.title("💬 RoBERTa-Based Sentiment Analysis System")
st.caption(f"Model: `{predictor.model_name}`  •  Classes: Positive / Negative / Neutral")

tab_single, tab_batch, tab_about = st.tabs(["🔍 Single Text", "📊 Batch / CSV Analysis", "ℹ️ About"])

# ------------------------------------------------------------------
# TAB 1: single text
# ------------------------------------------------------------------
with tab_single:
    examples = [
        "Absolutely love this phone, the camera is amazing!",
        "The delivery was late and the box was damaged. Very disappointed.",
        "The package arrived on Tuesday.",
        "I was not happy with the service.",
        "Honestly not bad, I would buy it again.",
        "Great, another update that breaks everything. Just perfect.",
    ]
    choice = st.selectbox("Pick an example or write your own below:", [""] + examples)
    text = st.text_area("Text to analyse", value=choice, height=100)

    if st.button("Analyse sentiment", type="primary") and text.strip():
        r = predictor.predict(text)
        st.subheader(f"{EMOJI[r['sentiment']]} {r['sentiment']}  —  {r['confidence']:.1%} confident")
        for label in config.LABELS:
            score = r["scores"].get(label, 0.0)
            st.write(f"**{label}**  {score:.1%}")
            st.progress(score)

# ------------------------------------------------------------------
# TAB 2: batch analysis of a CSV
# ------------------------------------------------------------------
with tab_batch:
    uploaded = st.file_uploader("Upload a CSV file with a text column", type="csv")
    use_sample = st.checkbox("Or use the sample dataset (data/sample_reviews.csv)", value=uploaded is None)

    df = pd.read_csv(uploaded) if uploaded else (pd.read_csv(config.SAMPLE_DATA_FILE) if use_sample else None)

    if df is not None:
        col1, col2 = st.columns(2)
        text_col = col1.selectbox("Text column", df.columns,
                                  index=list(df.columns).index("text") if "text" in df.columns else 0)
        cat_options = ["(none)"] + [c for c in df.columns if c != text_col]
        category_col = col2.selectbox("Group by (optional)", cat_options,
                                      index=cat_options.index("product") if "product" in cat_options else 0)
        category_col = None if category_col == "(none)" else category_col

        if st.button("Run analysis", type="primary"):
            with st.spinner(f"Running RoBERTa on {len(df)} texts ..."):
                results = predictor.predict_batch(df[text_col].fillna("").tolist())
            out = insights.predictions_to_dataframe(results, df.drop(columns=[text_col]))
            dist = insights.sentiment_distribution(out)

            # Key numbers
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("😊 Positive", f"{dist.loc['Positive', 'percentage']}%")
            m2.metric("😐 Neutral", f"{dist.loc['Neutral', 'percentage']}%")
            m3.metric("😠 Negative", f"{dist.loc['Negative', 'percentage']}%")
            m4.metric("Net Sentiment Score",
                      f"{dist.loc['Positive', 'percentage'] - dist.loc['Negative', 'percentage']:+.1f}")

            c1, c2 = st.columns(2)
            c1.subheader("Distribution")
            c1.bar_chart(dist["count"])
            if category_col:
                c2.subheader(f"Sentiment % by {category_col}")
                c2.bar_chart(insights.category_breakdown(out, category_col))

            k1, k2 = st.columns(2)
            k1.subheader("Top words in Positive texts")
            k1.dataframe(pd.DataFrame(insights.top_keywords(out, "Positive"), columns=["word", "count"]))
            k2.subheader("Top words in Negative texts")
            k2.dataframe(pd.DataFrame(insights.top_keywords(out, "Negative"), columns=["word", "count"]))

            st.subheader("All predictions")
            st.dataframe(out, use_container_width=True)
            st.download_button("Download predictions CSV", out.to_csv(index=False), "predictions.csv")

# ------------------------------------------------------------------
# TAB 3: about
# ------------------------------------------------------------------
with tab_about:
    st.markdown("""
**Pipeline**

`Raw text → Cleaning → RoBERTa tokenizer → RoBERTa (12 transformer layers) → Classification head → Softmax → Positive / Neutral / Negative`

- **Model:** RoBERTa-base (125M parameters), fine-tuned for 3-class sentiment.
- **Why RoBERTa?** Improves on BERT by training longer on 10× more data (160 GB),
  using dynamic masking and dropping the Next-Sentence-Prediction task.
- **Insights:** distribution, net sentiment score, keywords per class, per-category breakdown.
""")
