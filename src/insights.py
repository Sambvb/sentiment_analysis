"""
src/insights.py
===============
STEP 3 OF THE PIPELINE: EXTRACTING INSIGHTS FROM PREDICTIONS

What this file does:
    A single prediction ("this review is Negative") is not very useful to a
    business. The VALUE comes from analysing thousands of texts together:
        - What % of customers are happy / unhappy / neutral?
        - How confident is the model on average?
        - Which words appear most in NEGATIVE texts? (-> what to fix)
        - Which words appear most in POSITIVE texts? (-> what customers love)
        - If the data has a category column (e.g. product), which category
          gets the most negative feedback?

    This covers the resume point:
    "Extracted meaningful insights from textual data to understand opinions,
     emotions, and user perspectives."
"""

import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")           # draw charts without needing a screen (saves to file)
import matplotlib.pyplot as plt
import pandas as pd

import config

# Colours used consistently in every chart
COLORS = {"Negative": "#e74c3c", "Neutral": "#95a5a6", "Positive": "#2ecc71"}

# Very common words that carry no meaning; ignored when finding keywords.
# (We only remove them for the KEYWORD insight, never before the model.)
STOP_WORDS = set("""
a an the and or but if is are was were be been being am i me my we our you your he
she it its they them their this that these those to of in on at for with as by from
so very just really too than then there here have has had do does did not no can
could would should will shall may might also only about into over after before all
any some more most other such own same what which who whom when where why how out up
user http s t im ive dont didnt its it's i'm i've don't didn't one two three nothing less get got
""".split())


def predictions_to_dataframe(results: list, extra_columns: pd.DataFrame = None) -> pd.DataFrame:
    """
    Convert the list of dicts from SentimentPredictor.predict_batch() into a table.
    extra_columns: original CSV columns (e.g. product, date) to keep alongside.
    """
    df = pd.DataFrame({
        "text": [r["text"] for r in results],
        "sentiment": [r["sentiment"] for r in results],
        "confidence": [r["confidence"] for r in results],
        "score_negative": [r["scores"].get("Negative") for r in results],
        "score_neutral": [r["scores"].get("Neutral") for r in results],
        "score_positive": [r["scores"].get("Positive") for r in results],
    })
    if extra_columns is not None:
        extra = extra_columns.drop(columns=["text"], errors="ignore").reset_index(drop=True)
        df = pd.concat([df, extra], axis=1)
    return df


def sentiment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of each sentiment class."""
    counts = df["sentiment"].value_counts().reindex(config.LABELS, fill_value=0)
    return pd.DataFrame({
        "count": counts,
        "percentage": (counts / max(len(df), 1) * 100).round(1),
    })


def top_keywords(df: pd.DataFrame, sentiment: str, n: int = 10) -> list:
    """Most frequent meaningful words inside texts of one sentiment class."""
    words = []
    for text in df.loc[df["sentiment"] == sentiment, "text"]:
        tokens = re.findall(r"[a-zA-Z']+", str(text).lower())
        words += [w for w in tokens if w not in STOP_WORDS and len(w) > 2]
    return Counter(words).most_common(n)


def category_breakdown(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    """% of each sentiment per category (e.g. per product). Rows sum to 100."""
    table = pd.crosstab(df[category_col], df["sentiment"], normalize="index") * 100
    return table.reindex(columns=config.LABELS, fill_value=0).round(1)


def build_report(df: pd.DataFrame, category_col: str = None) -> str:
    """Create a human-readable text report with all insights."""
    dist = sentiment_distribution(df)
    lines = ["=" * 60, "SENTIMENT ANALYSIS REPORT", "=" * 60,
             f"Total texts analysed : {len(df)}",
             f"Average confidence   : {df['confidence'].mean():.1%}", "",
             "Sentiment distribution:"]
    for label, row in dist.iterrows():
        lines.append(f"  {label:<9}: {int(row['count']):>5}  ({row['percentage']}%)")

    # Overall "Net Sentiment Score" = %positive - %negative  (range -100..+100)
    nss = dist.loc["Positive", "percentage"] - dist.loc["Negative", "percentage"]
    lines += ["", f"Net Sentiment Score (%Pos - %Neg): {nss:+.1f}"]

    for label in ["Positive", "Negative"]:
        kws = top_keywords(df, label)
        if kws:
            lines += ["", f"Top words in {label} texts:",
                      "  " + ", ".join(f"{w} ({c})" for w, c in kws)]

    if category_col and category_col in df.columns:
        lines += ["", f"Sentiment % by '{category_col}':",
                  category_breakdown(df, category_col).to_string()]

    # Least confident predictions = texts that are ambiguous / sarcastic / mixed
    lines += ["", "Least confident predictions (worth a human look):"]
    for _, row in df.nsmallest(3, "confidence").iterrows():
        lines.append(f"  [{row['sentiment']} {row['confidence']:.0%}] {str(row['text'])[:80]}")
    return "\n".join(lines)


def save_charts(df: pd.DataFrame, out_dir=config.OUTPUT_DIR, category_col: str = None) -> list:
    """Save a pie chart (+ category bar chart if available). Returns file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # --- Pie chart of overall distribution ---
    dist = sentiment_distribution(df)
    dist = dist[dist["count"] > 0]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(dist["count"], labels=dist.index, autopct="%1.1f%%",
           colors=[COLORS[l] for l in dist.index], startangle=90)
    ax.set_title("Sentiment Distribution")
    path = out_dir / "sentiment_distribution.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    saved.append(path)

    # --- Stacked bar chart per category ---
    if category_col and category_col in df.columns:
        table = category_breakdown(df, category_col)
        ax = table.plot(kind="barh", stacked=True, figsize=(8, 4),
                        color=[COLORS[l] for l in table.columns])
        ax.set_xlabel("% of texts"); ax.set_title(f"Sentiment by {category_col}")
        path = out_dir / f"sentiment_by_{category_col}.png"
        ax.figure.savefig(path, bbox_inches="tight"); plt.close(ax.figure)
        saved.append(path)
    return saved
