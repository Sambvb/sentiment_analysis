"""
src/evaluate.py
===============
MODEL EVALUATION: "How good is our model?"

What this file does:
    Runs the model on a CSV that already has the CORRECT answer
    (a 'label' column), compares predictions vs truth, and reports:

    - Accuracy  : % of texts classified correctly.
    - Precision : of everything the model CALLED "Positive", how many really were?
    - Recall    : of all texts that REALLY are "Positive", how many did it find?
    - F1-score  : harmonic mean of precision & recall (one balanced number).
    - Confusion matrix : a grid showing WHICH classes get mixed up
                         (usually Neutral is confused most - it is the hardest class).

    Outputs saved in outputs/:
        evaluation_report.txt, confusion_matrix.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import config


def evaluate(predictor, csv_path=config.SAMPLE_DATA_FILE, text_col="text", label_col="label"):
    """Evaluate `predictor` (a SentimentPredictor) on a labelled CSV file."""
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"CSV must contain a '{label_col}' column with true labels.")

    # Normalise labels, e.g. "positive " -> "Positive"
    y_true = df[label_col].astype(str).str.strip().str.capitalize().tolist()
    y_pred = [r["sentiment"] for r in predictor.predict_batch(df[text_col].tolist())]

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=config.LABELS, digits=3, zero_division=0)

    text = (f"Model     : {predictor.model_name}\nData      : {csv_path}\n"
            f"Samples   : {len(df)}\nAccuracy  : {acc:.2%}\n\n{report}")
    print(text)

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    (config.OUTPUT_DIR / "evaluation_report.txt").write_text(text, encoding="utf-8")

    # Confusion matrix picture
    cm = confusion_matrix(y_true, y_pred, labels=config.LABELS)
    disp = ConfusionMatrixDisplay(cm, display_labels=config.LABELS)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix (accuracy {acc:.1%})")
    plt.savefig(config.OUTPUT_DIR / "confusion_matrix.png", bbox_inches="tight")
    plt.close()
    print(f"Saved report + confusion matrix to {config.OUTPUT_DIR}")
    return acc
