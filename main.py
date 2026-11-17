"""
main.py
=======
COMMAND-LINE ENTRY POINT - run everything from the terminal.

Commands:
    python main.py predict "I love this product!"     -> sentiment of one sentence
    python main.py interactive                        -> type sentences live (great for demos)
    python main.py analyze data/sample_reviews.csv    -> analyse a whole CSV + insights + charts
    python main.py evaluate                           -> accuracy, F1, confusion matrix
    python main.py train                              -> (optional) fine-tune roberta-base

For the visual web demo instead, run:
    streamlit run app/streamlit_app.py
"""

import argparse

import pandas as pd

import config


def cmd_predict(args):
    from src.predictor import SentimentPredictor
    result = SentimentPredictor().predict(args.text)
    print_result(result)


def cmd_interactive(args):
    from src.predictor import SentimentPredictor
    predictor = SentimentPredictor()
    print("\nType a sentence and press Enter (type 'quit' to exit).")
    while True:
        text = input("\n> ").strip()
        if text.lower() in {"quit", "exit", "q"}:
            break
        if text:
            print_result(predictor.predict(text))


def cmd_analyze(args):
    from src.predictor import SentimentPredictor
    from src import insights

    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns:
        raise SystemExit(f"Column '{args.text_col}' not found. Columns: {list(df.columns)}")

    results = SentimentPredictor().predict_batch(df[args.text_col].fillna("").tolist())
    out_df = insights.predictions_to_dataframe(results, df.drop(columns=[args.text_col]))

    # Save results, report and charts to outputs/
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    out_df.to_csv(config.OUTPUT_DIR / "predictions.csv", index=False)
    report = insights.build_report(out_df, category_col=args.category_col)
    (config.OUTPUT_DIR / "insights_report.txt").write_text(report, encoding="utf-8")
    charts = insights.save_charts(out_df, category_col=args.category_col)

    print(report)
    print(f"\nSaved: outputs/predictions.csv, outputs/insights_report.txt, "
          + ", ".join(f"outputs/{c.name}" for c in charts))


def cmd_evaluate(args):
    from src.predictor import SentimentPredictor
    from src.evaluate import evaluate
    evaluate(SentimentPredictor(), csv_path=args.csv)


def cmd_train(args):
    from src.train import train
    train()


def print_result(r: dict):
    """Pretty-print one prediction with a small text bar chart of the scores."""
    print(f"\nText       : {r['text']}")
    print(f"Sentiment  : {r['sentiment']}  (confidence {r['confidence']:.1%})")
    for label, score in r["scores"].items():
        print(f"  {label:<9} {'#' * int(score * 30):<30} {score:.1%}")


def main():
    parser = argparse.ArgumentParser(description="RoBERTa Sentiment Analysis System")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="Predict sentiment of one text")
    p.add_argument("text")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("interactive", help="Type texts one by one")
    p.set_defaults(func=cmd_interactive)

    p = sub.add_parser("analyze", help="Analyse a CSV file and generate insights")
    p.add_argument("csv", nargs="?", default=str(config.SAMPLE_DATA_FILE))
    p.add_argument("--text-col", default="text", help="column containing the text")
    p.add_argument("--category-col", default="product", help="optional column to group by")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("evaluate", help="Measure accuracy on a labelled CSV")
    p.add_argument("csv", nargs="?", default=str(config.SAMPLE_DATA_FILE))
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("train", help="Fine-tune roberta-base (optional)")
    p.set_defaults(func=cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
