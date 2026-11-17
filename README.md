# RoBERTa-Based Sentiment Analysis System

**NLP · RoBERTa · Deep Learning**

The system reads text (reviews, tweets, comments) and classifies each one as **Positive**, **Negative** or **Neutral**. It uses **RoBERTa** (Robustly Optimized BERT Pretraining Approach). When it analyses many texts together, it also produces **insights**:

- how the sentiments are split
- a net sentiment score
- the most common words in each class
- a sentiment breakdown for each product

---

## 1. Folder structure

```
sentiment analysis system/
│
├── README.md                 ← you are here: setup and how to run
├── requirements.txt          ← Python libraries to install
├── config.py                 ← ALL settings in one place (model names, paths, training params)
├── main.py                   ← command-line entry point (predict / analyze / evaluate / train)
│
├── src/                      ← core source code (the ML pipeline)
│   ├── preprocessing.py      ← STEP 1: clean the text (light cleaning for transformers)
│   ├── predictor.py          ← STEP 2: RoBERTa model → Positive / Negative / Neutral
│   ├── insights.py           ← STEP 3: statistics, keywords, charts from many predictions
│   ├── evaluate.py           ← accuracy, precision, recall, F1, confusion matrix
│   └── train.py              ← (optional) fine-tune roberta-base on the TweetEval dataset
│
├── app/
│   └── streamlit_app.py      ← web UI for the live demo
│
├── data/
│   └── sample_reviews.csv    ← 30 labelled product reviews for the demo and evaluation
│
├── models/                   ← the fine-tuned model is saved here (after running train)
├── outputs/                  ← generated predictions, reports and charts
└── docs/
    └── INTERVIEW_GUIDE.md    ← how to explain the project, demo script, likely Q&A
```

## 2. Setup (one time, about 10 minutes)

```bash
cd "d:\sentiment analysis system"
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

> The first run downloads the RoBERTa model (about 500 MB). Later runs load it from the local cache.

## 3. How to run

| What | Command |
|---|---|
| Predict one sentence | `python main.py predict "I love this product!"` |
| Live typing demo | `python main.py interactive` |
| Analyse a CSV and get insights and charts | `python main.py analyze data/sample_reviews.csv` |
| Measure accuracy | `python main.py evaluate` |
| **Web demo** | `streamlit run app/streamlit_app.py` |
| Fine-tune RoBERTa yourself (optional) | `python main.py train` |

Results are saved to `outputs/`:

- `predictions.csv`
- `insights_report.txt`
- `sentiment_distribution.png`
- `sentiment_by_product.png`
- `evaluation_report.txt`
- `confusion_matrix.png`

## 4. How it works

```
Raw text ─► Cleaning ─► RoBERTa Tokenizer ─► RoBERTa encoder (12 layers)
        ─► Classification head (768 → 3) ─► Softmax ─► Positive / Neutral / Negative + confidence
                                                         │
                                         many texts ─────┴─► Insights (distribution, keywords, charts)
```

- **Default model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`. This is RoBERTa-base already fine-tuned for 3-class sentiment, so it works immediately.
- **Own training:** `python main.py train` fine-tunes plain `roberta-base` and saves it to `models/`. After that, the system uses your own model automatically.
