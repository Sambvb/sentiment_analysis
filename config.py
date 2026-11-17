"""
config.py
=========
CENTRAL SETTINGS FILE for the whole project.

Why this file exists:
    Instead of hard-coding paths and model names in many files, we keep
    them in ONE place. If you want to change the model or a folder, you
    change it here only.

Who uses it:
    Every file inside src/, main.py and app/streamlit_app.py imports from here.
"""

from pathlib import Path

# ------------------------------------------------------------------
# 1. FOLDER PATHS
# ------------------------------------------------------------------
# PROJECT_ROOT = the folder that contains this config.py file
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"        # input data (CSV files)
OUTPUT_DIR = PROJECT_ROOT / "outputs"   # results: predictions, charts, reports
MODELS_DIR = PROJECT_ROOT / "models"    # our own fine-tuned model is saved here

SAMPLE_DATA_FILE = DATA_DIR / "sample_reviews.csv"   # demo dataset with labels
FINETUNED_MODEL_DIR = MODELS_DIR / "roberta-finetuned-sentiment"

# ------------------------------------------------------------------
# 2. MODEL SETTINGS
# ------------------------------------------------------------------
# PRETRAINED_MODEL:
#   RoBERTa-base that has ALREADY been fine-tuned for 3-class sentiment
#   (by Cardiff NLP, on ~124 million tweets + TweetEval benchmark).
#   Using it means the system works immediately, no training required.
PRETRAINED_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# BASE_MODEL:
#   Plain RoBERTa-base (only language pre-training, no sentiment knowledge).
#   train.py starts from this and fine-tunes it on our sentiment data,
#   which demonstrates the full training workflow.
BASE_MODEL = "roberta-base"

# The three sentiment classes. The ORDER matters: index 0 = Negative, etc.
LABELS = ["Negative", "Neutral", "Positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}   # "Negative" -> 0
ID2LABEL = {i: label for i, label in enumerate(LABELS)}   # 0 -> "Negative"

# RoBERTa can read at most 512 tokens. Reviews/tweets are short, so 128
# is enough and keeps it fast.
MAX_LENGTH = 128

# How many texts we send to the model at once during batch prediction.
BATCH_SIZE = 16

# ------------------------------------------------------------------
# 3. TRAINING SETTINGS (only used by src/train.py)
# ------------------------------------------------------------------
TRAIN_SAMPLES = 3000     # use a small subset so training finishes on a laptop
EVAL_SAMPLES = 600
EPOCHS = 2
LEARNING_RATE = 2e-5     # standard learning rate for fine-tuning BERT-style models
TRAIN_BATCH_SIZE = 16
