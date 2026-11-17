"""
src/train.py
============
(OPTIONAL) FINE-TUNING ROBERTA OURSELVES

What this file does:
    Shows the complete deep-learning TRAINING workflow:
      1. Download a labelled sentiment dataset  -> TweetEval "sentiment"
         (public benchmark: tweets labelled 0=Negative, 1=Neutral, 2=Positive)
      2. Tokenize the text for RoBERTa
      3. Load plain "roberta-base" and add a new classification head with 3 outputs
      4. Fine-tune with the Hugging Face Trainer (AdamW optimizer, cross-entropy loss)
      5. Evaluate on validation data (accuracy + macro F1)
      6. Save the model to models/roberta-finetuned-sentiment/

    After this runs, SentimentPredictor automatically uses the saved model.

Key concept - TRANSFER LEARNING (interview point):
    roberta-base already "understands English" from pre-training on 160 GB of
    text (masked-language-modelling). We only need a small labelled dataset and
    a few epochs to teach it the specific task of sentiment classification.

Time: ~20-40 min on a laptop CPU with the default small subset, ~3 min on a GPU.
      Change TRAIN_SAMPLES / EPOCHS in config.py to make it faster or better.
"""

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

import config
from src.preprocessing import clean_text


def compute_metrics(eval_pred):
    """Called by the Trainer after each epoch to report accuracy and F1."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro")}


def train():
    # ---------- 1. Load dataset (small subset to keep it laptop-friendly) ----------
    print("Downloading TweetEval sentiment dataset ...")
    ds = load_dataset("tweet_eval", "sentiment")
    train_ds = ds["train"].shuffle(seed=42).select(range(config.TRAIN_SAMPLES))
    eval_ds = ds["validation"].shuffle(seed=42).select(range(config.EVAL_SAMPLES))

    # ---------- 2. Tokenize ----------
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)

    def tokenize(batch):
        texts = [clean_text(t) for t in batch["text"]]
        return tokenizer(texts, truncation=True, max_length=config.MAX_LENGTH)

    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)

    # ---------- 3. Model: roberta-base + new 3-class head ----------
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_MODEL,
        num_labels=len(config.LABELS),
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )

    # ---------- 4. Training configuration ----------
    args = TrainingArguments(
        output_dir=str(config.MODELS_DIR / "checkpoints"),
        num_train_epochs=config.EPOCHS,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.TRAIN_BATCH_SIZE,
        weight_decay=0.01,              # small regularisation to reduce overfitting
        warmup_ratio=0.1,               # slowly increase LR at start -> stable training
        eval_strategy="epoch",          # evaluate after each epoch
        save_strategy="no",             # we save the final model manually below
        logging_steps=20,
        report_to="none",               # no external logging services
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer),   # dynamic padding per batch
        compute_metrics=compute_metrics,
    )

    # ---------- 5. Train + evaluate ----------
    trainer.train()
    metrics = trainer.evaluate()
    print("Final validation metrics:", metrics)

    # ---------- 6. Save ----------
    trainer.save_model(str(config.FINETUNED_MODEL_DIR))
    tokenizer.save_pretrained(str(config.FINETUNED_MODEL_DIR))
    print(f"Fine-tuned model saved to {config.FINETUNED_MODEL_DIR}")


if __name__ == "__main__":
    train()
