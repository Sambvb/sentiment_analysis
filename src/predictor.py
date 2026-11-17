"""
src/predictor.py
================
STEP 2 OF THE PIPELINE: THE ROBERTA SENTIMENT MODEL  (the heart of the project)

What this file does:
    1. Loads a RoBERTa model + its tokenizer from Hugging Face.
    2. Converts text -> tokens -> numbers (token IDs).
    3. Passes them through RoBERTa to get 3 raw scores ("logits").
    4. Applies SOFTMAX to turn logits into probabilities that sum to 1.
    5. Picks the class with the highest probability as the final sentiment.

How the data flows (good to draw on a whiteboard in an interview):

    "The food was great!"
          |  tokenizer
          v
    [<s>, The, food, was, great, !, </s>]  -> [0, 133, 689, 21, 372, 328, 2]
          |  RoBERTa (12 transformer layers, 125M parameters)
          v
    vector for <s> token (768 numbers = meaning of whole sentence)
          |  classification head (linear layer 768 -> 3)
          v
    logits  [-2.1, -0.4, 3.0]
          |  softmax
          v
    probs   [0.01, 0.03, 0.96]  ->  "Positive" (96% confidence)
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.utils import logging as hf_logging

# Hide Hugging Face's technical loading messages so the demo output stays clean
hf_logging.set_verbosity_error()

import config
from src.preprocessing import clean_text, clean_texts


class SentimentPredictor:
    """
    Wraps a RoBERTa model so the rest of the project can simply call:

        predictor = SentimentPredictor()
        predictor.predict("I love this phone")
        -> {'text': ..., 'sentiment': 'Positive', 'confidence': 0.98,
            'scores': {'Negative': 0.01, 'Neutral': 0.01, 'Positive': 0.98}}
    """

    def __init__(self, model_name_or_path: str = None):
        """
        model_name_or_path:
            - None (default)  -> use our fine-tuned model if train.py was run,
                                 otherwise the pretrained Cardiff NLP RoBERTa.
            - any HF name or local folder path -> load that instead.
        """
        if model_name_or_path is None:
            if config.FINETUNED_MODEL_DIR.exists():
                model_name_or_path = str(config.FINETUNED_MODEL_DIR)
            else:
                model_name_or_path = config.PRETRAINED_MODEL
        self.model_name = model_name_or_path

        # Use the GPU if one is available, otherwise the CPU (works fine, a bit slower)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[predictor] Loading model '{self.model_name}' on {self.device} ...")
        # First run downloads ~500 MB; after that it is cached on disk.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()   # evaluation mode: turns off dropout -> stable predictions

        # The model stores its own label names, e.g. {0: 'negative', 1: 'neutral', ...}.
        # We map them to our nicely capitalised names from config.LABELS.
        self.id2label = {
            int(i): name.capitalize() for i, name in self.model.config.id2label.items()
        }
        print("[predictor] Model ready. Labels:", list(self.id2label.values()))

    # --------------------------------------------------------------
    # Internal helper: run the model on a list of already-cleaned texts
    # --------------------------------------------------------------
    @torch.no_grad()   # we are not training -> no gradients -> faster & less memory
    def _predict_probabilities(self, texts: list) -> torch.Tensor:
        # 1. Tokenize: text -> token IDs. padding/truncation make all texts same length.
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.MAX_LENGTH,
            return_tensors="pt",            # return PyTorch tensors
        ).to(self.device)

        # 2. Forward pass through RoBERTa -> logits of shape [batch_size, 3]
        logits = self.model(**encoded).logits

        # 3. Softmax -> probabilities between 0 and 1 that sum to 1 for each text
        return torch.softmax(logits, dim=-1).cpu()

    def _format_result(self, original_text: str, probs: torch.Tensor) -> dict:
        """Turn a probability vector into a readable dictionary."""
        best_id = int(torch.argmax(probs))
        return {
            "text": original_text,
            "sentiment": self.id2label[best_id],
            "confidence": round(float(probs[best_id]), 4),
            "scores": {self.id2label[i]: round(float(p), 4) for i, p in enumerate(probs)},
        }

    # --------------------------------------------------------------
    # Public methods used by the rest of the project
    # --------------------------------------------------------------
    def predict(self, text: str) -> dict:
        """Predict the sentiment of ONE text."""
        probs = self._predict_probabilities([clean_text(text)])[0]
        return self._format_result(text, probs)

    def predict_batch(self, texts: list) -> list:
        """
        Predict the sentiment of MANY texts.
        Texts are processed in chunks of config.BATCH_SIZE (much faster than
        one-by-one, and avoids running out of memory on large files).
        """
        texts = list(texts)
        cleaned = clean_texts(texts)
        results = []
        for start in range(0, len(texts), config.BATCH_SIZE):
            chunk_clean = cleaned[start:start + config.BATCH_SIZE]
            chunk_orig = texts[start:start + config.BATCH_SIZE]
            probs = self._predict_probabilities(chunk_clean)
            for original, p in zip(chunk_orig, probs):
                results.append(self._format_result(original, p))
        return results
