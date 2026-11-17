"""
src/preprocessing.py
====================
STEP 1 OF THE PIPELINE: TEXT CLEANING

What this file does:
    Takes raw text (a review, tweet, comment) and cleans it BEFORE it goes
    to the RoBERTa model.

Why we keep cleaning LIGHT (interview point):
    Classic ML (TF-IDF + Logistic Regression) needs heavy cleaning:
    lower-casing, removing stop-words, stemming...
    Transformers like RoBERTa are DIFFERENT. They understand context, so words
    like "not", capital letters ("AMAZING") and punctuation ("!!!") actually
    carry sentiment. Removing them would HURT accuracy.
    So we only:
        1. replace @usernames with "@user"   (names carry no sentiment)
        2. replace URLs with "http"          (links carry no sentiment)
        3. collapse extra whitespace
    Steps 1 and 2 match exactly how the pretrained Cardiff NLP model was trained.
"""

import re

# Regular expressions compiled once (faster than compiling on every call)
_USER_PATTERN = re.compile(r"@\w+")                    # e.g. @john_doe
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")    # e.g. https://abc.com
_SPACES_PATTERN = re.compile(r"\s+")                   # multiple spaces/newlines


def clean_text(text) -> str:
    """
    Clean a single piece of text.

    Example:
        >>> clean_text("@amazon   LOVE it!!  https://amzn.to/xyz")
        '@user LOVE it!! http'
    """
    # Handle missing values (empty cells in a CSV come in as NaN / None)
    if text is None or (isinstance(text, float)):
        return ""

    text = str(text)
    text = _USER_PATTERN.sub("@user", text)
    text = _URL_PATTERN.sub("http", text)
    text = _SPACES_PATTERN.sub(" ", text)
    return text.strip()


def clean_texts(texts) -> list:
    """Clean a list of texts (just calls clean_text on each one)."""
    return [clean_text(t) for t in texts]
