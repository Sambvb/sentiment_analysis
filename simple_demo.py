"""
simple_demo.py
==============
THE SIMPLEST VERSION OF THE PROJECT (about 15 lines of real code).

Uses an ALREADY-TRAINED RoBERTa model, so no training is needed.
You type a sentence, and it tells you Positive / Negative / Neutral.

Run:   python simple_demo.py
"""

from transformers import pipeline

# Load a RoBERTa model that is already trained for sentiment (3 classes).
# The first run downloads it (~500 MB); after that it loads from the cache in seconds.
classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
)

print("Type a sentence (or 'quit' to exit)")
while True:
    text = input("\nYour text: ")
    if text.lower() == "quit":
        break
    result = classifier(text)[0]                 # e.g. {'label': 'positive', 'score': 0.98}
    print(f"Sentiment: {result['label'].upper()}  (confidence {result['score']:.0%})")
