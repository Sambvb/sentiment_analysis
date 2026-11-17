"""
src/  -  the core source code of the Sentiment Analysis System.

Modules (read them in this order to understand the pipeline):
    preprocessing.py  -> STEP 1: clean raw text
    predictor.py      -> STEP 2: run RoBERTa and get Positive / Negative / Neutral
    insights.py       -> STEP 3: turn many predictions into insights + charts
    evaluate.py       -> measure how accurate the model is on labelled data
    train.py          -> (optional) fine-tune plain RoBERTa ourselves
"""
