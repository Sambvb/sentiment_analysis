# Interview Guide: RoBERTa Sentiment Analysis System

## 1. 30-second pitch

> "I built a sentiment analysis system that classifies text as Positive, Negative or Neutral. It uses RoBERTa, a transformer model.
> The pipeline has three stages: light text preprocessing, RoBERTa inference with softmax confidence scores, and an insights layer.
> The insights layer turns thousands of predictions into business information: the sentiment split, a net sentiment score, the keywords that drive negative feedback, and a breakdown for each product.
> I also wrote a fine-tuning script that trains roberta-base on the TweetEval benchmark. The model was evaluated with accuracy, macro-F1 and a confusion matrix, and there is a Streamlit web app for the demo."

## 2. Demo script (about 5 minutes)

Open the terminal in the project folder **before** the interview. Run the commands once beforehand so the model is already downloaded.

1. **Show the folder structure** (VS Code sidebar).
   Say: "config is in one place, the pipeline is in `src/` and split into steps, the UI is separate."
2. **Run** `streamlit run app/streamlit_app.py`.
3. **Single Text tab.** Try these in order:
   - "Absolutely love this phone…" → Positive
   - "The package arrived on Tuesday." → Neutral (it is a fact with no opinion)
   - "Honestly not bad, I would buy it again." → **Positive (about 90%)**. Point out: *"A keyword model sees 'bad' and says Negative. RoBERTa understands negation from context."*
   - "I was not happy with the service." → **Negative**. The word "happy" is there, but the model sees the "not".
   - "Great, another update that breaks everything. Just perfect." → **Negative (about 61%)**, even though it contains "Great" and "perfect". The model catches the sarcasm, but the lower confidence shows it is unsure. Sarcasm is still a known weak spot.
   - Honest example of a failure: "Not bad at all, I expected worse." → Negative (50%). The words "bad" and "worse" outweigh the negation. Use it if they ask about limitations.
   - **Let the interviewer type their own sentence.** This makes a strong impression.
4. **Batch / CSV tab.** Run it on the sample data and show:
   - the distribution metrics
   - the net sentiment score
   - the per-product chart ("Smartwatch has the most negative feedback")
   - the top negative words ("battery", "broke"…). These are the actionable insights.
5. **Terminal.** Run `python main.py evaluate` and show the accuracy and the confusion matrix in `outputs/`.
6. **Code walkthrough.** Open `src/predictor.py` and explain the flow from tokenize to logits to softmax to argmax.

## 3. Likely questions and answers

**Q: What is RoBERTa and how is it different from BERT?**
RoBERTa is BERT with an improved training recipe (Facebook AI, 2019):
- It was trained longer, with bigger batches, on 10× more data (160 GB compared with 16 GB).
- It uses **dynamic masking**: a new mask pattern each epoch instead of one fixed mask.
- It **removed Next Sentence Prediction**, which turned out not to help.
- It uses a byte-level BPE tokenizer (50K vocabulary).

The architecture is the same as BERT-base: 12 layers, 768 hidden size, 12 attention heads, about 125M parameters.

**Q: Why a transformer and not TF-IDF + Logistic Regression?**
Self-attention lets each word "look at" every other word. That means the model handles context, negation ("not good") and word order. TF-IDF is a bag of words: it ignores order and context.

**Q: How does classification actually happen?**
1. The tokenizer adds `<s>` at the start.
2. RoBERTa produces a 768-dimensional vector for every token. The `<s>` vector represents the whole sentence.
3. A classification head (dense → tanh → dense) maps it to 3 logits.
4. Softmax turns the logits into probabilities, and argmax picks the label.

**Q: Why so little preprocessing?**
RoBERTa was pre-trained on raw text, so casing, punctuation and stop words like "not" carry meaning. I only normalise usernames and URLs, the same way the training data was normalised.

**Q: What is fine-tuning / transfer learning?**
- **Pre-training:** the model learns general language by predicting masked words in huge unlabelled text.
- **Fine-tuning:** we add a small classification layer and train the whole network briefly, with a small learning rate (2e-5), on labelled sentiment data.

You need far less labelled data than when training from scratch.

**Q: What loss and optimizer?**
Cross-entropy loss, with the AdamW optimizer, linear LR warm-up and decay, and weight decay of 0.01. These are the Hugging Face Trainer defaults.

**Q: Which metrics and why?**
Accuracy, plus **macro-F1**, which weights each class equally so the minority classes count. The confusion matrix shows *which* classes get confused. Usually it is Neutral, because neutral and mildly positive or negative text overlap.

**Q: Limitations?**
- sarcasm and irony
- mixed sentiment ("great camera but awful battery")
- domain shift (a tweet-trained model applied to medical text)
- the 512-token limit for long documents
- no multilingual support in this model

**Q: How would you improve it?**
- Fine-tune on domain-specific data.
- Use **aspect-based sentiment**: a separate sentiment for camera, battery and so on.
- Use a larger model (roberta-large) or a distilled one (DistilRoBERTa) for speed.
- Use XLM-RoBERTa for multiple languages.
- Handle class imbalance with class weights.
- Deploy as a FastAPI service with Docker.

**Q: How would you deploy it?**
- Wrap `SentimentPredictor` in a FastAPI `/predict` endpoint and containerise it with Docker.
- Load the model once at startup and batch the requests.
- Optionally export to ONNX for faster CPU inference.

## 4. Key numbers to remember

| Item | Value |
|---|---|
| Model | RoBERTa-base, about 125M parameters, 12 layers, hidden size 768 |
| Classes | 3 (Negative=0, Neutral=1, Positive=2) |
| Max sequence length used | 128 tokens (model limit 512) |
| Fine-tuning LR / epochs | 2e-5 / 2 |
| Training data (optional) | TweetEval sentiment (about 45k tweets; a 3k subset is used for laptop speed) |
