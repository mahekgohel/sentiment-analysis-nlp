# Sentiment Analysis Web App

A web app that compares three NLP sentiment models — TextBlob, VADER, and DistilBERT — on any English text. Built with Streamlit and deployed on Streamlit Cloud.

**Live demo:** *(add your Streamlit Cloud link here after deploying)*

---

## What it does

Paste any text and all three models run on it simultaneously. You get:

- Sentiment label (Positive / Negative / Neutral) from each model
- Confidence score per model
- A majority-vote consensus verdict (or "MIXED" if models disagree)
- VADER's detailed pos/neg/neu breakdown as a chart
- TextBlob's subjectivity score as an extra signal
- Batch mode — paste multiple texts, see all results in a table, download as CSV

---

## Why three models?

Each model has different strengths:

**TextBlob** is the simplest — it averages word-level polarity scores from a dictionary. Fast, but it has no context awareness. It also returns a subjectivity score which the other two don't, so I kept it for that reason.

**VADER** was built specifically for social media. It handles emphasis signals like ALL-CAPS and `!!!`, and has an emoji lexicon. One thing I changed from the standard usage: the original code used `max(pos, neg, neu)` as the confidence value, but this often returns the neutral proportion (which can be 0.6+) even for clearly positive text. I switched to `abs(compound)` which is a much better proxy for how certain the model is.

**DistilBERT** is a transformer model fine-tuned on SST-2 (Stanford Sentiment Treebank). It actually understands word context and order, so it handles negation correctly — "not bad" → positive. It's 40% smaller and 60% faster than full BERT while keeping ~97% of the accuracy. The main limitation is that SST-2 is binary (positive/negative only), so there's no neutral class.

---

## Project structure

```
sentiment-analysis-nlp/
├── app.py            # Streamlit UI — tabs, charts, CSV download
├── model.py          # All model logic — preprocessing, inference, consensus vote
├── requirements.txt  # Dependencies
├── config.toml       # Streamlit theme settings
└── README.md
```

`model.py` handles everything related to the models. `app.py` just calls functions from `model.py` and handles the display. Kept them separate so the model logic is easy to test independently (run `python model.py` directly to see test outputs).

---

## Run locally

```bash
git clone https://github.com/mahekgohel/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp

pip install -r requirements.txt

streamlit run app.py
```

First run will download the DistilBERT model from HuggingFace (~300 MB). After that it's cached locally and loads fast. TextBlob and VADER are instant.

---

## Deploy to Streamlit Cloud

1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub, select this repo
4. Set main file path to `app.py`
5. Click Deploy — you get a live URL in about 2 minutes

---

## Known limitations

- DistilBERT has no neutral class (SST-2 is binary), so neutral text gets pushed into positive or negative
- 512 token limit on DistilBERT — longer texts get truncated
- All three models are English only
- None of them handle sarcasm reliably (DistilBERT does best)

---

## Tech stack

- Python 3.10+
- Streamlit — web interface
- TextBlob — rule-based sentiment
- NLTK / VADER — social media sentiment
- HuggingFace Transformers — DistilBERT inference
- Plotly — charts
- Pandas — batch results table

---

**Mahek Gohel**
B.Tech CSE (3rd year)
[LinkedIn](https://www.linkedin.com/in/mahek-gohel-b0120b39a) · [GitHub](https://github.com/mahekgohel)
