import re
import nltk
import pandas as pd
from functools import lru_cache
from textblob import TextBlob
from transformers import pipeline

# download required NLTK data on first run
for resource, pkg in [
    ("tokenizers/punkt", "punkt"),
    ("sentiment/vader_lexicon", "vader_lexicon"),
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.sentiment import SentimentIntensityAnalyzer


def preprocess_text(text):
    # clean the text before passing to any model
    # removes URLs, HTML tags, and extra spaces
    text = text.strip()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# BERT takes ~3 seconds to load — cache it so it only loads once per session
@lru_cache(maxsize=1)
def _load_bert():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )

@lru_cache(maxsize=1)
def _load_vader():
    return SentimentIntensityAnalyzer()


def analyze_sentiment_textblob(text):
    text = preprocess_text(text)
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.1:
        sentiment = "POSITIVE 😊"
        confidence = round(min(polarity * 100, 99.0), 2)
    elif polarity < -0.1:
        sentiment = "NEGATIVE 😞"
        confidence = round(min(abs(polarity) * 100, 99.0), 2)
    else:
        sentiment = "NEUTRAL 😐"
        confidence = round((0.1 - abs(polarity)) / 0.1 * 60, 2)

    return {
        "model": "TextBlob",
        "sentiment": sentiment,
        "confidence": confidence,
        "raw_score": round(polarity, 4),
        "subjectivity": round(subjectivity, 4),
    }


def analyze_sentiment_vader(text):
    text = preprocess_text(text)
    sia = _load_vader()
    scores = sia.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "POSITIVE 😊"
    elif compound <= -0.05:
        sentiment = "NEGATIVE 😞"
    else:
        sentiment = "NEUTRAL 😐"

    # note: using abs(compound) instead of max(pos, neg, neu)
    # max(pos,neg,neu) often returns the neu score (e.g. 0.6) even for
    # clearly positive text, which gives misleading confidence values
    if sentiment == "NEUTRAL 😐":
        confidence = round((0.05 - abs(compound)) / 0.05 * 60, 2)
    else:
        confidence = round(min(abs(compound) * 100, 99.0), 2)

    return {
        "model": "VADER",
        "sentiment": sentiment,
        "confidence": confidence,
        "raw_score": round(compound, 4),
        "pos": scores["pos"],
        "neg": scores["neg"],
        "neu": scores["neu"],
    }


def analyze_sentiment_bert(text):
    text = preprocess_text(text)
    classifier = _load_bert()
    result = classifier(text[:512])[0]  # distilbert has a 512 token limit

    sentiment = "POSITIVE 😊" if result["label"] == "POSITIVE" else "NEGATIVE 😞"

    return {
        "model": "BERT (DistilBERT)",
        "sentiment": sentiment,
        "confidence": round(result["score"] * 100, 2),
        "raw_score": round(result["score"], 4),
    }


def get_consensus(results):
    # simple majority vote across the three model results
    votes = {}
    for r in results:
        label = r["sentiment"].split()[0]  # get just POSITIVE/NEGATIVE/NEUTRAL
        votes[label] = votes.get(label, 0) + 1

    top_label = max(votes, key=votes.get)
    top_count = votes[top_label]

    return {
        "consensus_sentiment": top_label if top_count > len(results) / 2 else "MIXED",
        "agreement_pct": round(top_count / len(results) * 100),
        "votes": votes,
    }


def analyze_multiple_texts(texts_list):
    rows = []
    for text in texts_list:
        if not text.strip():
            continue
        try:
            tb = analyze_sentiment_textblob(text)
            vd = analyze_sentiment_vader(text)
            bt = analyze_sentiment_bert(text)
            cons = get_consensus([tb, vd, bt])

            rows.append({
                "Text": text[:60] + "…" if len(text) > 60 else text,
                "TextBlob": tb["sentiment"],
                "VADER": vd["sentiment"],
                "BERT": bt["sentiment"],
                "Consensus": cons["consensus_sentiment"],
                "TB_Conf%": tb["confidence"],
                "VADER_Conf%": vd["confidence"],
                "BERT_Conf%": bt["confidence"],
            })
        except Exception as e:
            print(f"skipped: {text[:40]} | error: {e}")

    return pd.DataFrame(rows)


def get_model_comparison():
    return {
        "TextBlob": {
            "speed": "Very Fast ⚡",
            "accuracy": "~75%",
            "best_for": "General text, quick checks",
        },
        "VADER": {
            "speed": "Very Fast ⚡",
            "accuracy": "~80%",
            "best_for": "Social media, tweets, emojis",
        },
        "BERT": {
            "speed": "Moderate 🚀",
            "accuracy": "~87%",
            "best_for": "Complex sentences, formal text",
        },
    }


if __name__ == "__main__":
    test_inputs = [
        "I absolutely love this, works perfectly!",
        "Worst experience I've ever had.",
        "It's okay I guess, nothing special.",
        "Visit https://example.com for details.",
    ]

    for t in test_inputs:
        print(f"\nInput: {t}")
        print("  TextBlob:", analyze_sentiment_textblob(t))
        print("  VADER:   ", analyze_sentiment_vader(t))
        print("  BERT:    ", analyze_sentiment_bert(t))
