import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from model import (
    analyze_sentiment_textblob,
    analyze_sentiment_vader,
    analyze_sentiment_bert,
    get_model_comparison,
    get_consensus,
    analyze_multiple_texts,
)

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="metric-container"] {
        background: #f7f9fc;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #4a90d9;
    }
    </style>
""", unsafe_allow_html=True)

# colour map used across all charts
COLORS = {
    "POSITIVE": "#27ae60",
    "NEGATIVE": "#e74c3c",
    "NEUTRAL":  "#f39c12",
    "MIXED":    "#8e44ad",
}

def get_color(label):
    for key, color in COLORS.items():
        if key in label.upper():
            return color
    return "#7f8c8d"


# ── page header ──────────────────────────────────────────────────────────────
st.title("🧠 Sentiment Analysis")
st.markdown("Run **TextBlob**, **VADER**, and **DistilBERT** on any English text and compare results.")
st.divider()

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("How to use")
    st.markdown("""
    1. **Single Analysis** — paste one text, click Analyse
    2. **Batch Analysis** — one sentence per line, download CSV
    3. **About Models** — explains how each model works
    """)
    st.divider()
    st.header("Model quick info")
    for name, info in get_model_comparison().items():
        with st.expander(name):
            st.write(f"**Speed:** {info['speed']}")
            st.write(f"**Accuracy:** {info['accuracy']}")
            st.write(f"**Best for:** {info['best_for']}")
    st.divider()
    st.caption("DistilBERT downloads once (~300 MB) and is cached for the rest of the session.")


# ── tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Single Analysis", "📊 Batch Analysis", "📚 About Models"])


# ── tab 1 : single text ──────────────────────────────────────────────────────
with tab1:
    st.header("Analyse a Single Text")

    user_text = st.text_area(
        "Enter your text here:",
        placeholder="e.g. I really enjoyed using this product, it works exactly as described!",
        height=130,
    )

    col_btn, col_note = st.columns([1, 4])
    with col_btn:
        analyse_clicked = st.button("Analyse", use_container_width=True, type="primary")
    with col_note:
        st.caption("TextBlob and VADER run instantly. BERT takes ~5 s on first run, then it's cached.")

    if analyse_clicked:
        if not user_text.strip():
            st.error("Please enter some text first.")
        else:
            with st.spinner("Running all three models..."):
                tb  = analyze_sentiment_textblob(user_text)
                vd  = analyze_sentiment_vader(user_text)
                bt  = analyze_sentiment_bert(user_text)
                con = get_consensus([tb, vd, bt])

            # overall verdict banner
            label = con["consensus_sentiment"]
            pct   = con["agreement_pct"]
            emoji_map = {"POSITIVE": "😊", "NEGATIVE": "😞", "NEUTRAL": "😐"}

            if label == "MIXED":
                st.warning(f"⚖️ **Models disagree** — no clear majority. Agreement: {pct}%")
            else:
                st.success(f"**Verdict: {label} {emoji_map.get(label,'')}** — {pct}% of models agree")

            st.divider()

            # per-model metric cards
            st.subheader("Results by Model")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("TextBlob", tb["sentiment"], delta=f"Confidence: {tb['confidence']}%")
                st.caption(f"Polarity: `{tb['raw_score']}` | Subjectivity: `{tb['subjectivity']}`")

            with c2:
                st.metric("VADER", vd["sentiment"], delta=f"Confidence: {vd['confidence']}%")
                st.caption(f"Compound: `{vd['raw_score']}` | pos={vd['pos']}  neg={vd['neg']}  neu={vd['neu']}")

            with c3:
                st.metric("BERT", bt["sentiment"], delta=f"Confidence: {bt['confidence']}%")
                st.caption(f"Softmax score: `{bt['raw_score']}`")

            st.divider()

            # chart 1 — confidence comparison
            st.subheader("Confidence Scores")
            fig1 = go.Figure(data=[go.Bar(
                x=["TextBlob", "VADER", "BERT"],
                y=[tb["confidence"], vd["confidence"], bt["confidence"]],
                marker_color=[
                    get_color(tb["sentiment"]),
                    get_color(vd["sentiment"]),
                    get_color(bt["sentiment"]),
                ],
                text=[f"{tb['confidence']}%", f"{vd['confidence']}%", f"{bt['confidence']}%"],
                textposition="outside",
            )])
            fig1.update_layout(
                yaxis=dict(title="Confidence (%)", range=[0, 115]),
                xaxis_title="Model",
                height=370,
                margin=dict(t=20),
            )
            st.plotly_chart(fig1, use_container_width=True)

            # chart 2 — VADER breakdown (only VADER returns these 3 proportions)
            st.subheader("VADER Score Breakdown")
            st.caption("VADER returns separate positive, negative, and neutral proportions alongside the compound score.")
            fig2 = go.Figure(data=[
                go.Bar(name="Positive", x=["Score"], y=[vd["pos"]], marker_color="#27ae60"),
                go.Bar(name="Negative", x=["Score"], y=[vd["neg"]], marker_color="#e74c3c"),
                go.Bar(name="Neutral",  x=["Score"], y=[vd["neu"]], marker_color="#f39c12"),
            ])
            fig2.update_layout(barmode="stack", height=280, yaxis_title="Proportion", margin=dict(t=10))
            st.plotly_chart(fig2, use_container_width=True)

            # chart 3 — consensus pie
            st.subheader("Model Consensus")
            votes = con["votes"]
            fig3 = go.Figure(data=[go.Pie(
                labels=list(votes.keys()),
                values=list(votes.values()),
                marker=dict(colors=[get_color(k) for k in votes]),
                hole=0.35,
            )])
            fig3.update_layout(height=340, margin=dict(t=10))
            st.plotly_chart(fig3, use_container_width=True)

            # subjectivity note
            subj = tb["subjectivity"]
            if subj > 0.6:
                st.info(f"📝 Subjectivity is high ({subj}) — this text is mostly opinion-based, which can affect how reliable the sentiment score is.")
            elif subj < 0.3:
                st.info(f"📝 Subjectivity is low ({subj}) — this text reads as mostly factual, which makes sentiment harder to detect accurately.")


# ── tab 2 : batch analysis ───────────────────────────────────────────────────
with tab2:
    st.header("Batch Analysis")
    st.write("Paste multiple texts below — one per line. All three models will run on each one.")

    batch_input = st.text_area(
        "Your texts (one per line):",
        placeholder="I love this product!\nThis is terrible service.\nIt's okay I guess.",
        height=200,
    )

    if st.button("Analyse All", use_container_width=True, type="primary"):
        if not batch_input.strip():
            st.error("Please enter at least one line of text.")
        else:
            texts = [t.strip() for t in batch_input.splitlines() if t.strip()]
            with st.spinner(f"Analysing {len(texts)} texts with all 3 models..."):
                df = analyze_multiple_texts(texts)

            st.subheader("Results Table")
            st.dataframe(df, use_container_width=True)

            st.subheader("Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total texts", len(df))
            m2.metric("Avg BERT confidence", f"{df['BERT_Conf%'].mean():.1f}%")
            m3.metric("Most common verdict", df["Consensus"].mode()[0] if len(df) > 0 else "—")
            full_agree = int(((df["TextBlob"] == df["VADER"]) & (df["VADER"] == df["BERT"])).sum())
            m4.metric("All 3 models agreed", full_agree)

            if len(df) > 0:
                counts = df["Consensus"].value_counts().reset_index()
                counts.columns = ["Sentiment", "Count"]
                fig4 = px.pie(
                    counts, names="Sentiment", values="Count",
                    color="Sentiment",
                    color_discrete_map=COLORS,
                    title="Consensus Distribution Across All Texts",
                )
                fig4.update_layout(height=370)
                st.plotly_chart(fig4, use_container_width=True)

            st.download_button(
                "⬇️ Download Results as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="sentiment_results.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ── tab 3 : about models ─────────────────────────────────────────────────────
with tab3:
    st.header("About the Models")

    st.subheader("TextBlob")
    st.markdown("""
    The simplest of the three. TextBlob uses a dictionary of words, each with a 
    pre-assigned polarity score from –1 (very negative) to +1 (very positive). 
    It averages these scores across the sentence to get a final polarity value.

    It also returns a **subjectivity score** (0 = factual, 1 = opinionated). 
    This app uses that as an extra signal — if the text is highly subjective, 
    the sentiment score is more meaningful; if it's factual, the model may struggle.

    Works well for general English text. Doesn't handle sarcasm or context well.
    """)

    st.subheader("VADER")
    st.markdown("""
    VADER (Hutto & Gilbert, 2014) was designed specifically for social media. 
    Unlike TextBlob it understands emphasis signals — ALL-CAPS words score higher, 
    exclamation marks add intensity, and it has a built-in emoji lexicon.

    Returns four scores: `pos`, `neg`, `neu`, and `compound`. The compound score 
    is the main one — it's a normalised value from –1 to +1.

    One thing worth noting: the original version of this project used 
    `max(pos, neg, neu)` as the confidence score. I changed it to `abs(compound)` 
    because the neutral proportion often comes out highest even for clearly 
    positive or negative text, which gives a misleading confidence value.

    Works best on: tweets, product reviews, comments, anything informal.
    """)

    st.subheader("BERT — DistilBERT")
    st.markdown("""
    DistilBERT (Sanh et al., 2019) is a smaller, faster version of BERT. 
    It's 40% smaller and 60% faster while keeping ~97% of BERT's accuracy. 
    This version is fine-tuned on SST-2 (Stanford Sentiment Treebank — about 
    67,000 labeled movie reviews from Rotten Tomatoes).

    The key difference from the other two: it actually understands word order 
    and context. So "not bad" → POSITIVE, "could have been better" → NEGATIVE. 
    Rule-based models struggle with both of these.

    One limitation: SST-2 is a binary dataset (positive/negative only), 
    so DistilBERT has no neutral class. Neutral text will be pushed into 
    whichever direction it leans, even slightly.

    The model is cached after the first load so it doesn't reload on every click.
    """)

    st.divider()
    st.subheader("Side by Side Comparison")

    cmp = pd.DataFrame({
        "": ["Speed", "Accuracy", "Understands negation", "Has neutral class", "Emoji support", "Extra output"],
        "TextBlob":    ["Instant",       "~75%", "❌",          "✅", "❌", "Subjectivity score"],
        "VADER":       ["Instant",       "~80%", "Partially",   "✅", "✅", "pos / neg / neu breakdown"],
        "DistilBERT":  ["Fast (cached)", "~87%", "✅",          "❌", "❌", "Softmax confidence"],
    })
    st.table(cmp)

    st.subheader("When Models Disagree")
    st.markdown("""
    The app shows a **MIXED** verdict when no single sentiment gets a majority vote. 
    This usually happens for one of these reasons:

    - **Sarcasm** — "Oh great, another delay" reads as positive to TextBlob/VADER
    - **Mixed opinions** — "the food was great but the service was terrible"
    - **Very short text** — one or two words don't give enough signal
    - **Domain mismatch** — VADER was trained on social media; formal text can confuse it

    In these cases BERT's prediction is usually the most reliable.
    """)


# ── footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · TextBlob · VADER (NLTK) · DistilBERT (HuggingFace Transformers) · Plotly")
