import streamlit as st

_MODELS = [
    {
        "name": "Logistic Regression",
        "badge": "Classical ML",
        "badge_color": "#1a6e8e",
        "key": "lr",
        "description": (
            "TF-IDF bag-of-words (50 000 features, unigrams + bigrams) paired with a "
            "regularised logistic regression. Fast, fully interpretable — every prediction "
            "can be traced back to individual tokens via coefficient weights."
        ),
        "training_data": "30 K balanced samples",
        "accuracy": "0.93",
        "precision": "0.93",
        "recall": "0.92",
        "f1": "0.93",
        "speed": "< 5 ms",
        "explainability": "Full (TF-IDF coefficients)",
    },
    {
        "name": "XGBoost",
        "badge": "Classical ML",
        "badge_color": "#1a6e8e",
        "key": "xgboost",
        "description": (
            "Gradient-boosted trees on top of a TF-IDF matrix (10 000 features). "
            "Captures non-linear interactions between tokens that a linear model misses, "
            "with tuned hyperparameters from grid search."
        ),
        "training_data": "30 K balanced samples",
        "accuracy": "0.89",
        "precision": "0.84",
        "recall": "0.97",
        "f1": "0.90",
        "speed": "< 10 ms",
        "explainability": "Partial (feature importance)",
    },
    {
        "name": "DistilBERT",
        "badge": "Transformer",
        "badge_color": "#0e7a5c",
        "key": "distilbert",
        "description": (
            "DistilBERT-base-uncased fine-tuned for binary sequence classification. "
            "Understands word order and context — 'not happy' ≠ 'happy'. "
            "Trained with weighted cross-entropy to handle class imbalance."
        ),
        "training_data": "30 K balanced samples",
        "accuracy": "0.96",
        "precision": "0.94",
        "recall": "0.97",
        "f1": "0.96",
        "speed": "200–400 ms",
        "explainability": "Gradient × input attribution",
    },
    {
        "name": "MentalRoBERTa",
        "badge": "Domain Transformer",
        "badge_color": "#6b3fa0",
        "key": "mental_roberta",
        "description": (
            "RoBERTa pretrained on mental-health corpora, then fine-tuned on our dataset. "
            "Domain pretraining gives it a vocabulary and representations tuned to clinical "
            "and self-disclosure language — the best precision on the depressed class."
        ),
        "training_data": "21 K samples",
        "accuracy": "0.96",
        "precision": "0.96",
        "recall": "0.97",
        "f1": "0.96",
        "speed": "200–400 ms",
        "explainability": "Gradient × input attribution",
    },
]

_DATASET_ROWS = [
    ("Reddit Depression (Kaggle)", "Positive (distress)", "~480 K posts", "Core positive class"),
    ("Positive subreddits (scraped)", "Negative (no distress)", "~15 K posts", "r/happy, r/aww, r/wholesomememes…"),
    ("Balanced subset", "Both", "30 K (1:1)", "Used to train all models"),
]


def _model_card(model: dict) -> None:
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(145deg, #072341 0%, #04152b 100%);
            border: 1px solid rgba(66,216,240,0.25);
            border-radius: 0.5rem;
            padding: 1.2rem 1.1rem 1rem;
            height: 100%;
        ">
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.8rem;">
                <span style="
                    background:{model["badge_color"]};
                    color:#eaf5ff;
                    font-size:0.68rem;
                    font-weight:800;
                    letter-spacing:0.05em;
                    text-transform:uppercase;
                    padding:0.2rem 0.55rem;
                    border-radius:0.2rem;
                ">{model["badge"]}</span>
            </div>
            <h3 style="margin:0 0 0.6rem; color:#ffffff; font-size:1.15rem;">{model["name"]}</h3>
            <p style="color:#bad7eb; font-size:0.88rem; line-height:1.55; margin-bottom:1rem;">
                {model["description"]}
            </p>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem 1rem; font-size:0.82rem;">
                <div><span style="color:#42d8f0;">Accuracy</span><br/><b style="color:#fff;">{model["accuracy"]}</b></div>
                <div><span style="color:#42d8f0;">F1 (distress)</span><br/><b style="color:#fff;">{model["f1"]}</b></div>
                <div><span style="color:#42d8f0;">Precision</span><br/><b style="color:#fff;">{model["precision"]}</b></div>
                <div><span style="color:#42d8f0;">Recall</span><br/><b style="color:#fff;">{model["recall"]}</b></div>
                <div><span style="color:#42d8f0;">Latency</span><br/><b style="color:#fff;">{model["speed"]}</b></div>
                <div><span style="color:#42d8f0;">Training data</span><br/><b style="color:#fff;">{model["training_data"]}</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_about_page() -> None:
    """Render the About the Models page."""
    st.markdown(
        """
        <div class="hero-banner">PROJET FINAL • BOOTCAMP DATA SCIENCE • ARTEFACT SCHOOL OF DATA</div>
        <h1 class="hero-title">About the<br/>Models</h1>
        <p class="hero-subtitle">
            Four models, one task: detect early distress signals in text.
            Each represents a different trade-off between speed, accuracy, and interpretability.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Model cards ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">The models</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        _model_card(_MODELS[0])
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        _model_card(_MODELS[2])
    with col_b:
        _model_card(_MODELS[1])
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        _model_card(_MODELS[3])

    # ── Performance comparison ────────────────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Performance comparison — depressed class</p>', unsafe_allow_html=True)

    import pandas as pd

    df = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "XGBoost", "DistilBERT", "MentalRoBERTa"],
            "Accuracy": [0.93, 0.89, 0.96, 0.96],
            "Precision": [0.93, 0.84, 0.94, 0.96],
            "Recall": [0.92, 0.97, 0.97, 0.97],
            "F1": [0.93, 0.90, 0.96, 0.96],
            "Training samples": ["30 K", "30 K", "30 K", "21 K"],
            "Latency": ["< 5 ms", "< 10 ms", "200–400 ms", "200–400 ms"],
        }
    ).set_index("Model")

    st.dataframe(df, use_container_width=True)

    # ── Training data ─────────────────────────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Training data</p>', unsafe_allow_html=True)

    df_data = pd.DataFrame(
        _DATASET_ROWS,
        columns=["Dataset", "Class", "Size", "Notes"],
    )
    st.dataframe(df_data, use_container_width=True, hide_index=True)
