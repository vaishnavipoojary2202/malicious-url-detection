import joblib
import pandas as pd
import streamlit as st

from pathlib import Path
from scipy.sparse import csr_matrix, hstack

from src.feature_extraction import extract_url_features


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
TFIDF_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"
SCALER_PATH = BASE_DIR / "models" / "structural_scaler.pkl"


# --------------------------------------------------
# Structural feature order
# Must match the training pipeline
# --------------------------------------------------

STRUCTURAL_FEATURE_NAMES = [
    "url_length",
    "num_digits",
    "num_letters",
    "num_dots",
    "num_hyphens",
    "num_at",
    "num_question_marks",
    "num_equals",
    "num_ampersands",
    "num_slashes",
    "num_percent",
    "num_underscores",
    "num_colons",
    "has_https",
    "has_ip",
    "hostname_length",
    "path_length",
    "num_subdomains",
    "suspicious_keyword_count",
    "has_shortening_service"
]


# --------------------------------------------------
# Load trained components
# --------------------------------------------------

@st.cache_resource
def load_components():
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, tfidf, scaler


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Malicious URL Detection",
    page_icon="🛡️",
    layout="centered"
)


# --------------------------------------------------
# Custom UI styling
# --------------------------------------------------

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600&display=swap');

        :root {
            --navy: #0B1830;
            --navy-2: #132A49;
            --sapphire: #244B73;
            --gold: #C9A86A;
            --gold-soft: #E4D3AE;
            --ivory: #F7F3EA;
            --cream: #EFE8DA;
            --taupe: #B7AD9C;
            --ink: #172238;
        }

        .stApp {
            background: var(--ivory);
            color: var(--ink);
        }

        .block-container {
            max-width: 900px;
            padding-top: 2.5rem;
            padding-bottom: 3.5rem;
        }

        /* Luxury hero */
        .hero {
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(circle at 88% 15%, rgba(201,168,106,.20), transparent 24%),
                radial-gradient(circle at 12% 90%, rgba(36,75,115,.42), transparent 30%),
                linear-gradient(135deg, #081427 0%, #0B1830 52%, #19395D 100%);
            padding: 4.2rem 2.5rem 3.6rem;
            text-align: center;
            border: 1px solid rgba(201,168,106,.55);
            box-shadow: 0 24px 60px rgba(11,24,48,.20);
            margin-bottom: 2.6rem;
        }

        .hero::before {
            content: "";
            position: absolute;
            inset: 13px;
            border: 1px solid rgba(228,211,174,.18);
            pointer-events: none;
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 180px;
            height: 180px;
            border: 1px solid rgba(201,168,106,.22);
            border-radius: 50%;
            right: -55px;
            top: -70px;
        }

        .hero-icon {
            position: relative;
            color: var(--gold-soft);
            font-size: 1.15rem;
            letter-spacing: .45em;
            margin-left: .45em;
            margin-bottom: 1.2rem;
        }

        .hero-title {
            position: relative;
            color: #FAF7F0;
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 3rem;
            line-height: 1.1;
            font-weight: 500;
            letter-spacing: .01em;
            margin: 0;
        }

        .hero-subtitle {
            position: relative;
            color: #D9E0E9;
            font-family: 'DM Sans', sans-serif;
            font-size: .9rem;
            letter-spacing: .16em;
            text-transform: uppercase;
            margin-top: 1rem;
        }

        .gold-line {
            position: relative;
            width: 72px;
            height: 1px;
            background: var(--gold);
            margin: 1.35rem auto 0;
        }

        /* Typography */
        .section-title {
            color: var(--navy);
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 1.65rem;
            font-weight: 500;
            margin-top: 1.8rem;
            margin-bottom: .65rem;
        }

        .eyebrow {
            color: #8D7650;
            font-family: 'DM Sans', sans-serif;
            font-size: .72rem;
            font-weight: 700;
            letter-spacing: .18em;
            text-transform: uppercase;
            margin-bottom: .2rem;
        }

        /* Input */
        div[data-testid="stTextInput"] input {
            border: 1px solid #CFC6B5;
            border-radius: 3px;
            background: #FFFDF8;
            color: var(--ink);
            font-family: 'DM Sans', sans-serif;
            font-size: 1rem;
            padding: 1rem 1.05rem;
            box-shadow: 0 8px 24px rgba(11,24,48,.05);
        }

        div[data-testid="stTextInput"] input:focus {
            border-color: var(--gold);
            box-shadow: 0 0 0 2px rgba(201,168,106,.16), 0 8px 24px rgba(11,24,48,.06);
        }

        /* CTA */
        div.stButton > button {
            border-radius: 3px;
            border: 1px solid #B6924E;
            background: linear-gradient(180deg, #D7BB82 0%, #C9A86A 100%);
            color: #0B1830;
            font-family: 'DM Sans', sans-serif;
            font-size: .82rem;
            font-weight: 700;
            letter-spacing: .12em;
            min-height: 3.1rem;
            box-shadow: 0 10px 24px rgba(11,24,48,.12);
            transition: all .2s ease;
        }

        div.stButton > button:hover {
            background: linear-gradient(180deg, #E1C993 0%, #D0AE6C 100%);
            color: #0B1830;
            border-color: #A98240;
            transform: translateY(-1px);
        }

        /* Result */
        .result-card {
            background: linear-gradient(135deg, #0B1830 0%, #163454 100%);
            border: 1px solid rgba(201,168,106,.65);
            border-radius: 3px;
            padding: 2rem 2.1rem;
            margin-top: 2rem;
            box-shadow: 0 22px 45px rgba(11,24,48,.18);
        }

        .result-card.safe {
            border-top: 4px solid var(--gold);
        }

        .result-card.danger {
            border-top: 4px solid #B1555A;
        }

        .result-label {
            color: var(--gold-soft);
            font-family: 'DM Sans', sans-serif;
            font-size: .68rem;
            font-weight: 700;
            letter-spacing: .18em;
            text-transform: uppercase;
        }

        .result-title {
            color: #FAF7F0;
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 2rem;
            font-weight: 500;
            margin-top: .35rem;
        }

        .probability {
            color: #F0D9A4;
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 2.7rem;
            margin-top: .55rem;
        }

        .probability-caption {
            color: #C7D0DB;
            font-family: 'DM Sans', sans-serif;
            font-size: .8rem;
            letter-spacing: .04em;
        }

        /* URL display */
        .url-card {
            background: #FFFDF8;
            border: 1px solid #D3C9B8;
            border-left: 3px solid var(--gold);
            border-radius: 3px;
            padding: 1rem 1.1rem;
            margin-top: .7rem;
            color: #26364D;
            word-break: break-all;
            font-family: 'DM Sans', sans-serif;
            font-size: .9rem;
        }

        .helper {
            color: #7A7A73;
            font-family: 'DM Sans', sans-serif;
            font-size: .78rem;
            margin-top: .55rem;
        }

        .footer {
            text-align: center;
            color: #8D8A82;
            font-family: 'DM Sans', sans-serif;
            font-size: .7rem;
            letter-spacing: .12em;
            text-transform: uppercase;
            margin-top: 3rem;
            padding-top: 1.3rem;
            border-top: 1px solid #D7CFBF;
        }

        div[data-testid="stAlert"] {
            border-radius: 3px;
            font-family: 'DM Sans', sans-serif;
        }

        .stSpinner > div {
            color: var(--navy);
        }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">◆ &nbsp; ◆ &nbsp; ◆</div>
        <div class="hero-title">Malicious URL Detection</div>
        <div class="hero-subtitle">Intelligent URL Security Analysis</div>
        <div class="gold-line"></div>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Input
# --------------------------------------------------

st.markdown(
    '<div class="eyebrow">Security Scanner</div><div class="section-title">Analyze a URL</div>',
    unsafe_allow_html=True
)

url = st.text_input(
    "URL",
    placeholder="https://example.com/login",
    label_visibility="collapsed"
)

st.markdown(
    '<div class="helper">Enter a URL below. Our trained machine-learning model analyzes its structural and character patterns.</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("ANALYZE  URL", use_container_width=True):

    if not url.strip():

        st.warning("Please enter a URL before analyzing.")

    else:

        try:

            with st.spinner("Analyzing URL..."):

                # Keep URL as a string
                url = url.strip()

                # ------------------------------------------
                # 1. Extract 20 structural URL features
                # ------------------------------------------

                structural_features = extract_url_features(url)

                structural_df = pd.DataFrame(
                    [structural_features]
                )

                # Ensure exact feature order
                structural_df = structural_df[
                    STRUCTURAL_FEATURE_NAMES
                ]

                # ------------------------------------------
                # 2. Load trained components
                # ------------------------------------------

                model, tfidf, scaler = load_components()

                # ------------------------------------------
                # 3. Character TF-IDF
                # ------------------------------------------

                url_tfidf = tfidf.transform([url])

                # ------------------------------------------
                # 4. Scale structural features
                # ------------------------------------------

                structural_scaled = scaler.transform(
                    structural_df
                )

                structural_scaled = csr_matrix(
                    structural_scaled
                )

                # ------------------------------------------
                # 5. Combine features
                # ------------------------------------------

                final_features = hstack(
                    [url_tfidf, structural_scaled]
                ).tocsr()

                # ------------------------------------------
                # 6. Prediction
                # ------------------------------------------

                prediction = model.predict(final_features)[0]

                probabilities = model.predict_proba(
                    final_features
                )[0]

                # ------------------------------------------
                # 7. Display result
                # ------------------------------------------

                if prediction == 1:

                    confidence = probabilities[1] * 100

                    st.markdown(
                        f"""
                        <div class="result-card danger">
                            <div class="result-label">Security Analysis</div>
                            <div class="result-title">Phishing URL Detected</div>
                            <div class="probability">{confidence:.2f}%</div>
                            <div class="probability-caption">Phishing Probability</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        """<div style="background:#742D36;color:#F3EEF0;border:1px solid #DB8384;border-radius:2px;padding:0.8rem 1rem;">This URL has been classified as phishing by the machine learning model.</div>""",
                        unsafe_allow_html=True
                    )

                else:

                    confidence = probabilities[0] * 100

                    st.markdown(
                        f"""
                        <div class="result-card safe">
                            <div class="result-label">Security Analysis</div>
                            <div class="result-title">Legitimate URL</div>
                            <div class="probability">{confidence:.2f}%</div>
                            <div class="probability-caption">Legitimate Probability</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        """<div style="background:#143559;color:#F3EEF0;border:1px solid #5B6F90;border-radius:2px;padding:0.8rem 1rem;">This URL has been classified as legitimate by the machine learning model.</div>""",
                        unsafe_allow_html=True
                    )

                # ------------------------------------------
                # 8. Show analyzed URL
                # ------------------------------------------

                st.markdown(
                    '<div class="eyebrow" style="margin-top:2rem;">Analysis Detail</div><div class="section-title" style="margin-top:.2rem;">Analyzed URL</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="url-card">{url}</div>',
                    unsafe_allow_html=True
                )

        except Exception as e:

            st.error(
                "An error occurred while analyzing the URL."
            )

            st.exception(e)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        🛡️ Malicious URL Detection &nbsp;•&nbsp;
        Machine Learning Based URL Classification
    </div>
    """,
    unsafe_allow_html=True
)
