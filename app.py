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
        /* Main page */
        .stApp {
            background: linear-gradient(135deg, #000F22 0%, #1B3554 38%, #3F6593 68%, #80AAD3 100%);
            color: #C0E6FD;
        }

        .block-container {
            max-width: 850px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        /* Header */
        .hero {
            background: linear-gradient(135deg, #000F22 0%, #1B3554 48%, #3F6593 100%);
            padding: 3.2rem 2rem;
            border-radius: 2px;
            text-align: center;
            box-shadow: 0 16px 35px rgba(17, 34, 80, 0.16);
            border: 1px solid rgba(192, 230, 253, 0.28);
            margin-bottom: 2rem;
        }

        .hero-icon {
            font-size: 3rem;
            margin-bottom: 0.3rem;
        }

        .hero-title {
            color: #C0E6FD;
            font-size: 2.45rem;
            font-weight: 500;
            letter-spacing: 0.02em;
            margin: 0;
        }

        .hero-subtitle {
            color: #C0E6FD;
            font-size: 0.98rem;
            letter-spacing: 0.04em;
            margin-top: 0.7rem;
        }

        div[data-testid="stCaptionContainer"] {
            color: #80AAD3;
        }

        .stSpinner > div {
            color: #C0E6FD;
        }

        /* Section headings */
        .section-title {
            color: #C0E6FD;
            font-size: 1.05rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
            margin-top: 1rem;
            margin-bottom: 0.55rem;
        }

        /* Input */
        div[data-testid="stTextInput"] label {
            color: #C0E6FD;
            font-weight: 650;
        }

        div[data-testid="stTextInput"] input {
            border: 1px solid #80AAD3;
            border-radius: 2px;
            background: #1B3554;
            color: #C0E6FD;
            padding: 1rem 1rem;
        }

        div[data-testid="stTextInput"] input:focus {
            border-color: #C0E6FD;
            box-shadow: 0 0 0 3px rgba(192, 230, 253, 0.12);
        }

        /* Analyze button */
        div.stButton > button {
            width: 100%;
            border-radius: 2px;
            border: 1px solid #80AAD3;
            background: #5B86B6;
            color: white;
            font-weight: 700;
            font-size: 1rem;
            padding: 0.75rem 1rem;
            min-height: 3rem;
            box-shadow: 0 8px 20px rgba(0, 15, 34, 0.28);
            transition: all 0.2s ease;
        }

        div.stButton > button:hover {
            background: #BEB7A7;
            border: 1px solid #80AAD3;
            color: #112250;
            transform: translateY(-1px);
        }

        /* Result cards */
        .result-card {
            border-radius: 2px;
            padding: 1.6rem 1.5rem;
            margin-top: 1.25rem;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #1B3554 0%, #3F6593 100%);
            box-shadow: 0 14px 30px rgba(0, 15, 34, 0.28);
            border: 1px solid #80AAD3;
        }

        .result-safe {
            border-left: 5px solid #E7E2CE;
        }

        .result-danger {
            border-left: 5px solid #BEB7A7;
        }

        .result-label {
            font-size: 0.8rem;
            font-weight: 750;
            letter-spacing: 0.08em;
            color: #C0E6FD;
            text-transform: uppercase;
        }

        .result-title {
            font-size: 1.55rem;
            font-weight: 800;
            margin-top: 0.25rem;
            color: #C0E6FD;
        }

        .probability {
            font-size: 2rem;
            font-weight: 800;
            color: #C0E6FD;
            margin-top: 0.35rem;
        }

        /* URL card */
        .url-card {
            background: #1B3554;
            border: 1px solid #80AAD3;
            border-radius: 2px;
            padding: 0.9rem 1rem;
            margin-top: 0.8rem;
            color: #112250;
            word-break: break-all;
            font-family: monospace;
        }

        /* Decorative luxury divider */
        .luxury-divider {
            width: 70px;
            height: 2px;
            background: #C0E6FD;
            margin: 1rem auto 0;
        }

        /* Footer */
        .footer {
            text-align: center;
            color: #80AAD3;
            font-size: 0.82rem;
            margin-top: 2.2rem;
            padding-top: 1rem;
            border-top: 1px solid #3F6593;
        }

        /* Streamlit alerts */
        div[data-testid="stAlert"] {
            border-radius: 2px;
            background: #5B86B6;
            color: white;
            border: 1px solid #80AAD3;
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
        <div class="hero-icon">◈</div>
        <div class="hero-title">Malicious URL Detection</div>
        <div class="hero-subtitle">
            Machine-learning based analysis for phishing URL detection
        </div>
    <div class="luxury-divider"></div>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Input
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Enter a URL to analyze</div>',
    unsafe_allow_html=True
)

url = st.text_input(
    "URL",
    placeholder="https://example.com/login",
    label_visibility="collapsed"
)

st.caption(
    "Analyze the URL structure and character patterns using our trained ML model."
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("ANALYZE URL", use_container_width=True):

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
                        <div class="result-card result-danger">
                            <div class="result-label">Security Analysis</div>
                            <div class="result-title">🚨 Phishing URL Detected</div>
                            <div class="probability">{confidence:.2f}%</div>
                            <div>Phishing probability</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.warning(
                        "This URL has been classified as phishing "
                        "by the machine learning model."
                    )

                else:

                    confidence = probabilities[0] * 100

                    st.markdown(
                        f"""
                        <div class="result-card result-safe">
                            <div class="result-label">Security Analysis</div>
                            <div class="result-title">✅ Legitimate URL</div>
                            <div class="probability">{confidence:.2f}%</div>
                            <div>Legitimate probability</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.info(
                        "This URL has been classified as legitimate "
                        "by the machine learning model."
                    )

                # ------------------------------------------
                # 8. Show analyzed URL
                # ------------------------------------------

                st.markdown(
                    '<div class="section-title">Analyzed URL</div>',
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
