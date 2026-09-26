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
            background: radial-gradient(circle at 50% 0%, #173a6b 0%, #0b1b34 42%, #061224 100%);
            color: #e5efff;
        }

        .block-container {
            max-width: 850px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        /* Header */
        .hero {
            background: linear-gradient(135deg, #07172e 0%, #123a6b 55%, #1d5ca8 100%);
            padding: 2.2rem 2rem;
            border-radius: 24px;
            text-align: center;
            box-shadow: 0 14px 40px rgba(0, 0, 0, 0.28);
            margin-bottom: 1.5rem;
        }

        .hero-icon {
            font-size: 3rem;
            margin-bottom: 0.3rem;
        }

        .hero-title {
            color: white;
            font-size: 2.15rem;
            font-weight: 750;
            margin: 0;
        }

        .hero-subtitle {
            color: #b9d8ff;
            font-size: 1rem;
            margin-top: 0.55rem;
        }

        /* Section headings */
        .section-title {
            color: #b9d8ff;
            font-size: 1.15rem;
            font-weight: 700;
            margin-top: 1rem;
            margin-bottom: 0.55rem;
        }

        /* Input */
        div[data-testid="stTextInput"] label {
            color: #c7dcf7;
            font-weight: 650;
        }

        div[data-testid="stTextInput"] input {
            border: 1.5px solid #315986;
            border-radius: 14px;
            background: #0d203b;
            color: #eef6ff;
            padding: 0.85rem 1rem;
        }

        div[data-testid="stTextInput"] input:focus {
            border-color: #60a5fa;
            box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.14);
        }

        /* Analyze button */
        div.stButton > button {
            width: 100%;
            border-radius: 14px;
            border: none;
            background: linear-gradient(90deg, #1d4ed8, #2563eb);
            color: white;
            font-weight: 700;
            font-size: 1rem;
            padding: 0.75rem 1rem;
            min-height: 3rem;
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.20);
            transition: all 0.2s ease;
        }

        div.stButton > button:hover {
            background: linear-gradient(90deg, #1e40af, #1d4ed8);
            border: none;
            color: white;
            transform: translateY(-1px);
        }

        /* Result cards */
        .result-card {
            border-radius: 20px;
            padding: 1.35rem 1.5rem;
            margin-top: 1.25rem;
            margin-bottom: 1rem;
            background: #102746;
            box-shadow: 0 14px 35px rgba(0, 0, 0, 0.24);
            border: 1px solid #24466e;
        }

        .result-safe {
            border-left: 6px solid #16a34a;
        }

        .result-danger {
            border-left: 6px solid #dc2626;
        }

        .result-label {
            font-size: 0.8rem;
            font-weight: 750;
            letter-spacing: 0.08em;
            color: #8fb6e8;
            text-transform: uppercase;
        }

        .result-title {
            font-size: 1.55rem;
            font-weight: 800;
            margin-top: 0.25rem;
            color: #edf5ff;
        }

        .probability {
            font-size: 2rem;
            font-weight: 800;
            color: #7db7ff;
            margin-top: 0.25rem;
        }

        /* URL card */
        .url-card {
            background: #091a30;
            border: 1px solid #24466e;
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin-top: 0.8rem;
            color: #c9dcf4;
            word-break: break-all;
            font-family: monospace;
        }

        /* Footer */
        .footer {
            text-align: center;
            color: #7896b8;
            font-size: 0.82rem;
            margin-top: 2.2rem;
            padding-top: 1rem;
            border-top: 1px solid #1e3b5d;
        }

        /* Streamlit alerts */
        div[data-testid="stAlert"] {
            border-radius: 14px;
            background: #122946;
            color: #dbeafe;
            border: 1px solid #24466e;
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
        <div class="hero-icon">🛡️</div>
        <div class="hero-title">Malicious URL Detection</div>
        <div class="hero-subtitle">
            Machine-learning based analysis for phishing URL detection
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Input
# --------------------------------------------------

st.markdown(
    '<div class="section-title">🔗 Enter a URL to analyze</div>',
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

if st.button("🔍  Analyze URL", use_container_width=True):

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
