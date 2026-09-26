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
        .stApp {
            background: #F7F8FA;
            color: #172238;
        }

        .block-container {
            max-width: 760px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        .hero {
            text-align: center;
            padding: 1.5rem 0 2.2rem;
            border-bottom: 1px solid #D9E0E8;
            margin-bottom: 2.5rem;
        }

        .hero-icon {
            color: #43597C;
            font-size: 1.25rem;
            margin-bottom: .45rem;
        }

        .hero-title {
            color: #131B36;
            font-family: Georgia, serif;
            font-size: 2.25rem;
            font-weight: 500;
            margin: 0;
        }

        .hero-subtitle {
            color: #6B7788;
            font-size: .9rem;
            margin-top: .5rem;
        }

        .section-title {
            color: #131B36;
            font-size: 1.05rem;
            font-weight: 600;
            margin: 0 0 .7rem;
        }

        .helper {
            color: #788392;
            font-size: .8rem;
            margin-top: .5rem;
            margin-bottom: 1.2rem;
        }

        div[data-testid="stTextInput"] input {
            background: white;
            color: #131B36;
            border: 1px solid #C8D0DB;
            border-radius: 6px;
            padding: .9rem 1rem;
        }

        div[data-testid="stTextInput"] input:focus {
            border-color: #43597C;
            box-shadow: 0 0 0 2px rgba(67, 89, 124, .10);
        }

        div.stButton > button {
            width: 100%;
            background: #143559;
            color: white;
            border: 1px solid #143559;
            border-radius: 6px;
            min-height: 2.9rem;
            font-weight: 600;
            letter-spacing: .03em;
        }

        div.stButton > button:hover {
            background: #1B3554;
            color: white;
            border-color: #1B3554;
        }

        .result-card {
            margin-top: 2rem;
            padding: 1.4rem 1.5rem;
            background: white;
            border: 1px solid #D9E0E8;
            border-radius: 6px;
        }

        .result-card.safe {
            border-left: 4px solid #5B86B6;
        }

        .result-card.danger {
            border-left: 4px solid #B1555A;
        }

        .result-label {
            color: #788392;
            font-size: .7rem;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
        }

        .result-title {
            color: #131B36;
            font-size: 1.45rem;
            font-weight: 600;
            margin-top: .25rem;
        }

        .probability {
            color: #43597C;
            font-size: 2.2rem;
            font-weight: 700;
            margin-top: .5rem;
        }

        .probability-caption {
            color: #788392;
            font-size: .8rem;
        }

        .url-card {
            background: white;
            border: 1px solid #D9E0E8;
            border-radius: 6px;
            padding: .9rem 1rem;
            color: #43597C;
            word-break: break-all;
            font-family: monospace;
            font-size: .85rem;
        }

        .footer {
            text-align: center;
            color: #98A1AD;
            font-size: .72rem;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #D9E0E8;
        }

        div[data-testid="stAlert"] {
            border-radius: 6px;
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
        <div class="hero-icon">🛡</div>
        <div class="hero-title">Malicious URL Detection</div>
        <div class="hero-subtitle">Machine-learning based URL classification</div>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Input
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Analyze a URL</div>',
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

if st.button("Analyze URL", use_container_width=True):

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
                    '<div class="section-title" style="margin-top:2rem;">Analyzed URL</div>',
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
