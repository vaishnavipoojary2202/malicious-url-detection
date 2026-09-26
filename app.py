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
# UI
# --------------------------------------------------

st.title("🛡️ Malicious URL Detection")

st.write(
    "Enter a URL below to classify it as "
    "**Legitimate** or **Phishing** using our machine learning model."
)

url = st.text_input(
    "Enter URL",
    placeholder="https://example.com/login"
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Analyze URL", use_container_width=True):

    if not url.strip():
        st.warning("Please enter a URL.")

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
                    [
                        url_tfidf,
                        structural_scaled
                    ]
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

                    st.error(
                        "🚨 PHISHING URL DETECTED"
                    )

                    st.metric(
                        "Phishing Probability",
                        f"{confidence:.2f}%"
                    )

                    st.warning(
                        "This URL has been classified as "
                        "phishing by the machine learning model."
                    )

                else:

                    confidence = probabilities[0] * 100

                    st.success(
                        "✅ LEGITIMATE URL"
                    )

                    st.metric(
                        "Legitimate Probability",
                        f"{confidence:.2f}%"
                    )

                    st.info(
                        "This URL has been classified as "
                        "legitimate by the machine learning model."
                    )

                # ------------------------------------------
                # 8. Show analyzed URL
                # ------------------------------------------

                st.divider()

                st.subheader("Analyzed URL")

                st.code(url)

        except Exception as e:

            st.error(
                "An error occurred while analyzing the URL."
            )

            st.exception(e)
