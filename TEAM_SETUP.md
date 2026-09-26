# 👥 Team Setup Guide

This guide explains how every team member can clone, install, run, test, and contribute to the Malicious URL Detection project.

## 1. Clone the repository

```bash
git clone https://github.com/vaishnavipoojary2202/malicious-url-detection.git
cd malicious-url-detection
```

Check the current branch:

```bash
git branch
```

The integrated project is on `main`.

## 2. Create a virtual environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

After activation, you should see `(.venv)` in the terminal.

## 3. Install dependencies

Run from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file installs:

- Streamlit
- Pandas
- SciPy
- Scikit-learn
- XGBoost
- Joblib

## 4. macOS only: XGBoost OpenMP

If XGBoost gives an error containing `libomp.dylib`, install the macOS OpenMP runtime:

```bash
brew install libomp
```

Then test:

```bash
python -c "import xgboost; print(xgboost.__version__)"
```

If the version prints without an error, XGBoost is ready.

## 5. Run the Streamlit app

From the repository root:

```bash
streamlit run app.py
```

Open the Local URL shown by Streamlit, normally:

```text
http://localhost:8501
```

The app requires these files:

```text
app.py
src/feature_extraction.py
models/best_model.pkl
models/tfidf_vectorizer.pkl
models/structural_scaler.pkl
requirements.txt
```

The original training dataset and training matrices are not required to run the prediction app.

## 6. How prediction works

The app follows the same preprocessing expected by the trained model:

```text
User URL
   ↓
20 structural URL features
   +
Character TF-IDF features
   ↓
Combined 50,020-feature vector
   ↓
Saved XGBoost model
   ↓
Legitimate / Phishing
```

The structural feature order must not be changed.

Do not fit a new TF-IDF vectorizer or scaler for a prediction. The app must use the saved artifacts in `models/`.

## 7. Test the app

Example legitimate URL:

```text
https://www.google.com
```

Another legitimate example:

```text
https://www.amazon.in/
```

For a controlled suspicious-looking demonstration:

```text
http://paypal-login-verify-account.com/login
```

A model prediction is not a guarantee that a real-world URL is safe or unsafe.

## 8. Git workflow for team members

Do not develop directly on `main`.

Before starting new work:

```bash
git checkout main
git pull origin main
```

Create your own branch:

```bash
git checkout -b your-feature-name
```

Example:

```bash
git checkout -b improve-streamlit-ui
```

After making changes:

```bash
git status
git add .
git commit -m "Describe the change"
git push origin your-feature-name
```

Then open a Pull Request on GitHub.

## 9. Before pushing

Check:

```bash
git status
```

Do not commit:

- `.venv/`
- Python cache files
- local secrets
- unnecessary large datasets
- temporary files

The repository's `.gitignore` should handle common local files.

## 10. If someone gets the latest changes

Use:

```bash
git checkout main
git pull origin main
```

If you already have a feature branch and need the latest main changes:

```bash
git checkout main
git pull origin main
git checkout your-feature-name
git merge main
```

Resolve any conflicts, test the project, then commit and push if needed.

## 11. Streamlit Cloud deployment

For deployment, use the GitHub repository and the `main` branch.

Deployment settings:

```text
Repository: vaishnavipoojary2202/malicious-url-detection
Branch: main
Main file: app.py
```

Streamlit Cloud installs packages from `requirements.txt`.

The saved model files under `models/` are used by the app, so they must remain in the repository.

## 12. Common errors

### `pip: command not found`

Use:

```bash
python3 -m pip
```

or, after activating the virtual environment:

```bash
python -m pip
```

### PEP 668 / externally managed environment

Do not install packages globally. Create the virtual environment first:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then:

```bash
python -m pip install -r requirements.txt
```

### XGBoost `libomp.dylib` error on Mac

Run:

```bash
brew install libomp
```

Then:

```bash
python -c "import xgboost; print(xgboost.__version__)"
```

### Streamlit cannot find model files

Make sure you are in the repository root:

```bash
cd malicious-url-detection
streamlit run app.py
```

Also verify:

```text
models/best_model.pkl
models/tfidf_vectorizer.pkl
models/structural_scaler.pkl
```

## 👥 Current responsibilities

| Member | Work |
|---|---|
| Avi / Vaishnavi | Feature Engineering + Final Integration + Streamlit |
| Smith | Dataset + EDA |
| Reese | Model Training + Evaluation |
| All members | Testing, documentation and final integration |

