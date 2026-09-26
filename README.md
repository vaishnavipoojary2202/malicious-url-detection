

# 🛡️ Malicious URL Detection using Machine Learning

A Machine Learning project for detecting whether a URL is **Legitimate** or **Phishing** using a combination of **handcrafted structural URL features** and **character-level TF-IDF features**.

The project compares multiple machine learning algorithms and selects **XGBoost** as the best-performing model based on the evaluation results. A **Streamlit web application** is provided for real-time URL prediction.

---

## 📌 Project Overview

Phishing websites are designed to deceive users into visiting malicious pages and potentially revealing sensitive information such as login credentials, banking details, and personal data.

Malicious URLs often contain identifiable patterns such as:

* Unusually long URLs
* Excessive numbers or special characters
* Multiple subdomains
* Suspicious keywords
* IP addresses instead of domain names
* URL shortening services
* Suspicious URL structures

This project combines these structural characteristics with **character-level TF-IDF representations** to build a machine learning-based malicious URL detection system.

### 🎯 Objective

The main objective is to develop a machine learning system that can:

1. Analyze URL characteristics.
2. Extract structural and lexical features.
3. Train multiple classification algorithms.
4. Compare their performance.
5. Select the best-performing model.
6. Predict whether a newly entered URL is **Legitimate** or **Phishing**.

### 🏆 Final Best Model

**XGBoost**

**Best Accuracy: 93.53%**

---

# ✨ Key Features

* Detects **Legitimate** and **Phishing** URLs
* Extracts **20 structural URL features**
* Uses **character-level TF-IDF**
* Combines structural and TF-IDF features
* Compares four machine learning algorithms
* Evaluates models using multiple performance metrics
* Automatically identifies the best-performing model
* Provides confusion matrix and accuracy comparison
* Provides real-time prediction through Streamlit
* Supports prediction of user-entered URLs

---

# 🔄 System Workflow

The complete workflow of the project is:

```text
                 ┌──────────────────────┐
                 │     URL Dataset      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Data Preprocessing   │
                 │ • Remove duplicates  │
                 │ • Encode labels      │
                 │ • Clean URL data     │
                 └──────────┬───────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │      Feature Extraction     │
              └─────────────┬───────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
   ┌─────────────────────┐     ┌─────────────────────┐
   │ Structural Features │     │ Character TF-IDF    │
   │     20 Features     │     │   3–5 char n-grams  │
   └──────────┬──────────┘     └──────────┬──────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
                 ┌──────────────────────┐
                 │ Feature Combination  │
                 │    50,020 Features   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Stratified 80:20     │
                 │ Train/Test Split     │
                 └──────────┬───────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │       Model Training         │
              │                              │
              │ • Logistic Regression       │
              │ • Decision Tree             │
              │ • Random Forest             │
              │ • XGBoost                   │
              └─────────────┬───────────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Model Evaluation     │
                 │ • Accuracy           │
                 │ • Precision          │
                 │ • Recall             │
                 │ • F1 Score           │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Best Model: XGBoost  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Streamlit Application│
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Enter New URL        │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Feature Extraction   │
                 │ + TF-IDF Transform   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Model Prediction     │
                 └──────────┬───────────┘
                            │
                   ┌────────┴────────┐
                   ▼                 ▼
             Legitimate           Phishing
```

---

# ⚙️ Methodology

The project follows the following machine learning pipeline:

### 1. Dataset Collection

A labeled URL dataset containing legitimate and phishing URLs is used for training and evaluation.

Each URL is associated with a target label:

```text
Legitimate → 0
Phishing   → 1
```

---

### 2. Data Preprocessing

The dataset is prepared before feature extraction.

The preprocessing steps include:

* Removing duplicate URLs
* Handling the target labels
* Encoding labels into numerical values
* Separating URL data from target labels
* Preparing the dataset for feature extraction

The final dataset contains:

**11,430 URLs**

---

### 3. Structural Feature Extraction

Twenty handcrafted structural features are extracted from every URL.

These features describe the **length, character composition, security properties, hostname structure, path structure, and suspicious patterns** of a URL.

### Structural Features Used

| No. | Feature                    | Description                                                        |
| --: | -------------------------- | ------------------------------------------------------------------ |
|   1 | `url_length`               | Total number of characters in the URL                              |
|   2 | `num_digits`               | Number of numeric digits in the URL                                |
|   3 | `num_letters`              | Number of alphabetic characters in the URL                         |
|   4 | `num_dots`                 | Number of `.` characters                                           |
|   5 | `num_hyphens`              | Number of `-` characters                                           |
|   6 | `num_at`                   | Number of `@` characters                                           |
|   7 | `num_question_marks`       | Number of `?` characters                                           |
|   8 | `num_equals`               | Number of `=` characters                                           |
|   9 | `num_ampersands`           | Number of `&` characters                                           |
|  10 | `num_slashes`              | Number of `/` characters                                           |
|  11 | `num_percent`              | Number of `%` characters                                           |
|  12 | `num_underscores`          | Number of `_` characters                                           |
|  13 | `num_colons`               | Number of `:` characters                                           |
|  14 | `has_https`                | Indicates whether HTTPS is present                                 |
|  15 | `has_ip`                   | Indicates whether an IP address is used instead of a normal domain |
|  16 | `hostname_length`          | Length of the URL hostname                                         |
|  17 | `path_length`              | Length of the URL path                                             |
|  18 | `num_subdomains`           | Number of detected subdomains                                      |
|  19 | `suspicious_keyword_count` | Number of suspicious keywords found in the URL                     |
|  20 | `has_shortening_service`   | Indicates whether a known URL shortening service is used           |

These features are represented as a **20-dimensional structural feature vector** for each URL.

---

### 4. Character-Level TF-IDF

In addition to handcrafted structural features, the raw URLs are converted into character-level TF-IDF representations.

Character n-grams of **3–5 characters** are used to capture patterns within URLs.

For example:

```text
https://secure-login-example.com
```

can be represented using character sequences such as:

```text
htt
ttp
tps
sec
ecu
cur
...
```

The TF-IDF representation captures lexical patterns that may help distinguish legitimate URLs from phishing URLs.

The TF-IDF vectorizer produces:

**50,000 features**

---

### 5. Feature Combination

The two feature sets are combined:

```text
20 Structural Features
          +
50,000 TF-IDF Features
          =
50,020 Total Features
```

The resulting feature matrix contains both:

* Handcrafted structural information
* Character-level lexical information

This combined representation is used for machine learning model training.

---

### 6. Train-Test Split

The dataset is divided using an **80:20 stratified train-test split**.

```text
Total Samples  = 11,430

Training       = 9,143
Testing        = 2,286
```

Stratified splitting maintains the class distribution between the training and testing datasets.

---

### 7. Feature Scaling

The numerical structural features are scaled before being combined with the TF-IDF representation.

This helps prevent structural features with larger numerical ranges from disproportionately affecting models that are sensitive to feature scale.

---

### 8. Model Training

Four machine learning algorithms are trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

The same prepared dataset and evaluation procedure are used to compare their performance.

---

# 🧠 Machine Learning Models

| Model               |   Accuracy |
| ------------------- | ---------: |
| **XGBoost**         | **93.53%** |
| Random Forest       |     92.96% |
| Logistic Regression |     91.91% |
| Decision Tree       |     87.66% |

Based on the experimental results, **XGBoost achieved the highest accuracy among the four implemented models**.

---

# 📊 XGBoost Performance

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **93.53%** |
| Precision | **93.22%** |
| Recall    | **93.88%** |
| F1 Score  | **93.55%** |

### Metric Definitions

**Accuracy:**
Percentage of total URLs classified correctly.

**Precision:**
Percentage of URLs predicted as phishing that were actually phishing.

**Recall:**
Percentage of actual phishing URLs correctly identified by the model.

**F1 Score:**
Harmonic mean of precision and recall.

---

# 📊 Dataset

| Property            |          Value |
| ------------------- | -------------: |
| Total Samples       |     **11,430** |
| Training Samples    |      **9,143** |
| Testing Samples     |      **2,286** |
| Train/Test Split    |      **80:20** |
| Split Type          | **Stratified** |
| Structural Features |         **20** |
| TF-IDF Features     |     **50,000** |
| Total Features      |     **50,020** |

### Target Encoding

```text
Legitimate = 0
Phishing   = 1
```

---

# 🔬 Feature Engineering Summary

The URL dataset was transformed into a machine learning-ready representation through the following process:

```text
Raw URL
   │
   ├──► 20 Structural Features
   │
   └──► Character-level TF-IDF
             │
             └──► 50,000 features
                      │
                      ▼
             Combined Feature Matrix
                      │
                      ▼
                 50,020 Features
```

The combination allows the model to learn from both explicit URL characteristics and character-level patterns.

---

# 📁 Project Structure

```text
malicious-url-detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── best_model.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Training.ipynb
│
├── results/
│   ├── metrics.csv
│   │
│   ├── graphs/
│   │   └── accuracy_comparison.png
│   │
│   └── confusion_matrix/
│       └── confusion_matrix.png
│
├── src/
│   ├── feature_extraction.py
│   └── predictor.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 🚀 Installation and Setup

## 1. Clone the Repository

```bash
git clone https://github.com/vaishnavipoojary2202/malicious-url-detection.git
cd malicious-url-detection
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# 💻 Streamlit Application Workflow

The Streamlit application performs prediction on a newly entered URL.

```text
User enters URL
       │
       ▼
URL preprocessing
       │
       ▼
Extract 20 structural features
       │
       ▼
Transform URL using trained TF-IDF vectorizer
       │
       ▼
Combine structural + TF-IDF features
       │
       ▼
Load trained XGBoost model
       │
       ▼
Generate prediction
       │
       ▼
┌─────────────────────┐
│ Legitimate / Phishing│
└─────────────────────┘
```

---

# 🧪 Example Predictions

### Legitimate URL

**Input:**

```text
https://www.google.com
```

**Prediction:**

```text
✅ Legitimate URL
```

### Phishing URL

**Input:**

```text
http://paypal-login-secure.xyz
```

**Prediction:**

```text
⚠️ Phishing URL
```

These examples demonstrate the intended behavior of the prediction interface; the model's output depends on the features extracted from the input URL.

---

# 📈 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

The project also generates visual comparisons of model performance.

### Accuracy Comparison

The accuracy comparison allows the performance of the four implemented algorithms to be analyzed side by side.

### Confusion Matrix

The confusion matrix provides the number of:

* True Positives
* True Negatives
* False Positives
* False Negatives

This provides more detailed information about classification errors than accuracy alone.

---

# 📚 Research Comparison

The implementation is additionally compared with the **PhishHunter-XLD** study.

The research comparison is provided as a **benchmark**, not as a direct reproduction, because the study and this project use different datasets, preprocessing procedures, feature engineering pipelines, and experimental setups.

| Implementation  | Model   |   Accuracy |
| --------------- | ------- | ---------: |
| PhishHunter-XLD | XGBoost | **96.17%** |
| Our Project     | XGBoost | **93.53%** |

Therefore, the reported values should be interpreted as results from separate experimental setups rather than as a controlled head-to-head experiment.

---

# 👥 Team Contributions

| Team Member           | Responsibility                                   |
| --------------------- | ------------------------------------------------ |
| **Smith Lopes**       | Exploratory Data Analysis (EDA)                  |
| **Vaishnavi Poojary** | Feature Engineering & Preprocessing              |
| **Reese Dabreo**      | Model Training, Evaluation & Research Comparison |
| **Person 4**          | Streamlit Deployment & Prediction System         |

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* XGBoost

### Data Processing

* Pandas
* NumPy
* SciPy

### Feature Engineering

* Character-level TF-IDF
* Custom structural URL feature extraction

### Visualization

* Matplotlib

### Model Persistence

* Joblib

### Deployment / Interface

* Streamlit

---

# 📌 Important Project Components

### Structural Feature Extraction

Implemented in:

```text
src/feature_extraction.py
```

The module extracts the 20 predefined structural characteristics from a URL.

### Prediction

Implemented in:

```text
src/predictor.py
```

The prediction pipeline processes a URL, generates the required features, and uses the trained model to generate the final classification.

### Streamlit Application

Implemented in:

```text
app.py
```

The application provides the user interface for entering URLs and displaying predictions.

---

# 🔮 Future Improvements

The current system can be extended with:

* Deep learning approaches such as LSTM or DistilBERT
* Explainable AI using SHAP
* Batch URL prediction
* URL reputation API integration
* Browser extension integration
* Real-time threat intelligence
* Cloud deployment using Streamlit Cloud
* Larger and more diverse URL datasets
* Additional lexical and domain-based features

---

# 📄 License

This project is developed for **academic and educational purposes** as part of a Machine Learning mini project.
