# Malicious URL Detection Using Machine Learning

## 📌 Project Overview

Malicious URLs are commonly used in phishing, malware distribution, scams, and other cyberattacks. Detecting whether a URL is legitimate or malicious before accessing it can help improve web security.

This project aims to develop a **Machine Learning-based Malicious URL Detection System** that analyzes the characteristics of a URL and classifies it as either:

* **Legitimate URL**
* **Malicious URL**

The project implements multiple Machine Learning algorithms and compares their performance using standard evaluation metrics. The obtained results are also compared with results reported in selected research papers.

---

## 🎯 Objectives

* Detect malicious and legitimate URLs using Machine Learning.
* Perform preprocessing and cleaning of the URL dataset.
* Extract meaningful lexical features from URLs.
* Train multiple Machine Learning classification models.
* Compare the performance of different models.
* Evaluate models using Accuracy, Precision, Recall and F1-Score.
* Analyze the Confusion Matrix of the trained models.
* Compare our experimental results with results reported in existing research papers.
* Develop a simple interface for testing new URLs.

---

## 🔄 System Workflow

```text
                URL Dataset
                     │
                     ▼
          Data Preprocessing
                     │
                     ▼
          URL Feature Extraction
                     │
                     ▼
              Feature Dataset
                     │
                     ▼
          Train/Test Data Split
                     │
                     ▼
             ML Model Training
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
     Logistic     Decision   Random Forest
    Regression      Tree
          │          │          │
          └──────────┼──────────┘
                     ▼
              Model Evaluation
                     │
                     ▼
          Performance Comparison
                     │
                     ▼
            Best Performing Model
                     │
                     ▼
             URL Prediction System
                     │
                     ▼
        Legitimate / Malicious
```

---

## 📊 Dataset

The project uses a publicly available dataset containing URLs labelled as legitimate or malicious.

The dataset is processed before model training by:

* Removing duplicate records
* Handling missing values
* Checking incorrect or invalid records
* Encoding class labels
* Analysing class distribution
* Splitting the dataset into training and testing sets

### Dataset Classes

| Class      | Description                                           |
| ---------- | ----------------------------------------------------- |
| Legitimate | Safe/benign URLs                                      |
| Malicious  | URLs associated with malicious or suspicious activity |

> The final dataset source, number of records and class distribution will be documented after the dataset is finalized.

---

## 🔍 URL Feature Extraction

Machine Learning models require numerical input. Therefore, different characteristics of URLs are extracted as features.

### Example Features

| Feature                      | Description                                                 |
| ---------------------------- | ----------------------------------------------------------- |
| URL Length                   | Total number of characters in the URL                       |
| Number of Dots               | Count of `.` characters                                     |
| Number of Slashes            | Count of `/` characters                                     |
| Number of Digits             | Number of numerical characters                              |
| Number of Hyphens            | Count of `-` characters                                     |
| Number of Special Characters | Count of special characters                                 |
| Number of Parameters         | Number of URL parameters                                    |
| Number of Subdomains         | Number of detected subdomains                               |
| HTTPS                        | Indicates whether HTTPS is used                             |
| IP Address                   | Indicates whether an IP address is used instead of a domain |
| Hostname Length              | Length of the hostname                                      |
| Path Length                  | Length of the URL path                                      |

The final feature set will be determined based on the selected dataset and research papers.

---

## 🤖 Machine Learning Models

Multiple classification algorithms will be implemented and evaluated.

The proposed models include:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)

Additional models such as XGBoost may be included if required by the experimental setup.

The purpose of using multiple algorithms is to determine how different Machine Learning approaches perform on the same dataset and feature set.

---

## 📈 Model Evaluation

The models will be evaluated using:

### Accuracy

Measures the percentage of correctly classified URLs.

### Precision

Measures how many URLs predicted as malicious are actually malicious.

### Recall

Measures how many actual malicious URLs are correctly detected.

### F1-Score

Provides a combined measure of Precision and Recall.

### Confusion Matrix

The confusion matrix will be used to analyse:

* True Positive
* True Negative
* False Positive
* False Negative

### Performance Comparison

The final results will be represented using tables and graphs.

Example:

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |      XX% |       XX% |    XX% |      XX% |
| Decision Tree       |      XX% |       XX% |    XX% |      XX% |
| Random Forest       |      XX% |       XX% |    XX% |      XX% |
| SVM                 |      XX% |       XX% |    XX% |      XX% |

> The values will be replaced with results obtained from the actual experiments.

---

## 📚 Research Paper Comparison

The project will also compare the experimental results with selected research papers related to malicious URL detection.

The comparison will consider:

* Dataset used
* Feature extraction technique
* Machine Learning algorithm
* Evaluation metrics
* Reported performance

Example:

| Study           | Dataset      | Model         | Accuracy | F1-Score |
| --------------- | ------------ | ------------- | -------: | -------: |
| Paper 1         | Dataset Name | Random Forest |      XX% |      XX% |
| Paper 2         | Dataset Name | SVM           |      XX% |      XX% |
| Proposed System | Our Dataset  | Random Forest |      XX% |      XX% |

> Results from different studies will be interpreted carefully because differences in datasets, features, data splits and experimental settings can affect performance.

---

## 🖥️ Prediction System

After training and evaluating the models, the selected trained model will be integrated into a simple prediction system.

### Working

```text
User enters URL
       ↓
URL Feature Extraction
       ↓
Feature Vector
       ↓
Trained ML Model
       ↓
Prediction
       ↓
Legitimate / Malicious
```

Example:

```text
Input:
https://example.com/login

Output:
Legitimate URL
```

or

```text
Input:
http://example-example-login.xyz/verify

Output:
Malicious URL
```

The prediction interface will be implemented using a lightweight framework such as Streamlit, depending on the final project requirements.

---

## 🛠️ Technology Stack

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib

### Development Environment

* Jupyter Notebook / Google Colab
* Visual Studio Code

### Optional Interface

* Streamlit

---

## 📁 Project Structure

```text
Malicious-URL-Detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── trained_model.pkl
│
├── results/
│   ├── metrics.csv
│   ├── confusion_matrix/
│   └── graphs/
│
├── app/
│   └── app.py
│
├── docs/
│   ├── research_papers.md
│   └── paper_comparison.md
│
├── requirements.txt
├── PERSON_WISE_TASKS.md
└── README.md
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone <repository-url>
```

Navigate to the project directory:

```bash
cd Malicious-URL-Detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Train the Models

```bash
python src/train.py
```

### Evaluate the Models

```bash
python src/evaluate.py
```

### Run the Prediction Interface

If Streamlit is used:

```bash
streamlit run app/app.py
```

---

## 📌 Expected Outcome

The final system will:

1. Accept a URL as input.
2. Extract relevant URL-based features.
3. Pass the features to the trained Machine Learning model.
4. Classify the URL as legitimate or malicious.
5. Display the prediction.
6. Provide experimental results comparing multiple Machine Learning models.
7. Compare the obtained results with selected research papers.

---

## 🔮 Future Scope

* Integration of additional URL and webpage-based features.
* Real-time URL checking.
* Integration with browser extensions.
* Use of deep learning and transformer-based models.
* Continuous learning using newly detected malicious URLs.
* Integration with threat intelligence sources.
* Deployment as a cloud-based security service.

---

## 👥 Project Team

| Member   | Responsibility                  |
| -------- | ------------------------------- |
| Member 1 | Dataset & Preprocessing         |
| Member 2 | Feature Extraction              |
| Member 3 | ML Models & Evaluation          |
| Member 4 | Prediction System & Integration |

---

## 📖 References

Research papers, datasets and other technical resources used during the development of the project will be documented here.

GitHub repositories may be used as **implementation references**, but the project's preprocessing, feature extraction, experiments, model evaluation and results will be independently implemented and documented.
