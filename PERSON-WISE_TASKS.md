# Person-Wise Implementation Tasks

## Malicious URL Detection Using Machine Learning

This document defines the technical responsibilities of all four team members.

The project is divided into four connected modules:

```text
Person 1
Dataset & Preprocessing
        ↓
Person 2
Feature Extraction
        ↓
Person 3
ML Models & Evaluation
        ↓
Person 4
Prediction System & Integration
```

---

# 👤 Person 1 — Dataset & Preprocessing

## Module

**Dataset Collection, Analysis and Preprocessing**

## Responsibilities

### 1. Dataset Selection

* Find suitable publicly available malicious URL datasets.
* Check whether the dataset contains both:

  * Legitimate URLs
  * Malicious URLs
* Record the dataset source and license/usage information.
* Check whether the selected research papers use the same or comparable datasets.

### 2. Dataset Analysis

Analyse:

* Total number of records
* Number of legitimate URLs
* Number of malicious URLs
* Available columns
* Missing values
* Duplicate URLs
* Class distribution

Create graphs for:

* Legitimate vs malicious URLs
* Dataset/class distribution

### 3. Data Cleaning

Perform:

* Duplicate removal
* Missing-value handling
* Invalid-record handling
* Label cleaning
* Data type correction

### 4. Label Encoding

Convert the target labels into a format suitable for Machine Learning.

Example:

```text
Legitimate → 0
Malicious  → 1
```

### 5. Train/Test Split

Create the dataset split required for model training and evaluation.

Ensure that the same evaluation methodology is followed when comparing models.

### 6. Documentation

Document:

* Dataset source
* Dataset size
* Classes
* Preprocessing techniques
* Train/test split
* Class distribution

## Deliverables

```text
data/raw/
data/processed/
notebooks/01_data_preprocessing.ipynb
```

Also provide:

* Clean dataset
* Dataset statistics
* Class distribution graph
* Preprocessing code
* Dataset documentation

---

# 👤 Person 2 — URL Feature Extraction

## Module

**URL Feature Engineering**

## Responsibilities

### 1. Study URL Characteristics

Identify characteristics that can help distinguish malicious URLs from legitimate URLs.

### 2. Implement Lexical Features

Extract features such as:

* URL length
* Number of dots
* Number of slashes
* Number of hyphens
* Number of underscores
* Number of digits
* Number of special characters
* Number of parameters
* Number of subdomains
* Hostname length
* Path length
* HTTPS presence
* IP address presence

### 3. Feature Extraction Function

Develop a reusable function that accepts a URL and returns its feature values.

Example:

```text
URL
 ↓
Feature Extraction
 ↓
Numerical Feature Vector
```

### 4. Apply Features to Dataset

Run the feature extraction process on all URLs.

Create the final feature dataset containing:

```text
Feature 1
Feature 2
Feature 3
...
Feature N
Target Label
```

### 5. Feature Analysis

Analyse:

* Feature distributions
* Correlations
* Potentially redundant features
* Important features

Create appropriate visualizations.

### 6. Research Paper Alignment

Check which features are used in the selected research papers.

Document:

```text
Feature used in paper
        ↓
Feature implemented by us
        ↓
Reason for inclusion
```

## Deliverables

```text
src/feature_extraction.py
notebooks/02_feature_extraction.ipynb
data/processed/feature_dataset.csv
```

Also provide:

* Feature list
* Feature descriptions
* Feature analysis graphs
* Feature extraction documentation

---

# 👤 Person 3 — ML Models & Evaluation

## Module

**Machine Learning Model Training, Evaluation and Research Comparison**

This is the main experimental module of the project.

## Responsibilities

### 1. Select ML Algorithms

Implement multiple classification algorithms.

Initial models:

```text
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. SVM
```

Additional algorithms may be added if required by the selected research papers.

### 2. Prepare Data

Load the feature dataset provided by Person 2.

Separate:

```text
X = Input Features
y = Target Label
```

Use the agreed train/test methodology.

### 3. Train Models

Train each selected ML algorithm using the same feature dataset and evaluation setup wherever possible.

### 4. Generate Predictions

Generate predictions for the test dataset.

### 5. Calculate Evaluation Metrics

For every model calculate:

* Accuracy
* Precision
* Recall
* F1-Score

Generate:

* Confusion Matrix
* Classification Report

### 6. Model Comparison

Create a final comparison table.

Example:

| Model               | Accuracy | Precision | Recall | F1 |
| ------------------- | -------: | --------: | -----: | -: |
| Logistic Regression |       XX |        XX |     XX | XX |
| Decision Tree       |       XX |        XX |     XX | XX |
| Random Forest       |       XX |        XX |     XX | XX |
| SVM                 |       XX |        XX |     XX | XX |

Create graphs comparing model performance.

### 7. Research Paper Comparison

Study the selected research papers and record:

* Dataset
* Features
* Algorithm
* Accuracy
* Precision
* Recall
* F1-score
* Experimental setup

Create:

```text
Our Results vs Research Paper Results
```

Important:

Results from different datasets or experimental setups should not be treated as directly equivalent. Differences should be clearly documented.

### 8. Final Model Selection

Based on the experimental results, identify the model that will be used for the prediction system.

The selection should be documented using the project's evaluation criteria rather than simply relying on accuracy.

### 9. Save Model

Save the selected trained model for Person 4.

Example:

```text
models/trained_model.pkl
```

## Deliverables

```text
src/train.py
src/evaluate.py
notebooks/03_model_training.ipynb
notebooks/04_model_evaluation.ipynb
results/metrics.csv
results/confusion_matrix/
results/graphs/
models/trained_model.pkl
```

Also provide:

* Model comparison table
* Confusion matrices
* Evaluation graphs
* Research paper comparison
* Final trained model

---

# 👤 Person 4 — Prediction System & Integration

## Module

**Model Integration and User Prediction System**

## Responsibilities

### 1. Receive Trained Model

Use the final model produced by Person 3.

```text
models/trained_model.pkl
```

### 2. Create Prediction Pipeline

Build the complete prediction flow:

```text
User URL
   ↓
URL Validation
   ↓
Feature Extraction
   ↓
Feature Vector
   ↓
Trained ML Model
   ↓
Prediction
   ↓
Legitimate / Malicious
```

### 3. Integrate Feature Extraction

Use the feature extraction implementation from Person 2.

The same feature extraction process used during training must be used during prediction.

### 4. Load Trained Model

Load the saved model using an appropriate library such as Joblib.

### 5. Create User Interface

Develop a simple interface where the user can enter a URL.

Example:

```text
--------------------------------
     Malicious URL Detector
--------------------------------

Enter URL:
[____________________________]

        [ Check URL ]

Result:
Malicious URL
--------------------------------
```

### 6. Prediction Testing

Test the system using URLs that were not used during model training.

Record:

| Test URL | Actual Class | Predicted Class |
| -------- | ------------ | --------------- |
| URL 1    | Legitimate   | Legitimate      |
| URL 2    | Malicious    | Malicious       |
| URL 3    | Legitimate   | Malicious       |

### 7. Integration Testing

Verify that:

* Feature extraction works correctly.
* Model loads correctly.
* Prediction works correctly.
* Invalid URLs are handled.
* The interface displays the result correctly.

### 8. Documentation

Document:

* System architecture
* Prediction workflow
* Interface
* Test cases
* Screenshots

## Deliverables

```text
src/predict.py
app/app.py
```

Also provide:

* Prediction pipeline
* User interface
* Test cases
* Screenshots
* Integration documentation

---

# 🔗 Integration Between Team Members

## Person 1 → Person 2

Person 1 provides:

```text
Clean Dataset
+
Labels
```

Person 2 uses it for feature extraction.

---

## Person 2 → Person 3

Person 2 provides:

```text
Feature Dataset
+
Feature Extraction Function
+
Feature List
```

Person 3 uses the feature dataset to train ML models.

---

## Person 3 → Person 4

Person 3 provides:

```text
Trained Model
+
Model Evaluation Results
+
Required Feature Order
```

Person 4 uses these for the prediction system.

---

# 📌 Common Team Responsibilities

Although each member has a separate module, everyone should contribute to:

* Understanding the selected research papers
* Understanding the complete project workflow
* GitHub commits
* Testing
* Final PPT
* Project report
* Viva preparation

Each member should be able to explain their own module as well as the overall system.

---

# 📁 Final GitHub Structure

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

# ✅ Completion Checklist

### Person 1

* [ ] Dataset finalized
* [ ] Dataset cleaned
* [ ] Duplicates handled
* [ ] Missing values handled
* [ ] Labels prepared
* [ ] Class distribution analysed
* [ ] Processed dataset committed

### Person 2

* [ ] Features finalized
* [ ] Feature extraction implemented
* [ ] Feature dataset generated
* [ ] Feature analysis completed
* [ ] Feature documentation completed

### Person 3

* [ ] ML algorithms implemented
* [ ] Models trained
* [ ] Predictions generated
* [ ] Accuracy calculated
* [ ] Precision calculated
* [ ] Recall calculated
* [ ] F1-score calculated
* [ ] Confusion matrices generated
* [ ] Model comparison completed
* [ ] Research paper comparison completed
* [ ] Final model saved

### Person 4

* [ ] Prediction pipeline implemented
* [ ] Trained model integrated
* [ ] Feature extraction integrated
* [ ] User interface implemented
* [ ] New URLs tested
* [ ] Integration testing completed
* [ ] Screenshots/documentation completed
