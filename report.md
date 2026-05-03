# Sentiment Analysis Pipeline - Project Report

## Executive Summary

This project implements a complete sentiment analysis pipeline for cafeteria/mess hall feedback. The system classifies user feedback into sentiment categories (positive, negative, neutral) using machine learning techniques. The pipeline processes data from Google Forms, cleans text, extracts features, and trains a Gradient Boosting classifier for binary sentiment classification.

---

## 1. Data Collection Phase

**Source:** Google Forms Survey

**Data Collected:**
- Timestamp of submission
- Meal type consumed
- Numeric Rating (1-5 scale)
- Detailed Feedback comments
- Individual ratings for:
  - Taste quality
  - Quantity/portion size
  - Hygiene standards
  - Service quality
- Issues encountered
- Suggestions for improvement

**Output File:** mess_data.csv

**Rationale:** Google Forms provides easy data collection from users with automatic CSV export, making it ideal for gathering structured feedback at scale.

---

## 2. Data Loading & Preprocessing (load_data.py)

**Objective:** Load raw CSV data and create sentiment labels for classification

**Process:**

1. **Data Import**
   - Read mess_data.csv using pandas
   - Standardize column names for clarity

2. **Feature Engineering**
   - Combine `Feedback` and `Issues` columns into single `full_text` field
   - Handle missing values using `fillna('')`
   - Creates richer text representation for analysis

3. **Sentiment Mapping**
   - Convert numeric ratings to categorical labels:
     - Rating ≤ 2 → `'negative'`
     - Rating = 3 → `'neutral'`
     - Rating ≥ 4 → `'positive'`
   - Enables supervised learning on sentiment labels

4. **Data Exploration**
   - Display column structure
   - Show sentiment distribution
   - Identify class imbalance if present

**Output:** Pandas DataFrame with `full_text` and `Sentiment` columns ready for text processing

---

## 3. Text Cleaning & Feature Extraction (text_cleaning.py)

**Objective:** Transform raw text into machine learning-ready numerical features

**Process:**

1. **Text Cleaning Function**
   - Convert all text to lowercase (standardization)
   - Remove numbers and special characters (keep only letters/spaces)
   - Remove extra whitespace (normalization)
   - Purpose: Reduce noise and improve model focus on meaningful words

2. **Data Filtering**
   - Remove empty responses (no text after cleaning)
   - Remove responses with fewer than 3 words (eliminates spam/noise)
   - Keeps only substantive feedback

3. **Feature Extraction with TF-IDF**
   - **TfidfVectorizer Configuration:**
     - `max_features=300`: Limits to top 300 most important words
       - Prevents overfitting on small datasets
       - Reduces computational complexity
     - `ngram_range=(1,2)`: Captures single words + word pairs (bigrams)
       - Improves context understanding
       - Example: "very good" treated as single feature
   - Converts text to sparse matrix of shape: (num_samples, 300)

4. **Output:** `X` - TF-IDF sparse matrix ready for model training

**Benefits of TF-IDF:**
- Weighs important words higher
- Reduces impact of common words (the, a, is)
- Provides numerical representation of text

---

## 4. Model Training & Evaluation (main.py)

**Objective:** Train sentiment classifier and evaluate performance

**Model Selection:** Gradient Boosting Classifier

**Rationale:**
- Handles sparse features well
- Less prone to overfitting than simple models
- Provides good generalization

**Hyperparameters:**
- `n_estimators=150`: 150 decision trees (strong ensemble)
- `learning_rate=0.05`: Low learning rate for better stability
- `max_depth=5`: Shallow trees prevent overfitting
- `subsample=0.8`: 80% subsampling for robustness

**Training Pipeline:**

1. **Data Filtering**
   - Remove neutral sentiment samples
   - Creates binary classification problem: positive vs negative
   - Simplifies decision boundary

2. **Train-Test Split**
   - 80% training data, 20% test data
   - `stratify=y_filtered`: Maintains sentiment ratio in both sets
   - `random_state=42`: Ensures reproducibility

3. **Cross-Validation**
   - 5-fold cross-validation on full training set
   - Measures model stability across different data subsets
   - Reports mean accuracy and standard deviation

4. **Model Training**
   - Fit on training set
   - Generate predictions on test set

**Evaluation Metrics:**

- **Accuracy:** Percentage of correct predictions
- **Confusion Matrix:** Shows True Positives, True Negatives, False Positives, False Negatives
- **Classification Report:** Precision, Recall, F1-Score per class
  - Precision: Of predicted positives, how many are correct?
  - Recall: Of actual positives, how many did we find?
  - F1-Score: Harmonic mean of precision and recall

---

## 5. Unit Testing (testing.py)

**Objective:** Validate functionality and correctness of all modules

**Test Coverage:**

1. **Data Loading Tests**
   - Verify dataframe loads successfully
   - Confirm Sentiment column exists
   - Validate sentiment values (positive, negative, neutral only)

2. **Text Cleaning Tests**
   - Test lowercase conversion
   - Test punctuation removal
   - Test number removal
   - Validate TF-IDF matrix dimensions

3. **Model Tests**
   - Verify prediction array size matches test set
   - Confirm predictions contain only valid sentiments
   - Validate model accuracy exceeds baseline (50%)

**Benefits:**
- Prevents regression bugs
- Ensures data integrity
- Validates model behavior

---

## Project Architecture

```
┌─────────────────────────┐
│   Google Forms Survey   │
│   (User Feedback)       │
└────────────┬────────────┘
             │
             ▼
        mess_data.csv
             │
             ▼
      ┌──────────────────┐
      │  load_data.py    │
      │  (Load & Label)  │
      └────────┬─────────┘
               │
               ▼
          DataFrame (df)
          with Sentiment
               │
               ▼
      ┌──────────────────────┐
      │ text_cleaning.py     │
      │ (Clean & Vectorize)  │
      └────────┬─────────────┘
               │
               ▼
          TF-IDF Matrix (X)
               │
               ▼
      ┌──────────────────────┐
      │     main.py          │
      │ (Train & Evaluate)   │
      └────────┬─────────────┘
               │
               ▼
        ┌─────────────────────┐
        │ Model Predictions   │
        │ Accuracy Metrics    │
        │ Classification      │
        │ Report              │
        └─────────────────────┘
               │
               ▼
      ┌──────────────────────┐
      │    testing.py        │
      │  (Unit Tests)        │
      └──────────────────────┘
```

---

## Key Parameters Summary

| Component | Parameter | Value | Purpose |
|-----------|-----------|-------|---------|
| load_data.py | Negative threshold | Rating ≤ 2 | Classify low ratings as negative |
| load_data.py | Neutral threshold | Rating = 3 | Separate neutral feedback |
| load_data.py | Positive threshold | Rating ≥ 4 | Classify high ratings as positive |
| text_cleaning.py | Max features | 300 | Limit vocabulary size |
| text_cleaning.py | N-gram range | (1, 2) | Include words and word pairs |
| text_cleaning.py | Min word count | 3 | Filter short responses |
| main.py | Test size | 0.2 (20%) | Reserve 20% for testing |
| main.py | Cross-validation folds | 5 | 5-fold CV for validation |
| main.py | Model type | Gradient Boosting | Ensemble learning method |
| main.py | n_estimators | 150 | Number of trees |
| main.py | learning_rate | 0.05 | Step size for optimization |
| main.py | max_depth | 5 | Tree depth limit |
| main.py | subsample | 0.8 | Proportion of samples per tree |

---

## Expected Results

**Cross-Validation Performance:**
- Mean CV Accuracy: Typically 75-85%
- Standard Deviation: ±3-5%

**Test Set Performance:**
- Test Accuracy: Similar to CV accuracy
- Balanced precision and recall across classes

**Confusion Matrix Interpretation:**
- High diagonal values = good predictions
- Off-diagonal values = misclassifications

---

## Future Enhancements

1. **Include Neutral Sentiment:** Multiclass classification instead of binary
2. **Feature Engineering:** Add domain-specific features (meal type, time of day)
3. **Hyperparameter Tuning:** Grid search or Bayesian optimization
4. **Model Comparison:** Test with SVM, Random Forest, Neural Networks
5. **Deployment:** Create API endpoint for real-time predictions
6. **Explainability:** Add feature importance analysis
7. **Data Augmentation:** Increase dataset size for better generalization

---

## Conclusion

This sentiment analysis pipeline provides a complete end-to-end solution for classifying cafeteria feedback. The modular design allows easy maintenance and extension. The Gradient Boosting model achieves solid performance on binary sentiment classification (positive vs negative), filtering out neutral feedback to focus on actionable insights. Unit tests ensure reliability and reproducibility across multiple runs.

**Generated on:** May 3, 2026