# Product Category Classifier

## Project Overview

This project uses Machine Learning and Natural Language Processing (NLP) to automatically classify products into one of 10 categories based on their titles.

The model analyzes product titles and predicts the most appropriate product category.

## Dataset

The dataset contains 35,311 product records and includes product titles, category labels, merchant information, ratings, views, and listing dates.

After data cleaning, 35,096 records were retained across 10 product categories.

## Data Preprocessing

The following preprocessing steps were performed:

- Removed records with missing product titles or category labels.
- Removed unnecessary whitespace.
- Standardized category names.
- Converted text to lowercase.
- Removed special characters.
- Prepared product titles for TF-IDF vectorization.

## Feature Engineering

TF-IDF (Term Frequency–Inverse Document Frequency) was used to convert product titles into numerical features.

Configuration:
- Unigrams and bigrams
- Maximum 20,000 features
- Minimum document frequency of 2
- Sublinear term frequency scaling

## Machine Learning Models

Three classification algorithms were compared:

1. Logistic Regression
2. Linear Support Vector Machine (Linear SVM)
3. Multinomial Naive Bayes

### Model Evaluation

| Model | Accuracy |
|---|---:|
| Logistic Regression | 96.10% |
| Linear SVM | 96.42% |
| Multinomial Naive Bayes | 96.17% |

Linear SVM achieved the highest accuracy and was selected as the final model.

## Final Model

The trained model is saved as:

`models/product_category_model.pkl`

The model includes the TF-IDF vectorizer and Linear SVM classifier in a single pipeline.

## Project Structure

```text
product-category-classifier/
│
├── data/
├── models/
│   └── product_category_model.pkl
│
├── notebooks/
│   └── product_category_classification.ipynb
│
├── predict_category.py
├── train_model.py
├── IMLP6_TASK_03-products.csv
└── README.md
```

## Installation

Install the required Python libraries:

```bash
pip install pandas scikit-learn joblib
```

## Training the Model

Run:

```bash
python train_model.py
```

The script
