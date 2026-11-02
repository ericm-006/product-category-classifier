# Product Category Classifier

## Overview
A Machine Learning project that automatically classifies products into 10 categories based on their titles using Natural Language Processing (NLP).

## Dataset
- Original dataset: 35,311 records.
- Cleaned dataset: 35,096 records.
- Number of categories: 10.

## Data Preprocessing
- Removed records with missing product titles or category labels.
- Standardized category names.
- Converted text to lowercase.
- Removed special characters and extra whitespace.

## Feature Engineering
TF-IDF vectorization was used with:
- Unigrams and bigrams
- Maximum 20,000 features
- Minimum document frequency of 2
- Sublinear term frequency scaling

## Models Compared
1. Logistic Regression
2. Linear Support Vector Machine (Linear SVM)
3. Multinomial Naive Bayes

## Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 96.10% |
| Linear SVM | 96.42% |
| Multinomial Naive Bayes | 96.17% |

**Linear SVM** was selected as the final model.

## Project Structure

```text
product-category-classifier/
├── models/
│   └── product_category_model.pkl
├── notebooks/
│   └── product_category_classification.ipynb
├── products.csv
├── predict_category.py
├── train_model.py
└── README.md
```

## Installation

```bash
pip install pandas scikit-learn joblib
```

## Train the Model

```bash
python train_model.py
```

The script trains and evaluates the Linear SVM model and saves it in the `models` folder.

## Predict Product Categories

```bash
python predict_category.py
```

Enter a product title when prompted. The program displays the predicted category. Type `exit` to quit.

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Linear SVM
- Joblib
- Google Colab
- GitHub

## Conclusion
The final Linear SVM model achieved 96.42% accuracy on the test dataset. The project demonstrates data preprocessing, feature engineering, model comparison, evaluation, model saving, and interactive prediction.
