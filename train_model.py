
import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


DATA_PATH = "IMLP6_TASK_03-products.csv"
MODEL_PATH = "models/product_category_model.pkl"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    df.columns = df.columns.str.strip()

    df = df.dropna(subset=["Product Title", "Category Label"])

    df["Product Title"] = df["Product Title"].astype(str).str.strip()
    df = df[df["Product Title"] != ""]

    category_mapping = {
        "fridge": "Fridges",
        "CPU": "CPUs",
        "Mobile Phone": "Mobile Phones"
    }

    df["Category Label"] = df["Category Label"].replace(category_mapping)

    df["clean_title"] = df["Product Title"].apply(clean_text)

    X = df["clean_title"]
    y = df["Category Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training Linear SVM model...")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
            min_df=2,
            sublinear_tf=True
        )),
        ("classifier", LinearSVC(class_weight="balanced"))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nModel accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
