

import joblib
import re

MODEL_PATH = "models/product_category_model.pkl"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_model():
    return joblib.load(MODEL_PATH)


def predict_category(model, product_title):
    cleaned_title = clean_text(product_title)
    prediction = model.predict([cleaned_title])
    return prediction[0]


def main():
    print("=" * 50)
    print("PRODUCT CATEGORY CLASSIFIER")
    print("=" * 50)

    model = load_model()

    print("\nEnter a product title to predict its category.")
    print("Type 'exit' to quit the program.\n")

    while True:
        product_title = input("Product title: ").strip()

        if product_title.lower() == "exit":
            print("Program terminated.")
            break

        if not product_title:
            print("Please enter a valid product title.\n")
            continue

        category = predict_category(model, product_title)

        print(f"Predicted category: {category}")
        print("-" * 50)


if __name__ == "__main__":
    main()
