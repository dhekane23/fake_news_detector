"""
predict.py - Load saved model and predict on new text.

Usage (CLI):
    python predict.py --model lr
    python predict.py --model lr --text "Breaking: Scientists discover water on Mars"
"""

import argparse
import pickle
from preprocess import clean_text

LABEL_MAP = {0: "[FAKE NEWS]", 1: "[REAL NEWS]"}


def load_artifacts(model_key: str = "lr"):
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(f"models/model_{model_key}.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model


def predict(text: str, vectorizer, model) -> dict:
    cleaned = clean_text(text)
    vec     = vectorizer.transform([cleaned])
    label   = model.predict(vec)[0]

    # Confidence score (probability) — not available for LinearSVC
    proba = None
    if hasattr(model, "predict_proba"):
        proba = round(float(model.predict_proba(vec).max()) * 100, 2)

    return {"label": LABEL_MAP[label], "confidence": proba, "cleaned": cleaned}


def interactive_loop(model_key: str) -> None:
    print(f"Loading model '{model_key}' …")
    vectorizer, model = load_artifacts(model_key)
    print("Model ready. Type 'quit' to exit.\n")

    while True:
        text = input("Enter news text: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue
        result = predict(text, vectorizer, model)
        print(f"  Prediction : {result['label']}")
        if result["confidence"]:
            print(f"  Confidence : {result['confidence']}%")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Fake/Real News")
    parser.add_argument("--model", default="lr", choices=["lr", "nb", "svm"])
    parser.add_argument("--text",  default=None, help="Single text to predict (optional)")
    args = parser.parse_args()

    vectorizer, model = load_artifacts(args.model)

    if args.text:
        result = predict(args.text, vectorizer, model)
        print(f"Prediction : {result['label']}")
        if result["confidence"]:
            print(f"Confidence : {result['confidence']}%")
    else:
        interactive_loop(args.model)
