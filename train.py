"""
train.py - Train, evaluate, and save the Fake News Detection model.

Usage:
    python train.py --fake data/Fake.csv --true data/True.csv
"""

import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import load_and_prepare

MODELS = {
    "lr":  LogisticRegression(max_iter=1000, random_state=42),
    "nb":  MultinomialNB(),
    "svm": LinearSVC(max_iter=2000, random_state=42),
}


def plot_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fake", "Real"],
                yticklabels=["Fake", "Real"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - " + model_name.upper())
    plt.tight_layout()
    path = "models/confusion_matrix_" + model_name + ".png"
    plt.savefig(path)
    print("  Saved -> " + path)
    plt.close()


def train(fake_path, true_path, model_key="lr"):
    # 1. Load & preprocess
    print("Loading and preprocessing data...")
    df = load_and_prepare(fake_path, true_path)
    print("  Dataset size: " + str(len(df)) + " rows")
    print("  Label balance:\n" + df["label"].value_counts().to_string() + "\n")

    # 2. Train/test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # 3. TF-IDF vectorisation
    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=10_000, ngram_range=(1, 1))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    # 4. Train chosen model
    model = MODELS[model_key]
    print("Training " + model_key.upper() + " model...")
    model.fit(X_train_vec, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "-" * 45)
    print("  Accuracy : " + str(round(acc, 4)) + " (" + str(round(acc * 100, 2)) + "%)")
    print("-" * 45)
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, model_key)

    # 6. Save artefacts
    os.makedirs("models", exist_ok=True)
    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("models/model_" + model_key + ".pkl", "wb") as f:
        pickle.dump(model, f)
    print("\nSaved: models/vectorizer.pkl | models/model_" + model_key + ".pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fake News Detector")
    parser.add_argument("--fake",  default="data/Fake.csv", help="Path to Fake.csv")
    parser.add_argument("--true",  default="data/True.csv", help="Path to True.csv")
    parser.add_argument("--model", default="lr", choices=MODELS.keys(),
                        help="Model: lr | nb | svm  (default: lr)")
    args = parser.parse_args()
    train(args.fake, args.true, args.model)
