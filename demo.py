"""
demo.py - Full pipeline demo using synthetic data (no Kaggle download needed).

Run:  python demo.py
"""

import os, pickle, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import clean_text

warnings.filterwarnings("ignore")

# ── 1. Synthetic dataset ──────────────────────────────────────────────────────
FAKE_SAMPLES = [
    "SHOCKING: Government secretly putting chemicals in water supply to control minds",
    "Doctors HATE this one weird trick that cures all diseases overnight",
    "BREAKING: Aliens landed in Nevada, government covering it up completely",
    "Miracle cure discovered, Big Pharma suppressing it to protect profits",
    "Celebrity caught in massive scandal that mainstream media refuses to report",
    "Scientists ADMIT climate change is a hoax invented by globalists",
    "Secret society controls all world governments, leaked documents reveal",
    "Vaccines contain microchips to track your every movement, whistleblower says",
    "Moon landing was staged in Hollywood studio, new evidence proves it",
    "5G towers spreading virus, thousands of engineers confirm the truth",
    "Politician caught stealing billions, media blackout ordered immediately",
    "Ancient cure suppressed for 100 years finally revealed by brave doctor",
    "World leaders meet in secret bunker to plan population reduction scheme",
    "New study proves smartphones cause instant brain damage in children",
    "Billionaire funds plan to replace human workers with robots by next year",
]

REAL_SAMPLES = [
    "Federal Reserve raises interest rates by 25 basis points amid inflation concerns",
    "NASA successfully launches new Mars rover to study geological formations",
    "Parliament passes new climate legislation targeting net-zero emissions by 2050",
    "Tech company reports quarterly earnings above analyst expectations",
    "Scientists publish peer-reviewed study on new cancer treatment in Nature journal",
    "Central bank governor addresses economic outlook at annual conference",
    "Olympic committee announces host city for upcoming summer games",
    "University researchers develop more efficient solar panel technology",
    "Health ministry updates vaccination guidelines following expert review",
    "Stock markets close higher after positive employment data released",
    "International trade agreement signed between multiple nations after negotiations",
    "City council approves infrastructure budget for road and bridge repairs",
    "Medical researchers identify new genetic marker linked to heart disease risk",
    "Renewable energy capacity surpasses coal for first time in national grid",
    "Central bank publishes annual financial stability report with key findings",
]

# Augment to ~300 samples per class
def augment(samples, n=300):
    rng = np.random.default_rng(42)
    base = samples * (n // len(samples) + 1)
    chosen = rng.choice(base, size=n, replace=False).tolist()
    # Add minor noise to make samples unique
    return [s + f" {rng.integers(1000)}" for s in chosen]

fake_texts = augment(FAKE_SAMPLES)
real_texts = augment(REAL_SAMPLES)

df = pd.DataFrame({
    "content": fake_texts + real_texts,
    "label":   [0] * len(fake_texts) + [1] * len(real_texts)
}).sample(frac=1, random_state=42).reset_index(drop=True)

df["content"] = df["content"].apply(clean_text)
print(f"Dataset: {len(df)} rows | Fake={sum(df.label==0)} | Real={sum(df.label==1)}\n")

# ── 2. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# ── 3. TF-IDF ─────────────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ── 4. Train Logistic Regression ──────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

# Confusion matrix plot
os.makedirs("models", exist_ok=True)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"], ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix — Logistic Regression")
plt.tight_layout()
plt.savefig("models/confusion_matrix_demo.png")
print("Confusion matrix saved -> models/confusion_matrix_demo.png\n")
plt.close()

# ── 6. Save model ─────────────────────────────────────────────────────────────
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("models/model_lr.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved -> models/vectorizer.pkl + models/model_lr.pkl\n")

# ── 7. Example predictions ────────────────────────────────────────────────────
EXAMPLES = [
    ("Scientists discover water ice deposits on the Moon's south pole", "Real"),
    ("Government putting mind-control chemicals in tap water, whistleblower reveals", "Fake"),
    ("Central bank raises interest rates to combat rising inflation", "Real"),
    ("Miracle pill cures cancer in 24 hours, Big Pharma hiding the truth", "Fake"),
]

print("-" * 55)
print(f"{'News Text':<42} {'Expected':<8} {'Predicted'}")
print("-" * 55)
for text, expected in EXAMPLES:
    vec    = vectorizer.transform([clean_text(text)])
    pred   = model.predict(vec)[0]
    conf   = round(model.predict_proba(vec).max() * 100, 1)
    label  = "Real" if pred == 1 else "Fake"
    status = "OK" if label == expected else "WRONG"
    print(f"{text[:42]:<42} {expected:<8} {label} ({conf}%) {status}")
print("-" * 55)
