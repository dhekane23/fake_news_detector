"""
preprocess.py - Text cleaning and preprocessing pipeline (no nltk dependency)
"""

import re
import pandas as pd

# Built-in English stopwords (no nltk/scipy needed)
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "ought", "used", "it", "its", "this", "that", "these", "those", "i",
    "me", "my", "we", "our", "you", "your", "he", "his", "she", "her",
    "they", "their", "what", "which", "who", "whom", "not", "no", "nor",
    "so", "yet", "both", "either", "neither", "as", "if", "then", "than",
    "too", "very", "just", "about", "above", "after", "before", "between",
    "into", "through", "during", "up", "down", "out", "off", "over",
    "under", "again", "also", "any", "each", "more", "most", "other",
    "some", "such", "only", "own", "same", "s", "t", "don", "now", "here",
    "there", "when", "where", "why", "how", "all", "both", "few", "more",
}


def _stem(word):
    """Minimal suffix-stripping stemmer (no external deps)."""
    for suffix in ("ing", "tion", "ness", "ment", "ies", "ed", "er", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) > 3:
            return word[: -len(suffix)]
    return word


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/HTML/punctuation, remove stopwords, stem."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # remove URLs
    text = re.sub(r"<.*?>", "", text)             # remove HTML tags
    text = re.sub(r"[^a-z\s]", "", text)          # keep only letters
    tokens = [_stem(t) for t in text.split() if t not in _STOP_WORDS]
    return " ".join(tokens)


def load_and_prepare(fake_path: str, true_path: str) -> pd.DataFrame:
    """
    Load Kaggle Fake/True CSVs, label them, merge, and clean.
    Expected columns: 'title', 'text'
    """
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    fake["label"] = 0   # 0 = Fake
    true["label"] = 1   # 1 = Real

    df = pd.concat([fake, true], ignore_index=True)
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df = df[["content", "label"]].dropna().sample(frac=1, random_state=42).reset_index(drop=True)
    df["content"] = df["content"].apply(clean_text)
    return df
