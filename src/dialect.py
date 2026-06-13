"""Arabic dialect detector — TF-IDF + Linear SVM.

Trained on the dialect labels that ship with ArSarcasm-v2 (msa, egypt,
gulf, levant, magreb). This is a SEPARATE, lightweight model from the
sentiment transformer: it adds a "which dialect?" signal to the app so
a single tweet gets both dialect + sentiment, e.g.

    Input : المطعم يجنن وأنصح فيه بشده
    Output: Dialect = Gulf | Sentiment = Positive

Why TF-IDF + LinearSVC (not another transformer)?
    Dialect cues live mostly in characteristic words/sub-words
    (يجنن, عشان, ايه, برشا...). A char/word n-gram linear model captures
    these cheaply with millisecond inference, and keeps the app light
    enough to deploy on a free Hugging Face Space alongside the
    sentiment transformer.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import clean_text

DIALECT_AR = {
    "msa": "الفصحى", "egypt": "مصري", "gulf": "خليجي",
    "levant": "شامي", "magreb": "مغاربي",
}
DIALECT_EMOJI = {
    "msa": "📰", "egypt": "🇪🇬", "gulf": "🌴", "levant": "🌿", "magreb": "🏜️",
}
DEFAULT_MODEL_PATH = "models/dialect_detector.joblib"


def build_pipeline() -> Pipeline:
    # Balanced Logistic Regression handles the heavy class imbalance in
    # ArSarcasm (MSA dominates) far better than a plain classifier, and
    # gives calibrated predict_proba out of the box for a confidence score.
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5),
            sublinear_tf=True, min_df=3, max_features=50000,
        )),
        ("clf", LogisticRegression(
            max_iter=3000, C=3.0, class_weight="balanced", n_jobs=-1,
        )),
    ])


# Classes with too few samples to learn reliably are dropped (documented).
MIN_CLASS_SIZE = 100


def train(
    train_csv: str | Path = "data/arsarcasm_train.csv",
    dialect_col: str = "dialect",
    text_col: str = "tweet",
    model_path: str | Path | None = DEFAULT_MODEL_PATH,
    seed: int = 42,
) -> dict:
    """Train the dialect detector from a CSV that has text + dialect columns.

    Works directly on the raw ArSarcasm training_data.csv (tweet, dialect),
    so no extra conversion step is needed.
    """
    df = pd.read_csv(train_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    if text_col not in df.columns:
        text_col = "text" if "text" in df.columns else df.columns[0]
    df = df.dropna(subset=[text_col, dialect_col])
    df[text_col] = df[text_col].map(clean_text)
    df = df[df[text_col].str.len() > 0]
    df[dialect_col] = df[dialect_col].str.strip().str.lower()

    # Drop dialects with too few examples to learn (e.g. magreb ~43 rows);
    # keeping them only hurts macro-F1 and produces a useless class.
    counts = df[dialect_col].value_counts()
    keep = counts[counts >= MIN_CLASS_SIZE].index
    dropped_classes = sorted(set(counts.index) - set(keep))
    if dropped_classes:
        df = df[df[dialect_col].isin(keep)]

    X, y = df[text_col], df[dialect_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    metrics = {
        "test_accuracy": accuracy_score(y_test, preds),
        "test_macro_f1": f1_score(y_test, preds, average="macro"),
        "report": classification_report(y_test, preds, digits=3),
        "y_test": y_test, "preds": preds,
        "labels": sorted(y.unique()),
        "dropped_classes": dropped_classes,
        "pipeline": pipe,
    }
    if model_path is not None:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, model_path)
    return metrics


def load(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Load the trained detector, or None if it hasn't been trained yet."""
    model_path = Path(model_path)
    return joblib.load(model_path) if model_path.exists() else None


def predict(pipe, text: str) -> tuple[str, float]:
    """Return (dialect, confidence) for one text."""
    cleaned = clean_text(text)
    proba = pipe.predict_proba([cleaned])[0]
    classes = pipe.classes_
    idx = proba.argmax()
    return classes[idx], float(proba[idx])
