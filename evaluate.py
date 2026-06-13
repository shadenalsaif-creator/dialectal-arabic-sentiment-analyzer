"""Evaluate the sentiment model on a labeled CSV.

Usage:
    python evaluate.py --data data/labeled_sample.csv

Expects columns: text, label   (label in {positive, negative, neutral})

Outputs:
    - Classification report (precision / recall / F1 per class) to stdout
    - assets/confusion_matrix.png
    - assets/metrics.txt

Run this after any change to preprocessing or after fine-tuning, and
commit the refreshed assets so the README results stay honest.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from src.model import SentimentAnalyzer

LABELS = ["positive", "negative", "neutral"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/labeled_sample.csv")
    parser.add_argument("--out-dir", default="assets")
    args = parser.parse_args()

    df = pd.read_csv(args.data).dropna(subset=["text", "label"])
    df["label"] = df["label"].str.strip().str.lower()
    unknown = set(df["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown labels in data: {unknown}. Expected {LABELS}.")

    print(f"Evaluating on {len(df)} labeled examples from {args.data} ...")
    analyzer = SentimentAnalyzer()
    preds = [r.label for r in analyzer.predict_batch(df["text"].tolist())]

    report = classification_report(df["label"], preds, labels=LABELS, digits=3)
    print(report)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    (out_dir / "metrics.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(df["label"], preds, labels=LABELS)
    disp = ConfusionMatrixDisplay(cm, display_labels=LABELS)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — CAMeLBERT-DA Sentiment")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    print(f"Saved: {out_dir / 'confusion_matrix.png'} and {out_dir / 'metrics.txt'}")


if __name__ == "__main__":
    main()
