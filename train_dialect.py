"""Train the Arabic dialect detector on ArSarcasm-v2.

Usage:
    python train_dialect.py
    python train_dialect.py --data data/arsarcasm_train.csv

Saves models/dialect_detector.joblib and prints a classification report.
The detector adds a "which dialect?" signal to the Streamlit app and
powers the per-dialect analysis. Tiny dialects (e.g. magreb, ~43 rows)
are dropped automatically — too few examples to learn reliably.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.dialect import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/arsarcasm_train.csv")
    args = parser.parse_args()

    m = train(train_csv=args.data)
    print(f"Dialect detector — accuracy {m['test_accuracy']:.3f} | "
          f"macro-F1 {m['test_macro_f1']:.3f}")
    if m["dropped_classes"]:
        print(f"Dropped tiny dialects: {m['dropped_classes']}")
    print(m["report"])

    assets = Path("assets"); assets.mkdir(exist_ok=True)
    labels = m["labels"]
    cm = confusion_matrix(m["y_test"], m["preds"], labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Greens", colorbar=False, xticks_rotation=30)
    ax.set_title("Dialect Detector — Confusion Matrix")
    fig.tight_layout(); fig.savefig(assets / "confusion_dialect.png", dpi=150)
    print(f"Saved: models/dialect_detector.joblib, assets/confusion_dialect.png")


if __name__ == "__main__":
    main()
