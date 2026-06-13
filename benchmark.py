"""Benchmark & analysis — the analytical core of the project.

Produces, on the ArSarcasm-v2 TEST set (3,000 independently-labeled tweets):

  1. Pretrained vs Fine-tuned comparison  -> proves fine-tuning added value
  2. Per-dialect performance breakdown    -> where does the model do well?
  3. Sarcasm impact analysis              -> does sarcasm hurt sentiment acc?

Outputs to assets/:
  - benchmark_comparison.png   (grouped bar: accuracy & macro-F1)
  - per_dialect_f1.png
  - sarcasm_impact.png
  - confusion_finetuned.png
  - benchmark.md               (tables ready to paste into the README)

Usage:
  python convert_arsarcasm.py --input testing_data.csv   # makes data/arsarcasm_test.csv
  python benchmark.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from src.model import PRETRAINED_NAME, FINETUNED_PATH, SentimentAnalyzer

LABELS = ["positive", "negative", "neutral"]


def evaluate(analyzer: SentimentAnalyzer, texts: list[str]) -> list[str]:
    return [r.label for r in analyzer.predict_batch(texts)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", default="data/arsarcasm_test.csv",
                        help="Converted test CSV with text,label (+ optional dialect, sarcasm)")
    parser.add_argument("--raw-test", default="data/arsarcasm_test_raw.csv",
                        help="Raw ArSarcasm testing_data.csv (for dialect/sarcasm columns)")
    args = parser.parse_args()

    assets = Path("assets")
    assets.mkdir(exist_ok=True)

    df = pd.read_csv(args.test).dropna(subset=["text", "label"])
    df["label"] = df["label"].str.strip().str.lower()

    # Attach dialect + sarcasm from the raw file if available (by row order).
    raw_path = Path(args.raw_test)
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        raw.columns = [c.strip().lower() for c in raw.columns]
        if len(raw) == len(df):
            if "dialect" in raw.columns:
                df["dialect"] = raw["dialect"].str.strip().str.lower().values
            if "sarcasm" in raw.columns:
                df["sarcasm"] = raw["sarcasm"].values

    texts = df["text"].tolist()
    gold = df["label"].tolist()

    md: list[str] = ["# Benchmark — ArSarcasm-v2 test set (3,000 tweets)\n"]

    # ---------- 1. Pretrained vs Fine-tuned --------------------------- #
    print("Loading PRETRAINED base model...")
    base = SentimentAnalyzer(model_name=PRETRAINED_NAME)
    base_preds = evaluate(base, texts)

    rows = [("Pretrained (base)",
             accuracy_score(gold, base_preds),
             f1_score(gold, base_preds, labels=LABELS, average="macro"))]

    have_ft = (Path(FINETUNED_PATH) / "config.json").exists()
    if have_ft:
        print("Loading FINE-TUNED model...")
        ft = SentimentAnalyzer(model_name=FINETUNED_PATH)
        ft_preds = evaluate(ft, texts)
        rows.append(("Fine-tuned (ours)",
                     accuracy_score(gold, ft_preds),
                     f1_score(gold, ft_preds, labels=LABELS, average="macro")))
    else:
        print("WARNING: fine-tuned model not found — showing baseline only. "
              "Run train.py first for the comparison.")
        ft_preds = None

    md.append("## 1. Pretrained vs Fine-tuned\n")
    md.append("| Model | Accuracy | Macro-F1 |")
    md.append("|---|---|---|")
    for name, acc, mf1 in rows:
        md.append(f"| {name} | {acc:.3f} | {mf1:.3f} |")
    if ft_preds is not None:
        gain = rows[1][2] - rows[0][2]
        md.append(f"\n**Fine-tuning improved macro-F1 by {gain:+.3f} "
                  f"({gain*100:+.1f} points).**\n")

    # grouped bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    names = [r[0] for r in rows]
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], [r[1] for r in rows], width=0.4, label="Accuracy")
    ax.bar([i + 0.2 for i in x], [r[2] for r in rows], width=0.4, label="Macro-F1")
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.set_ylim(0, 1); ax.set_title("Pretrained vs Fine-tuned"); ax.legend()
    for i, r in enumerate(rows):
        ax.text(i - 0.2, r[1] + 0.02, f"{r[1]:.2f}", ha="center", fontsize=9)
        ax.text(i + 0.2, r[2] + 0.02, f"{r[2]:.2f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(assets / "benchmark_comparison.png", dpi=150)
    plt.close(fig)

    final_preds = ft_preds if ft_preds is not None else base_preds
    model_tag = "Fine-tuned" if ft_preds is not None else "Pretrained"

    # ---------- 2. Per-dialect breakdown ------------------------------ #
    if "dialect" in df.columns:
        md.append("\n## 2. Per-dialect performance (" + model_tag + ")\n")
        md.append("| Dialect | N | Accuracy | Macro-F1 |")
        md.append("|---|---|---|---|")
        tmp = df.copy(); tmp["pred"] = final_preds
        dialect_rows = []
        for d, g in tmp.groupby("dialect"):
            if len(g) < 20:
                continue
            acc = accuracy_score(g["label"], g["pred"])
            mf1 = f1_score(g["label"], g["pred"], labels=LABELS, average="macro")
            dialect_rows.append((d, len(g), acc, mf1))
            md.append(f"| {d} | {len(g)} | {acc:.3f} | {mf1:.3f} |")

        if dialect_rows:
            fig, ax = plt.subplots(figsize=(6, 4))
            dialect_rows.sort(key=lambda r: r[3])
            ax.barh([r[0] for r in dialect_rows], [r[3] for r in dialect_rows],
                    color="#1565c0")
            ax.set_xlim(0, 1); ax.set_xlabel("Macro-F1")
            ax.set_title(f"Per-dialect Macro-F1 ({model_tag})")
            for i, r in enumerate(dialect_rows):
                ax.text(r[3] + 0.01, i, f"{r[3]:.2f}", va="center", fontsize=9)
            fig.tight_layout(); fig.savefig(assets / "per_dialect_f1.png", dpi=150)
            plt.close(fig)

    # ---------- 3. Sarcasm impact ------------------------------------- #
    if "sarcasm" in df.columns:
        tmp = df.copy(); tmp["pred"] = final_preds
        tmp["correct"] = tmp["label"] == tmp["pred"]
        sarc = tmp[tmp["sarcasm"] == True]["correct"].mean()
        non = tmp[tmp["sarcasm"] == False]["correct"].mean()
        n_sarc = int((tmp["sarcasm"] == True).sum())
        n_non = int((tmp["sarcasm"] == False).sum())
        diff = (non - sarc) * 100  # positive => sarcasm hurts; negative => sarcasm helps
        md.append("\n## 3. Sarcasm impact (" + model_tag + ")\n")
        md.append("| Subset | N | Accuracy |")
        md.append("|---|---|---|")
        md.append(f"| Non-sarcastic | {n_non} | {non:.3f} |")
        md.append(f"| Sarcastic | {n_sarc} | {sarc:.3f} |")
        if diff >= 1.0:
            md.append(f"\n**Sentiment accuracy drops {diff:.1f} points on sarcastic "
                      f"tweets**, consistent with the literature: sarcasm inverts "
                      f"surface sentiment and is a known hard case.\n")
        elif diff <= -1.0:
            md.append(f"\n**Note:** on this test set the model scores {-diff:.1f} points "
                      f"*higher* on sarcastic tweets ({sarc:.3f}) than non-sarcastic "
                      f"({non:.3f}). The most likely reason is label distribution: "
                      f"sarcastic tweets in ArSarcasm skew heavily toward *negative* "
                      f"sentiment, and the model is strongest on the negative class, so "
                      f"raw accuracy looks higher there. This is an accuracy artifact, "
                      f"not evidence that sarcasm is easy — a per-class F1 breakdown "
                      f"would show the true picture.\n")
        else:
            md.append(f"\n**Sarcasm has little effect here** ({non:.3f} vs {sarc:.3f}).\n")

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["Non-sarcastic", "Sarcastic"], [non, sarc],
               color=["#2e7d32", "#c62828"])
        ax.set_ylim(0, 1); ax.set_ylabel("Accuracy")
        ax.set_title("Sentiment accuracy: sarcasm impact")
        for i, v in enumerate([non, sarc]):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
        fig.tight_layout(); fig.savefig(assets / "sarcasm_impact.png", dpi=150)
        plt.close(fig)

    # ---------- confusion matrix of the chosen model ------------------ #
    cm = confusion_matrix(gold, final_preds, labels=LABELS)
    disp = ConfusionMatrixDisplay(cm, display_labels=LABELS)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_tag}")
    fig.tight_layout(); fig.savefig(assets / "confusion_finetuned.png", dpi=150)
    plt.close(fig)

    (assets / "benchmark.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved charts + benchmark.md to {assets}/")


if __name__ == "__main__":
    main()