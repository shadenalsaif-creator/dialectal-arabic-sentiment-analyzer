"""Fine-tune CAMeLBERT-DA on your own labeled dialect data.

This is what makes the CV line "fine-tuned CAMeLBERT transformer" true:
you continue training the pre-trained checkpoint on a labeled CSV (e.g.
Saudi-dialect app reviews) so it adapts to your exact domain.

Usage:
    python train.py --data data/labeled_sample.csv --epochs 3

Expects columns: text, label   (label in {positive, negative, neutral})
Saves the fine-tuned model to models/camelbert-finetuned/.

To make the Streamlit app use your fine-tuned model:
    SentimentAnalyzer(model_name="models/camelbert-finetuned")

Tip: for a real fine-tune, use a public dialectal dataset such as
ASTD or ArSarcasm-v2 (thousands of rows), not just the demo sample.
"""

import argparse

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.preprocess import clean_text

BASE_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/labeled_sample.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output", default="models/camelbert-finetuned")
    args = parser.parse_args()

    df = pd.read_csv(args.data).dropna(subset=["text", "label"])
    df["label"] = df["label"].str.strip().str.lower().map(LABEL2ID)
    if df["label"].isna().any():
        raise ValueError(f"Labels must be one of {list(LABEL2ID)}.")
    df["text"] = df["text"].map(clean_text)

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"Train: {len(train_df)} rows | Validation: {len(val_df)} rows")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]]).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print({k: round(v, 4) for k, v in metrics.items() if k.startswith("eval_")})

    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Fine-tuned model saved to {args.output}/")


if __name__ == "__main__":
    main()
