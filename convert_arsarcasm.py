"""Convert the ArSarcasm-v2 dataset to this project's text,label format.

ArSarcasm-v2 (Abu Farha et al., WANLP 2021): 15,548 Arabic tweets
independently annotated for sentiment (positive/negative/neutral),
sarcasm, and dialect. Download (no signup needed):

    https://github.com/iabufarha/ArSarcasm-v2
    -> Code -> Download ZIP -> extract -> find testing_data.csv

We use the official 3,000-tweet TEST split for evaluation, which makes
the README claim honest: "evaluated on 3,000 independently-labeled
dialectal tweets (ArSarcasm-v2 test set)".

Usage:
    python convert_arsarcasm.py --input testing_data.csv
    python convert_arsarcasm.py --input testing_data.csv --dialect gulf
    python evaluate.py --data data/arsarcasm_test.csv

The --dialect filter (msa / gulf / egypt / levant / magreb) lets you
report per-dialect performance — a strong analysis angle.
"""

import argparse
from pathlib import Path

import pandas as pd

TEXT_CANDIDATES = ("tweet", "text", "tweet_text")
LABEL_MAP = {
    "positive": "positive", "pos": "positive", "1": "positive",
    "negative": "negative", "neg": "negative", "-1": "negative",
    "neutral": "neutral", "neu": "neutral", "0": "neutral",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="Path to ArSarcasm-v2 testing_data.csv (or training_data.csv)")
    parser.add_argument("--output", default="data/arsarcasm_test.csv")
    parser.add_argument("--dialect", default=None,
                        help="Optional dialect filter: msa, gulf, egypt, levant, magreb")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df.columns = [c.strip().lower() for c in df.columns]

    text_col = next((c for c in TEXT_CANDIDATES if c in df.columns), None)
    if text_col is None:
        raise ValueError(f"No text column found. Columns: {list(df.columns)}")
    if "sentiment" not in df.columns:
        raise ValueError(f"No 'sentiment' column found. Columns: {list(df.columns)}")

    if args.dialect:
        if "dialect" not in df.columns:
            raise ValueError("This file has no 'dialect' column.")
        before = len(df)
        df = df[df["dialect"].str.strip().str.lower() == args.dialect.lower()]
        print(f"Dialect filter '{args.dialect}': {before} -> {len(df)} rows")

    out = pd.DataFrame({
        "text": df[text_col].astype(str).str.strip(),
        "label": df["sentiment"].astype(str).str.strip().str.lower().map(LABEL_MAP),
    })
    dropped = out["label"].isna().sum()
    out = out.dropna(subset=["label"])
    out = out[out["text"].str.len() > 0]

    Path(args.output).parent.mkdir(exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")

    # Also persist a raw copy (with dialect + sarcasm) aligned to `out`,
    # so benchmark.py can do per-dialect and sarcasm analysis.
    if not args.dialect:
        raw_cols = [c for c in ("dialect", "sarcasm") if c in df.columns]
        if raw_cols:
            raw_out = Path(args.output).with_name("arsarcasm_test_raw.csv")
            keep = df.loc[out.index, raw_cols] if len(df) == len(out) else df[raw_cols]
            keep.to_csv(raw_out, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(out)} rows to {args.output} "
          f"({dropped} rows dropped for unknown labels)")
    print(out["label"].value_counts().to_string())
    print(f"\nNext: python evaluate.py --data {args.output}")


if __name__ == "__main__":
    main()
