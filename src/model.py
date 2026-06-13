"""Sentiment model wrapper around CAMeLBERT-DA.

Model: CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment
    - Base: CAMeLBERT-DA (BERT pre-trained on ~54GB of dialectal Arabic)
    - Head: 3-class sentiment classifier (positive / negative / neutral)
      fine-tuned by CAMeL Lab (NYU Abu Dhabi).

Design decisions an interviewer may ask about:
    * Why CAMeLBERT-DA and not MSA models (e.g. AraBERT)?
      Because the target domain is COLLOQUIAL text (tweets, app reviews in
      Saudi/Gulf dialect). A model pre-trained on dialectal data has seen
      words like "مره", "زفت", "يجنن" -- MSA-only models largely have not.
    * Why confidence thresholding?
      Softmax probabilities are not calibrated guarantees. When the top
      probability is below the threshold we surface the prediction as
      "Uncertain" instead of silently committing -- this matters in batch
      analytics, where low-confidence rows should be reviewed by a human.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.preprocess import clean_text

# The model we fine-tuned on ArSarcasm-v2 (saved by train.py).
FINETUNED_PATH = "models/camelbert-finetuned"
# The public base checkpoint we started from.
PRETRAINED_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"


def default_model() -> str:
    """Prefer our fine-tuned model; fall back to the public checkpoint.

    This keeps the repo runnable for anyone who clones it WITHOUT the
    large fine-tuned weights (436MB) — the app still works out of the
    box on the base model, and automatically upgrades to the fine-tuned
    one once `python train.py` has produced it (or it's pulled from the
    Hugging Face Hub).
    """
    cfg = Path(FINETUNED_PATH) / "config.json"
    return FINETUNED_PATH if cfg.exists() else PRETRAINED_NAME


# Backwards-compatible alias used elsewhere in the app.
MODEL_NAME = default_model()
MAX_LENGTH = 128
DEFAULT_THRESHOLD = 0.70

LABEL_AR = {"positive": "إيجابي", "negative": "سلبي", "neutral": "محايد"}
LABEL_EMOJI = {"positive": "😊", "negative": "😠", "neutral": "😐"}


@dataclass
class SentimentResult:
    label: str            # positive / negative / neutral
    confidence: float     # top softmax probability
    is_confident: bool    # confidence >= threshold
    probabilities: dict   # full distribution, e.g. {"positive": 0.91, ...}

    @property
    def display_label(self) -> str:
        return self.label if self.is_confident else "uncertain"


class SentimentAnalyzer:
    """Loads the model once and serves single + batch predictions."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        model_name = model_name or default_model()
        self.model_name = model_name
        self.is_finetuned = Path(FINETUNED_PATH).resolve() == Path(model_name).resolve() \
            if Path(model_name).exists() else False
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        # id2label comes from the checkpoint config. Normalize ArSarcasm-style
        # short codes (POS/NEG/NEU) to full words so the UI is consistent.
        norm = {"pos": "positive", "neg": "negative", "neu": "neutral"}
        self.id2label = {
            i: norm.get(l.lower(), l.lower())
            for i, l in self.model.config.id2label.items()
        }

    @torch.inference_mode()
    def predict(self, text: str, threshold: float = DEFAULT_THRESHOLD) -> SentimentResult:
        return self.predict_batch([text], threshold=threshold)[0]

    @torch.inference_mode()
    def predict_batch(
        self,
        texts: list[str],
        threshold: float = DEFAULT_THRESHOLD,
        batch_size: int = 32,
    ) -> list[SentimentResult]:
        """Tokenize -> forward pass -> softmax, in mini-batches.

        Mini-batching keeps memory bounded for large CSV uploads while
        still being much faster than one-by-one inference.
        """
        cleaned = [clean_text(t) for t in texts]
        results: list[SentimentResult] = []

        for start in range(0, len(cleaned), batch_size):
            chunk = cleaned[start:start + batch_size]
            enc = self.tokenizer(
                chunk,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu()

            for row in probs:
                dist = {self.id2label[i]: float(p) for i, p in enumerate(row)}
                label = max(dist, key=dist.get)
                conf = dist[label]
                results.append(
                    SentimentResult(
                        label=label,
                        confidence=conf,
                        is_confident=conf >= threshold,
                        probabilities=dist,
                    )
                )
        return results
