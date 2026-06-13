"""FastAPI service — REST endpoint for sentiment (+ dialect) prediction.

Run:
    pip install fastapi uvicorn
    uvicorn api:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger docs.

Endpoints:
    GET  /health
    POST /predict        {"text": "...", "threshold": 0.7}
    POST /predict_batch  {"texts": ["...", "..."]}
"""

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src import dialect as dialect_mod
from src.model import DEFAULT_THRESHOLD, SentimentAnalyzer

app = FastAPI(
    title="Dialectal Arabic Sentiment API",
    description="Sentiment (+ dialect) analysis for colloquial Arabic, "
                "powered by a fine-tuned CAMeLBERT-DA transformer.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()


@lru_cache(maxsize=1)
def get_dialect():
    return dialect_mod.load()


class PredictRequest(BaseModel):
    text: str = Field(..., examples=["التطبيق مره حلو ويجنن"])
    threshold: float = DEFAULT_THRESHOLD


class BatchRequest(BaseModel):
    texts: list[str]
    threshold: float = DEFAULT_THRESHOLD


@app.get("/health")
def health() -> dict:
    a = get_analyzer()
    return {"status": "ok", "model": a.model_name, "fine_tuned": a.is_finetuned}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    r = get_analyzer().predict(req.text, threshold=req.threshold)
    out = {
        "text": req.text,
        "sentiment": r.display_label,
        "raw_label": r.label,
        "confidence": round(r.confidence, 4),
        "is_confident": r.is_confident,
        "probabilities": {k: round(v, 4) for k, v in r.probabilities.items()},
    }
    dm = get_dialect()
    if dm is not None:
        d, dconf = dialect_mod.predict(dm, req.text)
        out["dialect"] = d
        out["dialect_confidence"] = round(dconf, 4)
    return out


@app.post("/predict_batch")
def predict_batch(req: BatchRequest) -> dict:
    results = get_analyzer().predict_batch(req.texts, threshold=req.threshold)
    return {
        "count": len(results),
        "results": [
            {"text": t, "sentiment": r.display_label,
             "confidence": round(r.confidence, 4)}
            for t, r in zip(req.texts, results)
        ],
    }
