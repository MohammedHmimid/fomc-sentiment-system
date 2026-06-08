"""NLP channel service — scores an FOMC transcript with FinBERT.

The model is loaded lazily (first request only) so importing this module stays
cheap and unit tests can monkeypatch `score_text` without pulling in transformers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from pydantic import BaseModel

from shared.device import get_device
from shared.schema import ChannelScore, valence_label

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

_pipe = None


def _get_pipe():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        device = 0 if get_device() == "cuda" else -1
        _pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)
    return _pipe


def score_text(text: str):
    """Return (valence in [-1, 1], label, raw) for an FOMC transcript."""
    if not text.strip():
        raise ValueError("empty transcript")
    pipe = _get_pipe()
    # FinBERT handles ~512 tokens; average over ~1500-char chunks across the whole transcript.
    chunks = [text[i:i + 1500] for i in range(0, max(len(text), 1), 1500)] or [""]
    vals, all_outputs = [], []
    for ch in chunks:
        out = pipe(ch[:512])[0]
        all_outputs.append(out)
        lbl = out["label"].lower()
        p = float(out["score"])
        vals.append(p if lbl == "positive" else -p if lbl == "negative" else 0.0)
    valence = sum(vals) / len(vals)
    return valence, valence_label(valence), {"chunks": all_outputs, "n_chunks": len(vals)}


app = FastAPI(title="nlp-service")


class AnalyzeRequest(BaseModel):
    event_id: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "nlp"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    path = DATA_DIR / req.event_id / "transcript.txt"
    try:
        text = path.read_text(encoding="utf-8")
        valence, label, raw = score_text(text)
        return ChannelScore(event_id=req.event_id, channel="nlp", score=valence,
                            label=label, raw=raw, ok=True).model_dump()
    except Exception as e:
        return ChannelScore(event_id=req.event_id, channel="nlp", score=0.0,
                            label="neutral", raw={}, ok=False, error=str(e)).model_dump()
