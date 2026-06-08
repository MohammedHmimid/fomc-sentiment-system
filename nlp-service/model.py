import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.device import get_device

_pipe = None


def _get_pipe():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        device = 0 if get_device() == "cuda" else -1
        _pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)
    return _pipe


def score_text(text: str):
    """Return (valence in [-1,1], label, raw)."""
    pipe = _get_pipe()
    # FinBERT max 512 tokens; average over ~500-char chunks.
    chunks = [text[i:i + 1500] for i in range(0, max(len(text), 1), 1500)] or [""]
    vals = []
    last = {}
    for ch in chunks[:8]:  # cap chunks for speed
        out = pipe(ch[:512])[0]
        last = out
        lbl = out["label"].lower()
        p = float(out["score"])
        vals.append(p if lbl == "positive" else -p if lbl == "negative" else 0.0)
    valence = sum(vals) / len(vals)
    label = "positive" if valence > 0.15 else "negative" if valence < -0.15 else "neutral"
    return valence, label, {"last": last, "n_chunks": len(vals)}
