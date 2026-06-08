import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.device import get_device
from shared.schema import valence_label

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
    if not text.strip():
        raise ValueError("empty transcript")
    pipe = _get_pipe()
    # FinBERT max 512 tokens; average over ~1500-char chunks (whole transcript).
    chunks = [text[i:i + 1500] for i in range(0, max(len(text), 1), 1500)] or [""]
    vals = []
    all_outputs = []
    for ch in chunks:
        out = pipe(ch[:512])[0]
        all_outputs.append(out)
        lbl = out["label"].lower()
        p = float(out["score"])
        vals.append(p if lbl == "positive" else -p if lbl == "negative" else 0.0)
    valence = sum(vals) / len(vals)
    label = valence_label(valence)
    return valence, label, {"chunks": all_outputs, "n_chunks": len(vals)}
