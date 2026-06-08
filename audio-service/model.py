import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.device import get_device

# superb emotion labels -> valence
_VALENCE = {"hap": 1.0, "neu": 0.0, "sad": -1.0, "ang": -1.0}
_pipe = None


def _get_pipe():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        device = 0 if get_device() == "cuda" else -1
        _pipe = pipeline("audio-classification",
                         model="superb/wav2vec2-base-superb-er", device=device)
    return _pipe


def score_audio(wav_path: str):
    """Return (valence in [-1,1], label, raw)."""
    pipe = _get_pipe()
    preds = pipe(str(wav_path), top_k=4)  # [{label, score}, ...]
    valence = sum(_VALENCE.get(p["label"], 0.0) * float(p["score"]) for p in preds)
    valence = max(-1.0, min(1.0, valence))
    label = "positive" if valence > 0.15 else "negative" if valence < -0.15 else "neutral"
    return valence, label, {"preds": preds}
