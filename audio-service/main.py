"""Audio channel service — scores FOMC speech emotion with Wav2Vec2.

The model is loaded lazily (first request only) so importing this module stays
cheap and unit tests can monkeypatch `score_audio` without pulling in transformers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from pydantic import BaseModel

from shared.device import get_device
from shared.schema import ChannelScore, valence_label

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# superb speech-emotion labels -> valence
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


def score_audio(wav_path):
    """Return (valence in [-1, 1], label, raw) for a speech .wav file."""
    pipe = _get_pipe()
    preds = pipe(str(wav_path), top_k=4)  # [{label, score}, ...]
    valence = sum(_VALENCE.get(p["label"], 0.0) * float(p["score"]) for p in preds)
    valence = max(-1.0, min(1.0, valence))
    return valence, valence_label(valence), {"preds": preds}


app = FastAPI(title="audio-service")


class AnalyzeRequest(BaseModel):
    event_id: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "audio"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    wav = DATA_DIR / req.event_id / "audio.wav"
    try:
        if not wav.exists():
            raise FileNotFoundError(f"missing {wav}")
        valence, label, raw = score_audio(wav)
        return ChannelScore(event_id=req.event_id, channel="audio", score=valence,
                            label=label, raw=raw, ok=True).model_dump()
    except Exception as e:
        return ChannelScore(event_id=req.event_id, channel="audio", score=0.0,
                            label="neutral", raw={}, ok=False, error=str(e)).model_dump()
