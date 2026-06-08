import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from pydantic import BaseModel
from shared.schema import ChannelScore

import importlib.util
_m = importlib.util.spec_from_file_location("audio_model", Path(__file__).with_name("model.py"))
_model = importlib.util.module_from_spec(_m)
_m.loader.exec_module(_model)
score_audio = _model.score_audio

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
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
