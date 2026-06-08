import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from pydantic import BaseModel
from shared.schema import ChannelScore

import importlib.util
_m = importlib.util.spec_from_file_location("vision_model", Path(__file__).with_name("model.py"))
_model = importlib.util.module_from_spec(_m)
_m.loader.exec_module(_model)
score_video = _model.score_video
NoFaceFound = _model.NoFaceFound

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
app = FastAPI(title="vision-service")


class AnalyzeRequest(BaseModel):
    event_id: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "vision"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    mp4 = DATA_DIR / req.event_id / "video.mp4"
    try:
        if not mp4.exists():
            raise FileNotFoundError(f"missing {mp4}")
        valence, label, raw = score_video(mp4)
        return ChannelScore(event_id=req.event_id, channel="vision", score=valence,
                            label=label, raw=raw, ok=True).model_dump()
    except Exception as e:
        return ChannelScore(event_id=req.event_id, channel="vision", score=0.0,
                            label="neutral", raw={}, ok=False, error=str(e)).model_dump()
