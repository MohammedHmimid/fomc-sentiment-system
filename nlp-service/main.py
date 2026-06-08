import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from pydantic import BaseModel

from shared.schema import ChannelScore

# load sibling model.py explicitly
import importlib.util
_m = importlib.util.spec_from_file_location("nlp_model", Path(__file__).with_name("model.py"))
_model = importlib.util.module_from_spec(_m)
_m.loader.exec_module(_model)
score_text = _model.score_text

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

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
