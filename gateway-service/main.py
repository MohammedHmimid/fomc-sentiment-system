import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

import importlib.util
_o = importlib.util.spec_from_file_location("orchestrator", Path(__file__).with_name("orchestrator.py"))
_orch = importlib.util.module_from_spec(_o)
_o.loader.exec_module(_orch)
analyze_event = _orch.analyze_event

URLS = {
    "nlp": os.getenv("NLP_URL", "http://localhost:8001") + "/analyze",
    "audio": os.getenv("AUDIO_URL", "http://localhost:8002") + "/analyze",
    "vision": os.getenv("VISION_URL", "http://localhost:8003") + "/analyze",
    "fusion": os.getenv("FUSION_URL", "http://localhost:8004") + "/fuse",
}

app = FastAPI(title="gateway-service")


class AnalyzeRequest(BaseModel):
    event_id: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "gateway"}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    async with httpx.AsyncClient(timeout=600.0) as client:
        return await analyze_event(req.event_id, client, URLS)
