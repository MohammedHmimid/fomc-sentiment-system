import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import importlib.util
_o = importlib.util.spec_from_file_location("orchestrator", Path(__file__).with_name("orchestrator.py"))
_orch = importlib.util.module_from_spec(_o)
_o.loader.exec_module(_orch)
analyze_event = _orch.analyze_event

NLP_BASE = os.getenv("NLP_URL", "http://localhost:8001")
URLS = {
    "nlp": NLP_BASE + "/analyze",
    "audio": os.getenv("AUDIO_URL", "http://localhost:8002") + "/analyze",
    "vision": os.getenv("VISION_URL", "http://localhost:8003") + "/analyze",
    "fusion": os.getenv("FUSION_URL", "http://localhost:8004") + "/fuse",
}
NLP_TEXT_URL = NLP_BASE + "/score_text"

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
EVENTS_FILE = ROOT / "data" / "events.json"
RESULTS_DIR = ROOT / "results"

app = FastAPI(title="gateway-service")


class AnalyzeRequest(BaseModel):
    event_id: str


class TextRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "gateway"}


@app.get("/")
def index():
    """Sert l'interface de test."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/events")
def list_events():
    """Liste des événements FOMC disponibles (pour le menu déroulant)."""
    if EVENTS_FILE.exists():
        return json.loads(EVENTS_FILE.read_text())
    return []


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """Analyse multimodale en direct d'un événement préparé."""
    async with httpx.AsyncClient(timeout=600.0) as client:
        return await analyze_event(req.event_id, client, URLS)


@app.post("/analyze_text")
async def analyze_text(req: TextRequest):
    """Score NLP (FinBERT) d'un texte libre — proxy vers le service nlp."""
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            r = await client.post(NLP_TEXT_URL, json={"text": req.text})
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        return {"event_id": "(texte libre)", "channel": "nlp", "score": 0.0,
                "label": "neutral", "raw": {}, "ok": False, "error": str(exc)}


@app.get("/results")
def results():
    """Scores et corrélations agrégés (lus depuis results/)."""
    scores_csv = RESULTS_DIR / "scores.csv"
    if not scores_csv.exists():
        return {"available": False}
    corr_csv = RESULTS_DIR / "correlations.csv"
    scores = list(csv.DictReader(scores_csv.open(encoding="utf-8")))
    corr = list(csv.reader(corr_csv.open(encoding="utf-8"))) if corr_csv.exists() else []
    correlations = {row[0]: row[1] for row in corr if len(row) == 2 and row[0]}
    return {"available": True, "scores": scores, "correlations": correlations,
            "has_image": (RESULTS_DIR / "correlations.png").exists()}


@app.get("/results/correlations.png")
def results_png():
    p = RESULTS_DIR / "correlations.png"
    if p.exists():
        return FileResponse(p, media_type="image/png")
    return JSONResponse({"error": "aucune image de résultats"}, status_code=404)
