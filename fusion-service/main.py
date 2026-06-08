import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from pydantic import BaseModel

from shared.schema import ChannelScore

import importlib.util
_f = importlib.util.spec_from_file_location("fusion", Path(__file__).with_name("fusion.py"))
_fusion = importlib.util.module_from_spec(_f)
_f.loader.exec_module(_fusion)
combine_scores = _fusion.combine_scores

app = FastAPI(title="fusion-service")


class FuseRequest(BaseModel):
    event_id: str
    channels: list[ChannelScore]
    weights: dict[str, float] | None = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "fusion"}


@app.post("/fuse")
def fuse(req: FuseRequest):
    result = combine_scores(req.event_id, req.channels, req.weights)
    return result.model_dump()
