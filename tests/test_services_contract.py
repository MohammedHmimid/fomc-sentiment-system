import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


def _load_main(service_dir):
    spec = importlib.util.spec_from_file_location(
        f"{service_dir}_main", ROOT / service_dir / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fusion_health_and_combine():
    app = _load_main("fusion-service").app
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"

    payload = {
        "event_id": "e",
        "channels": [
            {"event_id": "e", "channel": "nlp", "score": 1.0, "label": "positive", "raw": {}, "ok": True},
            {"event_id": "e", "channel": "audio", "score": -1.0, "label": "negative", "raw": {}, "ok": True},
        ],
    }
    r = client.post("/fuse", json=payload)
    body = r.json()
    assert body["channels_used"] == ["nlp", "audio"]
    assert "combined_score" in body
