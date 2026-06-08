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


def test_nlp_analyze_mocks_model(monkeypatch, tmp_path):
    main = _load_main("nlp-service")
    # mock the model scoring so no FinBERT download happens
    monkeypatch.setattr(main, "score_text", lambda text: (-0.7, "negative", {"prob": 0.7}))
    # write a fake transcript the service will read
    proc = tmp_path / "2023-03-22"
    proc.mkdir()
    (proc / "transcript.txt").write_text("rates will rise")
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)

    from fastapi.testclient import TestClient
    client = TestClient(main.app)
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/analyze", json={"event_id": "2023-03-22"})
    body = r.json()
    assert body["channel"] == "nlp"
    assert body["score"] == -0.7
    assert body["ok"] is True
