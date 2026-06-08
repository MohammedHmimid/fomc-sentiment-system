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


def test_audio_analyze_mocks_model(monkeypatch, tmp_path):
    main = _load_main("audio-service")
    monkeypatch.setattr(main, "score_audio", lambda wav: (0.4, "positive", {"emotion": "hap"}))
    proc = tmp_path / "2023-03-22"; proc.mkdir()
    (proc / "audio.wav").write_bytes(b"RIFF....")  # presence only; model mocked
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)

    from fastapi.testclient import TestClient
    client = TestClient(main.app)
    r = client.post("/analyze", json={"event_id": "2023-03-22"})
    body = r.json()
    assert body["channel"] == "audio"
    assert body["score"] == 0.4
    assert body["ok"] is True


def test_vision_analyze_success(monkeypatch, tmp_path):
    main = _load_main("vision-service")
    monkeypatch.setattr(main, "score_video", lambda mp4: (0.2, "positive", {"frames": 5}))
    proc = tmp_path / "2023-03-22"; proc.mkdir()
    (proc / "video.mp4").write_bytes(b"\x00")
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    from fastapi.testclient import TestClient
    r = TestClient(main.app).post("/analyze", json={"event_id": "2023-03-22"})
    assert r.json()["score"] == 0.2 and r.json()["ok"] is True


def test_vision_degrades_gracefully_on_no_face(monkeypatch, tmp_path):
    main = _load_main("vision-service")
    def _boom(mp4):
        raise main.NoFaceFound("no face in any frame")
    monkeypatch.setattr(main, "score_video", _boom)
    proc = tmp_path / "2023-03-22"; proc.mkdir()
    (proc / "video.mp4").write_bytes(b"\x00")
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    from fastapi.testclient import TestClient
    body = TestClient(main.app).post("/analyze", json={"event_id": "2023-03-22"}).json()
    assert body["ok"] is False
    assert "no face" in body["error"]


def test_nlp_score_text_endpoint(monkeypatch):
    main = _load_main("nlp-service")
    monkeypatch.setattr(main, "score_text", lambda t: (0.6, "positive", {"n_chunks": 1}))
    body = TestClient(main.app).post("/score_text", json={"text": "rates steady"}).json()
    assert body["channel"] == "nlp"
    assert body["score"] == 0.6
    assert body["ok"] is True


def test_gateway_events_and_results_endpoints(monkeypatch, tmp_path):
    import json
    main = _load_main("gateway-service")
    events_file = tmp_path / "events.json"
    events_file.write_text(json.dumps([{"id": "2023-03-22"}, {"id": "2023-05-03"}]))
    monkeypatch.setattr(main, "EVENTS_FILE", events_file)
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path / "results")  # empty -> unavailable

    client = TestClient(main.app)
    assert client.get("/health").json()["service"] == "gateway"
    assert [e["id"] for e in client.get("/events").json()] == ["2023-03-22", "2023-05-03"]
    assert client.get("/results").json()["available"] is False
