import importlib.util
from pathlib import Path
import asyncio

ROOT = Path(__file__).resolve().parents[1]


def _load_orch():
    spec = importlib.util.spec_from_file_location(
        "orchestrator", ROOT / "gateway-service" / "orchestrator.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class FakeResponse:
    def __init__(self, payload): self._p = payload
    def json(self): return self._p
    def raise_for_status(self): pass


class FakeClient:
    """Returns canned per-URL responses; records calls."""
    def __init__(self, mapping): self.mapping = mapping; self.calls = []
    async def post(self, url, json):
        self.calls.append(url)
        return FakeResponse(self.mapping[url](json))


def test_analyze_event_fans_out_and_fuses():
    orch = _load_orch()
    urls = {"nlp": "http://nlp/analyze", "audio": "http://audio/analyze",
            "vision": "http://vision/analyze", "fusion": "http://fusion/fuse"}

    def chan(channel, score, ok=True, error=None):
        return lambda body: {"event_id": body["event_id"], "channel": channel,
                             "score": score, "label": "x", "raw": {}, "ok": ok, "error": error}

    def fusion(body):
        # echo what fusion would compute: average of usable scores
        usable = [c for c in body["channels"] if c["ok"]]
        avg = sum(c["score"] for c in usable) / len(usable)
        return {"event_id": body["event_id"], "combined_score": avg, "label": "x",
                "channels": body["channels"], "weights_used": {}, "channels_used": [c["channel"] for c in usable]}

    client = FakeClient({
        urls["nlp"]: chan("nlp", 1.0),
        urls["audio"]: chan("audio", 0.0),
        urls["vision"]: chan("vision", -1.0, ok=False, error="no face"),
        urls["fusion"]: fusion,
    })
    result = asyncio.run(orch.analyze_event("2023-03-22", client, urls))
    assert result["channels_used"] == ["nlp", "audio"]
    assert abs(result["combined_score"] - 0.5) < 1e-9
    # all three channel services + fusion called
    assert len(client.calls) == 4
