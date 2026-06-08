import importlib.util
from pathlib import Path
import asyncio

ROOT = Path(__file__).resolve().parents[1]


def _load_orch():
    spec = importlib.util.spec_from_file_location(
        "orchestrator", ROOT / "gateway-service" / "orchestrator.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _load_fusion():
    spec = importlib.util.spec_from_file_location(
        "fusion", ROOT / "fusion-service" / "fusion.py")
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
    fusion_mod = _load_fusion()
    urls = {"nlp": "http://nlp/analyze", "audio": "http://audio/analyze",
            "vision": "http://vision/analyze", "fusion": "http://fusion/fuse"}

    def chan(channel, score, ok=True, error=None):
        return lambda body: {"event_id": body["event_id"], "channel": channel,
                             "score": score, "label": "x", "raw": {}, "ok": ok, "error": error}

    def fusion(body):
        # delegate to real combine_scores for accurate math
        from shared.schema import ChannelScore
        chans = [ChannelScore(**c) for c in body["channels"]]
        result = fusion_mod.combine_scores(body["event_id"], chans)
        return result.model_dump()

    client = FakeClient({
        urls["nlp"]: chan("nlp", 1.0),
        urls["audio"]: chan("audio", 0.0),
        urls["vision"]: chan("vision", -1.0, ok=False, error="no face"),
        urls["fusion"]: fusion,
    })
    result = asyncio.run(orch.analyze_event("2023-03-22", client, urls))
    # real weighted result: (0.5*1.0 + 0.3*0.0) / (0.5+0.3) = 0.625
    assert abs(result["combined_score"] - 0.625) < 1e-9
    assert result["channels_used"] == ["nlp", "audio"]
    # all three channel services + fusion called
    assert len(client.calls) == 4


class RaisingClient:
    """Raises for the vision URL; returns canned responses for nlp/audio/fusion."""
    def __init__(self, mapping, raise_url):
        self.mapping = mapping
        self.raise_url = raise_url
        self.calls = []

    async def post(self, url, json):
        self.calls.append(url)
        if url == self.raise_url:
            raise RuntimeError("connect error")
        return FakeResponse(self.mapping[url](json))


def test_channel_failure_is_tolerated():
    orch = _load_orch()
    fusion_mod = _load_fusion()
    urls = {"nlp": "http://nlp/analyze", "audio": "http://audio/analyze",
            "vision": "http://vision/analyze", "fusion": "http://fusion/fuse"}

    def chan(channel, score):
        return lambda body: {"event_id": body["event_id"], "channel": channel,
                             "score": score, "label": "x", "raw": {}, "ok": True, "error": None}

    def fusion(body):
        from shared.schema import ChannelScore
        chans = [ChannelScore(**c) for c in body["channels"]]
        result = fusion_mod.combine_scores(body["event_id"], chans)
        return result.model_dump()

    client = RaisingClient(
        mapping={
            urls["nlp"]: chan("nlp", 1.0),
            urls["audio"]: chan("audio", 0.0),
            urls["fusion"]: fusion,
        },
        raise_url=urls["vision"],
    )
    # should not raise
    result = asyncio.run(orch.analyze_event("2023-03-22", client, urls))
    assert result is not None
    # vision channel should come back ok=False
    vision_chan = next(c for c in result["channels"] if c["channel"] == "vision")
    assert vision_chan["ok"] is False
    assert "connect error" in vision_chan["error"]
    # all 4 endpoints were attempted
    assert len(client.calls) == 4


def test_fusion_failure_returns_predictable_shape():
    orch = _load_orch()
    urls = {"nlp": "http://nlp/analyze", "audio": "http://audio/analyze",
            "vision": "http://vision/analyze", "fusion": "http://fusion/fuse"}

    def chan(channel, score):
        return lambda body: {"event_id": body["event_id"], "channel": channel,
                             "score": score, "label": "x", "raw": {}, "ok": True, "error": None}

    class FusionRaisingClient:
        """Normal responses for channel URLs; raises RuntimeError for the fusion URL."""
        def __init__(self, mapping, fusion_url):
            self.mapping = mapping
            self.fusion_url = fusion_url
            self.calls = []

        async def post(self, url, json):
            self.calls.append(url)
            if url == self.fusion_url:
                raise RuntimeError("fusion down")
            return FakeResponse(self.mapping[url](json))

    client = FusionRaisingClient(
        mapping={
            urls["nlp"]: chan("nlp", 0.5),
            urls["audio"]: chan("audio", 0.3),
            urls["vision"]: chan("vision", -0.2),
        },
        fusion_url=urls["fusion"],
    )

    # must not raise
    result = asyncio.run(orch.analyze_event("e", client, urls))
    assert result["combined_score"] == 0.0
    assert result["channels_used"] == []
    assert "error" in result
    assert "fusion down" in result["error"]
