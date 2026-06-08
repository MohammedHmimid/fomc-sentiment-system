import importlib.util
from pathlib import Path

from shared.schema import ChannelScore

ROOT = Path(__file__).resolve().parents[1]


def _load_fusion():
    spec = importlib.util.spec_from_file_location("fusion", ROOT / "fusion-service" / "fusion.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cs(channel, score, ok=True, error=None):
    return ChannelScore(event_id="e", channel=channel, score=score,
                        label="x", raw={}, ok=ok, error=error)


def test_combine_weighted_average():
    fusion = _load_fusion()
    chans = [_cs("nlp", 1.0), _cs("audio", 0.0), _cs("vision", -1.0)]
    res = fusion.combine_scores("e", chans, weights={"nlp": 0.5, "audio": 0.3, "vision": 0.2})
    # 0.5*1 + 0.3*0 + 0.2*-1 = 0.3 ; total weight 1.0
    assert abs(res.combined_score - 0.3) < 1e-9
    assert res.channels_used == ["nlp", "audio", "vision"]


def test_failed_channel_is_excluded_and_weights_renormalize():
    fusion = _load_fusion()
    chans = [_cs("nlp", 1.0), _cs("vision", -1.0, ok=False, error="no face")]
    res = fusion.combine_scores("e", chans, weights={"nlp": 0.5, "audio": 0.3, "vision": 0.2})
    # only nlp usable -> combined == nlp score
    assert abs(res.combined_score - 1.0) < 1e-9
    assert res.channels_used == ["nlp"]


def test_all_failed_returns_neutral():
    fusion = _load_fusion()
    chans = [_cs("nlp", 0.5, ok=False, error="boom")]
    res = fusion.combine_scores("e", chans)
    assert res.combined_score == 0.0
    assert res.label == "neutral"
    assert res.channels_used == []
