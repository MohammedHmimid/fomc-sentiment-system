import pytest
from pydantic import ValidationError
from shared.schema import ChannelScore, valence_label


def test_valence_label_thresholds():
    assert valence_label(-0.5) == "negative"
    assert valence_label(0.0) == "neutral"
    assert valence_label(0.5) == "positive"
    assert valence_label(0.15) == "positive"
    assert valence_label(-0.15) == "negative"
    assert valence_label(0.1) == "neutral"


def test_channel_score_valid():
    cs = ChannelScore(event_id="2023-03-22", channel="nlp", score=-0.4,
                      label="negative", raw={"p": 0.4}, ok=True)
    assert cs.score == -0.4
    assert cs.ok is True


def test_channel_score_rejects_out_of_range():
    with pytest.raises(ValidationError):
        ChannelScore(event_id="x", channel="nlp", score=2.0, label="positive", raw={}, ok=True)


def test_failed_channel_has_error():
    cs = ChannelScore(event_id="x", channel="vision", score=0.0, label="neutral",
                      raw={}, ok=False, error="no face found")
    assert cs.ok is False
    assert cs.error == "no face found"
