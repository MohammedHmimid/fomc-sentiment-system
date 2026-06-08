from shared.schema import ChannelScore, FusionResult, valence_label

DEFAULT_WEIGHTS = {"nlp": 0.5, "audio": 0.3, "vision": 0.2}


def combine_scores(event_id, channels, weights=None):
    if weights is None:
        weights = DEFAULT_WEIGHTS
    usable = [c for c in channels if c.ok]
    total_w = sum(weights.get(c.channel, 0.0) for c in usable)

    if not usable or total_w == 0.0:
        return FusionResult(
            event_id=event_id, combined_score=0.0, label="neutral",
            channels=channels, weights_used={}, channels_used=[],
        )

    combined = sum(weights.get(c.channel, 0.0) * c.score for c in usable) / total_w
    combined = max(-1.0, min(1.0, combined))
    used = [c.channel for c in usable]
    return FusionResult(
        event_id=event_id, combined_score=combined, label=valence_label(combined),
        channels=channels,
        weights_used={c.channel: weights.get(c.channel, 0.0) for c in usable},
        channels_used=used,
    )
