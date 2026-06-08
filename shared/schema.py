from typing import Literal, Optional
from pydantic import BaseModel, Field

Channel = Literal["nlp", "audio", "vision"]


def valence_label(score: float) -> str:
    if score >= 0.15:
        return "positive"
    if score <= -0.15:
        return "negative"
    return "neutral"


class ChannelScore(BaseModel):
    event_id: str
    channel: Channel
    score: float = Field(ge=-1.0, le=1.0)
    label: str
    raw: dict = Field(default_factory=dict)
    ok: bool = True
    error: Optional[str] = None


class FusionResult(BaseModel):
    event_id: str
    combined_score: float = Field(ge=-1.0, le=1.0)
    label: str
    channels: list[ChannelScore]
    weights_used: dict[str, float]
    channels_used: list[str]
