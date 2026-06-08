"""Vision channel service — scores FOMC face emotion with DeepFace.

Best-effort: if no usable face is found in any sampled frame the endpoint returns
ok=False rather than raising, so the gateway can still fuse the other channels.
The model is loaded lazily so importing this module stays cheap and unit tests can
monkeypatch `score_video` without pulling in deepface/opencv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from pydantic import BaseModel

from shared.schema import ChannelScore, valence_label

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# DeepFace emotions -> valence
_VALENCE = {"happy": 1.0, "surprise": 0.3, "neutral": 0.0,
            "sad": -1.0, "angry": -1.0, "fear": -0.7, "disgust": -0.7}


class NoFaceFound(Exception):
    pass


def _sample_frames(mp4_path, every_n_seconds=10, max_frames=12):
    import cv2
    cap = cv2.VideoCapture(str(mp4_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = int(fps * every_n_seconds)
    frames, idx = [], 0
    try:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % max(step, 1) == 0:
                frames.append(frame)
            idx += 1
    finally:
        cap.release()
    return frames


def score_video(mp4_path):
    """Return (valence in [-1, 1], label, raw). Raises NoFaceFound if no usable face."""
    from deepface import DeepFace
    frames = _sample_frames(mp4_path)
    vals = []
    for fr in frames:
        try:
            res = DeepFace.analyze(fr, actions=["emotion"], enforce_detection=True, silent=True)
            vals.append(_VALENCE.get(res[0]["dominant_emotion"], 0.0))
        except Exception:
            continue  # no face in this frame; skip
    if not vals:
        raise NoFaceFound("no face detected in any sampled frame")
    valence = max(-1.0, min(1.0, sum(vals) / len(vals)))
    return valence, valence_label(valence), {"frames_with_face": len(vals)}


app = FastAPI(title="vision-service")


class AnalyzeRequest(BaseModel):
    event_id: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "vision"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    mp4 = DATA_DIR / req.event_id / "video.mp4"
    try:
        if not mp4.exists():
            raise FileNotFoundError(f"missing {mp4}")
        valence, label, raw = score_video(mp4)
        return ChannelScore(event_id=req.event_id, channel="vision", score=valence,
                            label=label, raw=raw, ok=True).model_dump()
    except Exception as e:
        return ChannelScore(event_id=req.event_id, channel="vision", score=0.0,
                            label="neutral", raw={}, ok=False, error=str(e)).model_dump()
