import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.schema import valence_label

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
    """Return (valence in [-1,1], label, raw). Raises NoFaceFound if no usable face."""
    from deepface import DeepFace
    frames = _sample_frames(mp4_path)
    vals = []
    for fr in frames:
        try:
            res = DeepFace.analyze(fr, actions=["emotion"], enforce_detection=True,
                                   silent=True)
            emo = res[0]["dominant_emotion"]
            vals.append(_VALENCE.get(emo, 0.0))
        except Exception:
            continue  # no face in this frame; skip
    if not vals:
        raise NoFaceFound("no face detected in any sampled frame")
    valence = max(-1.0, min(1.0, sum(vals) / len(vals)))
    label = valence_label(valence)
    return valence, label, {"frames_with_face": len(vals)}
