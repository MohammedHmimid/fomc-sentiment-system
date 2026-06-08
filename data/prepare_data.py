import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROCESSED = HERE / "processed"
sys.path.insert(0, str(HERE))
from market import fetch_market_window, compute_market_signal  # noqa: E402


def _run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def prepare_event(ev):
    out = PROCESSED / ev["id"]
    out.mkdir(parents=True, exist_ok=True)
    video = out / "video.mp4"
    audio = out / "audio.wav"

    # 1. video via yt-dlp
    if not video.exists():
        _run(["yt-dlp", "-f", "mp4", "-o", str(video), ev["video_url"]])
    # 2. audio via ffmpeg (16kHz mono — what wav2vec2 expects)
    if not audio.exists():
        _run(["ffmpeg", "-y", "-i", str(video), "-ac", "1", "-ar", "16000", str(audio)])
    # 3. transcript
    tr = out / "transcript.txt"
    if not tr.exists():
        _download_transcript(ev["transcript_url"], tr)
    # 4. market window: day-of to day-after
    day = datetime.fromisoformat(ev["fomc_datetime"]).date()
    df = fetch_market_window(start=str(day), end=str(day + timedelta(days=3)))
    df.to_csv(out / "market.csv")
    signal = compute_market_signal(df)
    (out / "market_signal.json").write_text(json.dumps({"signal_pct": signal}))
    print(f"  market signal: {signal}")


def _download_transcript(url, dest):
    import urllib.request
    raw = dest.with_suffix(".pdf" if url.endswith(".pdf") else ".html")
    urllib.request.urlretrieve(url, raw)
    if raw.suffix == ".pdf":
        from pypdf import PdfReader
        text = "\n".join(p.extract_text() or "" for p in PdfReader(str(raw)).pages)
    else:
        from bs4 import BeautifulSoup
        text = BeautifulSoup(raw.read_text(encoding="utf-8", errors="ignore"), "html.parser").get_text()
    dest.write_text(text, encoding="utf-8")


def main():
    events = json.loads((HERE / "events.json").read_text())
    for ev in events:
        print(f"=== {ev['id']} ===")
        try:
            prepare_event(ev)
        except Exception as e:
            print(f"  FAILED {ev['id']}: {e}")


if __name__ == "__main__":
    main()
