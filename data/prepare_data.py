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


def _step(name, fn):
    """Run one preparation step; log and swallow its error so the others still run."""
    try:
        fn()
    except Exception as e:
        print(f"  [skip] {name}: {e}")


def prepare_event(ev):
    out = PROCESSED / ev["id"]
    out.mkdir(parents=True, exist_ok=True)
    video = out / "video.mp4"
    audio = out / "audio.wav"

    def video_step():
        # yt-dlp is invoked via the module so the up-to-date pip version is used.
        if not video.exists():
            _run([sys.executable, "-m", "yt_dlp",
                  "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                  "--merge-output-format", "mp4", "-o", str(video), ev["video_url"]])

    def audio_step():
        # 16kHz mono — what wav2vec2 expects. Needs the video.
        if not audio.exists():
            if not video.exists():
                raise FileNotFoundError("no video.mp4 to extract audio from")
            _run(["ffmpeg", "-y", "-i", str(video), "-ac", "1", "-ar", "16000", str(audio)])

    def transcript_step():
        tr = out / "transcript.txt"
        if not tr.exists():
            _download_transcript(ev["transcript_url"], tr)

    def market_step():
        # day-of to day-after (idempotent: skip if already fetched)
        if not (out / "market_signal.json").exists():
            day = datetime.fromisoformat(ev["fomc_datetime"]).date()
            df = fetch_market_window(start=str(day), end=str(day + timedelta(days=3)))
            df.to_csv(out / "market.csv")
            signal = compute_market_signal(df)
            (out / "market_signal.json").write_text(json.dumps({"signal_pct": signal}))
            print(f"  market signal: {signal}")

    # Each step is independent: a video failure must not block transcript/market.
    _step("video", video_step)
    _step("audio", audio_step)
    _step("transcript", transcript_step)
    _step("market", market_step)


def _download_transcript(url, dest):
    import urllib.request
    raw = dest.with_suffix(".pdf" if url.split("?")[0].endswith(".pdf") else ".html")
    # federalreserve.gov rejects the default Python user-agent (403); send a browser one.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        raw.write_bytes(resp.read())
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
