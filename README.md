# FOMC Multimodal Sentiment — Distributed System

Analyzes FOMC press conferences via text (FinBERT), audio (Wav2Vec2), and face (DeepFace),
fuses the three signals, and compares each against S&P 500 moves.

## Architecture
5 FastAPI services over REST: gateway (8000) orchestrates nlp (8001), audio (8002),
vision (8003) in parallel, then fusion (8004) combines the scores. See `docker-compose.yml`.

## Honest scope
Sample is small (3–5 events). Correlations are **illustrative, not statistically significant**.
Add events by appending to `data/events.json` — no code changes.

## Run (local, Docker)
```bash
pip install -r data/requirements.txt
python data/prepare_data.py            # download + extract data
docker compose up --build
curl -X POST localhost:8000/analyze -H 'content-type: application/json' -d '{"event_id":"2023-03-22"}'
```

## Run (Colab, GPU, no Docker)
```bash
pip install -r nlp-service/requirements.txt -r audio-service/requirements.txt \
            -r vision-service/requirements.txt -r gateway-service/requirements.txt \
            -r fusion-service/requirements.txt -r data/requirements.txt
python data/prepare_data.py
python run_all.py &
# then run report.ipynb
```
GPU is auto-detected (`shared/device.py`); models use CUDA when available, CPU otherwise.

## Tests
```bash
pip install -r requirements-dev.txt
pytest -q
```
