FROM python:3.11-slim
WORKDIR /app
# system deps: ffmpeg for audio, libgl for opencv/deepface
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 && rm -rf /var/lib/apt/lists/*
COPY audio-service/requirements.txt ./req.txt
RUN pip install --no-cache-dir -r req.txt
COPY shared/ ./shared/
COPY audio-service/ ./audio-service/
ENV PYTHONPATH=/app
CMD ["uvicorn", "main:app", "--app-dir", "audio-service", "--host", "0.0.0.0", "--port", "8002"]
