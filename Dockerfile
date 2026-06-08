# Single image shared by all five services. docker-compose passes SERVICE and PORT.
FROM python:3.11-slim

ARG SERVICE
ARG PORT
ENV SERVICE=${SERVICE} PORT=${PORT} PYTHONPATH=/app
WORKDIR /app

# ffmpeg (audio extraction) + libgl1 (opencv/deepface) cover every service's needs.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY ${SERVICE}/requirements.txt ./req.txt
RUN pip install --no-cache-dir -r req.txt

COPY shared/ ./shared/
COPY ${SERVICE}/ ./${SERVICE}/

# SERVICE/PORT are baked into env above; sh -c lets the CMD expand them.
CMD ["sh", "-c", "uvicorn main:app --app-dir ${SERVICE} --host 0.0.0.0 --port ${PORT}"]
