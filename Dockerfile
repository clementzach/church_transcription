FROM python:3.11-slim

WORKDIR /app

# ffmpeg is required by pydub and faster-whisper audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

CMD ["gunicorn", "--worker-class", "gevent", "-w", "1", \
     "--bind", "0.0.0.0:5001", \
     "--timeout", "120", \
     "app:app"]
