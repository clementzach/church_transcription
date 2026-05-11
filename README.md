# Church Transcription

Live transcription and translation web app for church services. A broadcaster records audio; Gladia transcribes and translates it in real time; Gemini TTS converts translations to speech; remote listeners hear audio and see captions in their chosen language (Spanish, Haitian Creole, Portuguese, Mandarin, French, or Norwegian).

When a fine-tuned local Whisper model is configured, selecting Haitian Creole as the speaker language bypasses Gladia entirely and uses the local model for transcription and Gemini for translation.

## Requirements

- [Gladia](https://gladia.io) API key
- [Google AI](https://aistudio.google.com) API key (Gemini TTS + optional translation)
- [OpenAI](https://platform.openai.com) API key (only if `TTS_PROVIDER=openai`)
- Docker + Docker Compose
- NVIDIA GPU + drivers (for local Whisper; optional otherwise)

## Local development

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in API keys
python app.py          # http://localhost:5001
```

## Deployment on Google Compute Engine (NVIDIA L4)

### 1. Open firewall ports

```bash
gcloud compute firewall-rules create allow-http  --allow tcp:80  --target-tags http-server
gcloud compute firewall-rules create allow-https --allow tcp:443 --target-tags http-server
```

### 2. Install Docker

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER && newgrp docker
```

### 3. Install NVIDIA container toolkit

Required to pass the GPU through to Docker containers.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 nvidia-smi
```

### 4. Clone the repo

```bash
git clone <your-repo-url> ~/church_transcription
cd ~/church_transcription
```

### 5. Convert and install the local Whisper model (optional)

Skip this section if you are not using the local Haitian Creole model.

The fine-tuned model must be converted from HuggingFace format to CTranslate2 format before faster-whisper can load it.

```bash
pip install ctranslate2

ct2-transformers-converter \
  --model /path/to/your/hf-checkpoint \
  --output_dir ~/church_transcription/models/haitian_creole \
  --quantization float16 \
  --force

# Verify
ls ~/church_transcription/models/haitian_creole
# Expected: config.json  model.bin  vocabulary.json
```

### 6. Configure environment

```bash
cp .env.example .env
nano .env
```

Key variables:

| Variable | Description |
|---|---|
| `GLADIA_API_KEY` | Required for all non-HT speakers |
| `GOOGLE_API_KEY` | Required for Gemini TTS and HT translation |
| `OPENAI_API_KEY` | Required only if `TTS_PROVIDER=openai` |
| `TTS_PROVIDER` | `google` (default) or `openai` |
| `LOCAL_HAITIAN_PATH` | Container path to the converted Whisper model, e.g. `/app/models/haitian_creole`. Leave blank to use Gladia for HT. |

### 7. Get SSL certificates

```bash
sudo apt-get install -y certbot
sudo certbot certonly --standalone -d translation.zacharyclement.com

mkdir -p ~/church_transcription/nginx/certs
sudo cp /etc/letsencrypt/live/translation.zacharyclement.com/fullchain.pem ~/church_transcription/nginx/certs/
sudo cp /etc/letsencrypt/live/translation.zacharyclement.com/privkey.pem  ~/church_transcription/nginx/certs/
sudo chown $USER ~/church_transcription/nginx/certs/*.pem
```

### 8. Build and start

```bash
cd ~/church_transcription
docker compose build
docker compose up -d

# Follow logs
docker compose logs -f app
docker compose logs -f nginx
```

### 9. Certificate renewal

Certs expire every 90 days. Add a monthly cron job:

```bash
sudo crontab -e
```

```
0 3 1 * * cd /home/<your-user>/church_transcription && docker compose stop nginx && certbot renew --quiet && cp /etc/letsencrypt/live/translation.zacharyclement.com/fullchain.pem nginx/certs/ && cp /etc/letsencrypt/live/translation.zacharyclement.com/privkey.pem nginx/certs/ && docker compose start nginx
```

## Updating the app

```bash
cd ~/church_transcription
git pull
docker compose build
docker compose up -d
```

## Useful commands

```bash
docker compose restart app   # restart app after config change
docker compose down          # stop everything
docker compose logs -f app   # stream app logs
```

## Architecture

```
Browser (broadcaster)
  └─ WebSocket /stream ──────► app.py ──► Gladia WebSocket (all languages except local HT)
  └─ WebSocket /stream-local ► app.py ──► local Whisper (HT speaker, when LOCAL_HAITIAN_PATH set)
                                    │
                                    └─ Gemini text model (translation, HT path only)
                                    │
                               Queue per language (maxsize=1, drops stale)
                                    │
                               _tts_worker ──► Gemini / OpenAI TTS
                                    │
                          WebSocket /listen-stream ──► Browser (listener)
```

Sessions expire automatically after 2 hours. All session state is in-process; a single gunicorn worker is required.
