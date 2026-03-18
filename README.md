# TADA TTS Server

Local FastAPI server for [HumeAI TADA](https://github.com/HumeAI/tada) text-to-speech.

## Setup

```bash
# Create venv
python3 -m venv venv
source venv/bin/activate

# Install deps
pip install -r requirements.txt

# Run server (MPS by default on Apple Silicon)
python server.py
```

Server runs on `http://127.0.0.1:18793`

## Environment Variables

- `TADA_MODEL` - Model ID (default: `HumeAI/tada-1b`, use `HumeAI/tada-3b-ml` for better quality)
- `TADA_DEVICE` - Device: `mps`, `cuda`, or `cpu` (auto-detected)
- `TADA_PORT` - Server port (default: 18793)
- `TADA_CACHE_DIR` - Prompt cache directory (default: `~/.cache/tada-server`)

## API

### Health Check
```bash
curl http://127.0.0.1:18793/health
```

### Generate Speech (WAV)
```bash
curl -X POST http://127.0.0.1:18793/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!"}' \
  --output speech.wav
```

### Generate Speech (Opus for Telegram)
```bash
curl -X POST http://127.0.0.1:18793/tts/opus \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!"}' \
  --output speech.opus
```

## Test Client
```bash
python test_client.py "Hello, this is a test!"
```

## OpenClaw Integration

Add TADA as a TTS provider in OpenClaw's `tts-core.ts` - see parent project for integration guide.
