# TADA TTS Server

Local FastAPI server for [HumeAI TADA](https://github.com/HumeAI/tada) text-to-speech with voice cloning support.

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

### Voice Cloning
Clone a voice from a reference audio sample:

```bash
curl -X POST http://127.0.0.1:18793/tts/clone \
  -F "audio=@samples/grimes-10s.wav" \
  -F "transcript=I have a proposition for the communists. So typically most of the communists I know are not big fans of AI. But if you think about it." \
  -F "text=Hello, I am speaking with a cloned voice." \
  -F "format=wav" \
  --output cloned.wav
```

Parameters:
- `audio` - Reference audio file (WAV, MP3, etc.) - 5-10 seconds recommended
- `transcript` - Text transcript of the reference audio
- `text` - Text to synthesize with cloned voice
- `format` - Output format: `wav` (default) or `opus`

## Samples

Included samples for testing voice cloning:

- `samples/grimes-communism.mp4` - Source video (46s)
- `samples/grimes-10s.wav` - Extracted 10s clip for voice prompting

## Test Client
```bash
python test_client.py "Hello, this is a test!"
```

## Notes

- Voice cloning works best with 5-10 second reference clips
- Longer clips may cause memory issues on 16GB machines
- The 3B model (`HumeAI/tada-3b-ml`) produces better quality but requires more VRAM

## License

MIT - wrapper code only. TADA model has its own license from HumeAI.
