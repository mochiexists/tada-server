#!/usr/bin/env python3
"""
TADA TTS Server - Local FastAPI wrapper for HumeAI TADA model.
Runs on MPS (Apple Silicon) or CUDA.
"""

import io
import os
import logging
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

from tada.modules.encoder import Encoder, EncoderOutput
from tada.modules.tada import TadaForCausalLM

# MPS fix: patch lm_head_forward to avoid CPU/MPS device mismatch
def _fixed_lm_head_forward(self, hidden_states):
    return self.lm_head(hidden_states)

TadaForCausalLM._lm_head_forward = _fixed_lm_head_forward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_ID = os.getenv("TADA_MODEL", "HumeAI/tada-1b")  # or tada-3b-ml for better quality
DEVICE = os.getenv("TADA_DEVICE", "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE != "cpu" else torch.float32
PORT = int(os.getenv("TADA_PORT", "18793"))
PROMPT_CACHE_DIR = Path(os.getenv("TADA_CACHE_DIR", "~/.cache/tada-server")).expanduser()
PROMPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- App ---
app = FastAPI(title="TADA TTS Server", version="0.1.0")

# --- Global model state ---
encoder: Optional[Encoder] = None
model: Optional[TadaForCausalLM] = None
default_prompt: Optional[EncoderOutput] = None


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None  # voice preset name (future: support multiple)


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


def get_default_prompt_path() -> Path:
    return PROMPT_CACHE_DIR / "default_prompt.pt"


def load_or_create_default_prompt() -> EncoderOutput:
    """Load cached prompt or create from LJSpeech sample."""
    cache_path = get_default_prompt_path()
    
    if cache_path.exists():
        logger.info(f"Loading cached prompt from {cache_path}")
        return EncoderOutput.load(str(cache_path), device=DEVICE)
    
    logger.info("Creating default prompt from LJSpeech sample...")
    
    # Download reference audio
    ref_path = hf_hub_download(
        repo_id="HumeAI/tada",
        repo_type="space",
        filename="samples/en/ljspeech.wav"
    )
    
    ref_wav, ref_sr = torchaudio.load(ref_path)
    ref_wav = ref_wav.to(DEVICE)
    
    # Reference text for LJSpeech sample
    prompt_text = "The examination and testimony of the experts enabled the commission to conclude that five shots may have been fired."
    
    prompt = encoder(ref_wav, text=[prompt_text], sample_rate=ref_sr)
    
    # Cache for future runs
    prompt.save(str(cache_path))
    logger.info(f"Cached prompt to {cache_path}")
    
    return prompt


@app.on_event("startup")
async def startup():
    global encoder, model, default_prompt
    
    logger.info(f"Loading TADA model: {MODEL_ID}")
    logger.info(f"Device: {DEVICE}, dtype: {DTYPE}")
    
    # Load encoder separately (saves ~2.5GB VRAM)
    encoder = Encoder.from_pretrained(
        "HumeAI/tada-codec",
        subfolder="encoder"
    ).to(DEVICE)
    
    # Load model
    model = TadaForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE
    ).to(DEVICE)
    
    # Pre-load default voice prompt
    default_prompt = load_or_create_default_prompt()
    
    logger.info("TADA server ready!")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if model is not None else "loading",
        model=MODEL_ID,
        device=DEVICE
    )


@app.post("/tts")
async def tts(req: TTSRequest):
    """Generate speech from text."""
    if model is None or default_prompt is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if len(req.text) > 2000:
        raise HTTPException(status_code=400, detail="Text too long (max 2000 chars)")
    
    try:
        with torch.no_grad():
            output = model.generate(
                prompt=default_prompt,
                text=req.text,
            )
        
        # Convert to WAV (ensure 2D tensor: channels x samples)
        audio = output.audio[0].cpu()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio, 24000, format="wav")
        buffer.seek(0)
        
        return Response(
            content=buffer.read(),
            media_type="audio/wav",
            headers={"X-TADA-Model": MODEL_ID}
        )
    
    except Exception as e:
        logger.exception("TTS generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/opus")
async def tts_opus(req: TTSRequest):
    """Generate speech and return as Opus (for Telegram voice notes)."""
    if model is None or default_prompt is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        with torch.no_grad():
            output = model.generate(
                prompt=default_prompt,
                text=req.text,
            )
        
        # Save as WAV first, then convert to Opus via ffmpeg
        import subprocess
        import tempfile
        
        audio = output.audio[0].cpu()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            torchaudio.save(wav_file.name, audio, 24000, format="wav")
            wav_path = wav_file.name
        
        try:
            # Convert to Opus
            opus_buffer = io.BytesIO()
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libopus", "-b:a", "64k", "-f", "opus", "-"],
                capture_output=True,
                check=True
            )
            opus_buffer.write(result.stdout)
            opus_buffer.seek(0)
            
            return Response(
                content=opus_buffer.read(),
                media_type="audio/opus",
                headers={"X-TADA-Model": MODEL_ID}
            )
        finally:
            os.unlink(wav_path)
    
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
        raise HTTPException(status_code=500, detail="Audio conversion failed")
    except Exception as e:
        logger.exception("TTS generation failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT)
