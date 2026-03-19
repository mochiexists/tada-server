#!/usr/bin/env python3
"""
TADA TTS Server - Local FastAPI wrapper for HumeAI TADA model.
Runs on MPS (Apple Silicon) or CUDA.
"""

import io
import os
import logging
import tempfile
import subprocess
import time
from pathlib import Path
from typing import Optional

import psutil
import torch
import torchaudio
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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


def log_memory(label):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss / 1024**3
    msg = f"[MEM] {label}: RSS={rss:.2f}GB"
    if DEVICE == "mps":
        mps_alloc = torch.mps.current_allocated_memory() / 1024**3
        mps_driver = torch.mps.driver_allocated_memory() / 1024**3
        msg += f", MPS_alloc={mps_alloc:.2f}GB, MPS_driver={mps_driver:.2f}GB"
    elif DEVICE == "cuda":
        cuda_alloc = torch.cuda.memory_allocated() / 1024**3
        msg += f", CUDA_alloc={cuda_alloc:.2f}GB"
    logger.info(msg)


@app.on_event("startup")
async def startup():
    global encoder, model, default_prompt

    logger.info(f"Loading TADA model: {MODEL_ID}")
    logger.info(f"Device: {DEVICE}, dtype: {DTYPE}")
    logger.info(f"System RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    log_memory("baseline")

    # Load encoder separately (saves ~2.5GB VRAM)
    t0 = time.time()
    encoder = Encoder.from_pretrained(
        "HumeAI/tada-codec",
        subfolder="encoder"
    ).to(DEVICE)
    logger.info(f"Encoder loaded in {time.time()-t0:.1f}s")
    log_memory("after encoder")

    # Load model
    t0 = time.time()
    model = TadaForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE
    ).to(DEVICE)
    logger.info(f"Model loaded in {time.time()-t0:.1f}s")
    log_memory("after model")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Pre-load default voice prompt
    t0 = time.time()
    default_prompt = load_or_create_default_prompt()
    logger.info(f"Default prompt loaded in {time.time()-t0:.1f}s")
    log_memory("after prompt")

    logger.info("TADA server ready!")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if model is not None else "loading",
        model=MODEL_ID,
        device=DEVICE
    )


@app.get("/stats")
async def stats():
    """Memory usage and system stats."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info()

    result = {
        "model": MODEL_ID,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "process": {
            "rss_mb": round(mem.rss / 1024 / 1024, 1),
            "vms_mb": round(mem.vms / 1024 / 1024, 1),
        },
        "system": {
            "total_ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
            "available_ram_gb": round(psutil.virtual_memory().available / 1024**3, 1),
            "ram_percent": psutil.virtual_memory().percent,
        },
    }

    if DEVICE == "mps":
        result["mps"] = {
            "allocated_mb": round(torch.mps.current_allocated_memory() / 1024**2, 1),
            "driver_mb": round(torch.mps.driver_allocated_memory() / 1024**2, 1),
        }
    elif DEVICE == "cuda":
        result["cuda"] = {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        }

    if model is not None:
        total_params = sum(p.numel() for p in model.parameters())
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        result["model_params"] = {
            "total": total_params,
            "memory_mb": round(param_bytes / 1024**2, 1),
        }

    return result


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


@app.post("/tts/clone")
async def tts_clone(
    audio: UploadFile = File(..., description="Reference audio file (WAV/MP3/etc)"),
    text: str = Form(..., description="Text to synthesize"),
    transcript: str = Form(..., description="Transcript of the reference audio"),
    format: str = Form("wav", description="Output format: wav or opus")
):
    """Generate speech cloning a voice from reference audio."""
    if model is None or encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if not transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty")
    
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="Text too long (max 2000 chars)")
    
    try:
        # Save uploaded audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Load and resample reference audio
            ref_wav, ref_sr = torchaudio.load(tmp_path)
            if ref_sr != 24000:
                ref_wav = torchaudio.functional.resample(ref_wav, ref_sr, 24000)
                ref_sr = 24000
            ref_wav = ref_wav.to(DEVICE)
            
            # Create voice prompt from reference
            voice_prompt = encoder(ref_wav, text=[transcript], sample_rate=ref_sr)
            
            # Generate with cloned voice
            with torch.no_grad():
                output = model.generate(prompt=voice_prompt, text=text)
            
            # Convert to output format
            audio_out = output.audio[0].cpu()
            if audio_out.dim() == 1:
                audio_out = audio_out.unsqueeze(0)
            
            if format == "opus":
                # Save as WAV then convert to Opus
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
                    torchaudio.save(wav_tmp.name, audio_out, 24000, format="wav")
                    wav_path = wav_tmp.name
                
                try:
                    result = subprocess.run(
                        ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libopus", "-b:a", "64k", "-f", "opus", "-"],
                        capture_output=True,
                        check=True
                    )
                    return Response(
                        content=result.stdout,
                        media_type="audio/opus",
                        headers={"X-TADA-Model": MODEL_ID}
                    )
                finally:
                    os.unlink(wav_path)
            else:
                # Return as WAV
                buffer = io.BytesIO()
                torchaudio.save(buffer, audio_out, 24000, format="wav")
                buffer.seek(0)
                return Response(
                    content=buffer.read(),
                    media_type="audio/wav",
                    headers={"X-TADA-Model": MODEL_ID}
                )
        finally:
            os.unlink(tmp_path)
    
    except Exception as e:
        logger.exception("Voice cloning failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT)
