#!/usr/bin/env python3
"""Minimal test of TADA on MPS."""

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from tada.modules.encoder import Encoder
from tada.modules.tada import TadaForCausalLM

DEVICE = "mps"
DTYPE = torch.bfloat16

print(f"Device: {DEVICE}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Load encoder
print("Loading encoder...")
encoder = Encoder.from_pretrained("HumeAI/tada-codec", subfolder="encoder").to(DEVICE)
print(f"Encoder device: {next(encoder.parameters()).device}")

# Load model
print("Loading model...")
model = TadaForCausalLM.from_pretrained("HumeAI/tada-1b", torch_dtype=DTYPE).to(DEVICE)
print(f"Model device: {next(model.parameters()).device}")

# Load reference audio
print("Loading reference audio...")
ref_path = hf_hub_download(repo_id="HumeAI/tada", repo_type="space", filename="samples/en/ljspeech.wav")
ref_wav, ref_sr = torchaudio.load(ref_path)
print(f"Audio shape: {ref_wav.shape}, sample_rate: {ref_sr}, device: {ref_wav.device}")

ref_wav = ref_wav.to(DEVICE)
print(f"Audio device after move: {ref_wav.device}")

# Create prompt
print("Creating prompt...")
prompt_text = "The examination and testimony of the experts enabled the commission to conclude that five shots may have been fired."
prompt = encoder(ref_wav, text=[prompt_text], sample_rate=ref_sr)
print(f"Prompt type: {type(prompt)}")
print(f"Prompt codes device: {prompt.codes.device if hasattr(prompt, 'codes') else 'N/A'}")

# Generate
print("Generating...")
with torch.no_grad():
    output = model.generate(prompt=prompt, text="Hello world, this is a test.")

print(f"Output audio shape: {output.audio[0].shape}")
print("Saving...")
torchaudio.save("/tmp/tada_minimal_test.wav", output.audio[0].cpu(), 24000)
print("Done! Saved to /tmp/tada_minimal_test.wav")
