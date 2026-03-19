#!/usr/bin/env python3
"""
Memory profiling script for TADA 1B vs 3B models on Apple Silicon.
Measures RSS, MPS allocator, and per-stage memory usage.
"""

import os
import sys
import time
import gc
import psutil
import torch
import torchaudio
from huggingface_hub import hf_hub_download

# Patch MPS device mismatch
from tada.modules.tada import TadaForCausalLM

def _fixed_lm_head_forward(self, hidden_states):
    return self.lm_head(hidden_states)
TadaForCausalLM._lm_head_forward = _fixed_lm_head_forward

from tada.modules.encoder import Encoder, EncoderOutput

DEVICE = os.getenv("TADA_DEVICE", "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE != "cpu" else torch.float32
PROCESS = psutil.Process(os.getpid())

def fmt_mb(bytes_val):
    return f"{bytes_val / 1024 / 1024:.1f} MB"

def fmt_gb(bytes_val):
    return f"{bytes_val / 1024 / 1024 / 1024:.2f} GB"

def get_memory_stats():
    """Return dict of current memory usage."""
    mem = PROCESS.memory_info()
    stats = {
        "rss": mem.rss,
        "vms": mem.vms,
    }
    if DEVICE == "mps":
        stats["mps_allocated"] = torch.mps.current_allocated_memory()
        stats["mps_driver"] = torch.mps.driver_allocated_memory()
    elif DEVICE == "cuda":
        stats["cuda_allocated"] = torch.cuda.memory_allocated()
        stats["cuda_reserved"] = torch.cuda.memory_reserved()
    return stats

def print_memory(label, stats, baseline=None):
    """Print memory stats with optional delta from baseline."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Process RSS:      {fmt_gb(stats['rss'])}")

    if DEVICE == "mps":
        print(f"  MPS Allocated:    {fmt_gb(stats['mps_allocated'])}")
        print(f"  MPS Driver:       {fmt_gb(stats['mps_driver'])}")
    elif DEVICE == "cuda":
        print(f"  CUDA Allocated:   {fmt_gb(stats['cuda_allocated'])}")
        print(f"  CUDA Reserved:    {fmt_gb(stats['cuda_reserved'])}")

    if baseline:
        rss_delta = stats["rss"] - baseline["rss"]
        print(f"  RSS Delta:        +{fmt_gb(rss_delta)}")
        if DEVICE == "mps":
            mps_delta = stats["mps_allocated"] - baseline["mps_allocated"]
            print(f"  MPS Alloc Delta:  +{fmt_gb(mps_delta)}")


def profile_model(model_id):
    """Profile a single model through full lifecycle."""
    print(f"\n{'#'*60}")
    print(f"  PROFILING: {model_id}")
    print(f"  Device: {DEVICE}, Dtype: {DTYPE}")
    print(f"{'#'*60}")

    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()

    baseline = get_memory_stats()
    print_memory("Baseline (before loading anything)", baseline)

    # Stage 1: Load encoder
    print("\n>>> Loading encoder (HumeAI/tada-codec)...")
    t0 = time.time()
    encoder = Encoder.from_pretrained("HumeAI/tada-codec", subfolder="encoder").to(DEVICE)
    t1 = time.time()
    encoder_stats = get_memory_stats()
    print_memory(f"After encoder load ({t1-t0:.1f}s)", encoder_stats, baseline)

    # Stage 2: Load model
    print(f"\n>>> Loading model ({model_id})...")
    t0 = time.time()
    model = TadaForCausalLM.from_pretrained(model_id, torch_dtype=DTYPE).to(DEVICE)
    t1 = time.time()
    model_stats = get_memory_stats()
    print_memory(f"After model load ({t1-t0:.1f}s)", model_stats, encoder_stats)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"\n  Model parameters: {total_params:,}")
    print(f"  Parameter memory: {fmt_gb(param_bytes)}")

    # Stage 3: Create voice prompt
    print("\n>>> Creating voice prompt from LJSpeech sample...")
    ref_path = hf_hub_download(repo_id="HumeAI/tada", repo_type="space", filename="samples/en/ljspeech.wav")
    ref_wav, ref_sr = torchaudio.load(ref_path)
    ref_wav = ref_wav.to(DEVICE)
    prompt_text = "The examination and testimony of the experts enabled the commission to conclude that five shots may have been fired."

    t0 = time.time()
    prompt = encoder(ref_wav, text=[prompt_text], sample_rate=ref_sr)
    t1 = time.time()
    prompt_stats = get_memory_stats()
    print_memory(f"After prompt encoding ({t1-t0:.1f}s)", prompt_stats, model_stats)

    # Stage 4: Generate speech
    test_text = "Hello, this is a memory profiling test for the TADA text to speech model."
    print(f"\n>>> Generating speech: '{test_text}'")

    t0 = time.time()
    with torch.no_grad():
        output = model.generate(prompt=prompt, text=test_text)
    t1 = time.time()
    gen_stats = get_memory_stats()
    print_memory(f"After generation ({t1-t0:.1f}s)", gen_stats, prompt_stats)

    # Peak stats
    print_memory(f"PEAK (total for {model_id})", gen_stats, baseline)

    # Audio info
    audio = output.audio[0].cpu()
    duration = audio.shape[-1] / 24000
    print(f"\n  Audio duration:   {duration:.1f}s")
    print(f"  Generation time:  {t1-t0:.1f}s")
    print(f"  RTF:              {(t1-t0)/duration:.2f}x realtime")

    # Cleanup
    del model, encoder, prompt, output, audio, ref_wav
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()

    time.sleep(2)  # let MPS driver release
    cleanup_stats = get_memory_stats()
    print_memory("After cleanup", cleanup_stats, baseline)

    return gen_stats


def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else ["HumeAI/tada-1b", "HumeAI/tada-3b-ml"]

    print(f"System RAM: {fmt_gb(psutil.virtual_memory().total)}")
    print(f"Available:  {fmt_gb(psutil.virtual_memory().available)}")
    print(f"Device:     {DEVICE}")

    results = {}
    for model_id in models:
        try:
            results[model_id] = profile_model(model_id)
        except Exception as e:
            print(f"\n!!! FAILED to profile {model_id}: {e}")
            import traceback
            traceback.print_exc()
            results[model_id] = None

    # Summary
    print(f"\n\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for model_id, stats in results.items():
        if stats:
            print(f"\n  {model_id}:")
            print(f"    Peak RSS:       {fmt_gb(stats['rss'])}")
            if DEVICE == "mps":
                print(f"    Peak MPS Alloc: {fmt_gb(stats['mps_allocated'])}")
                print(f"    Peak MPS Driver:{fmt_gb(stats['mps_driver'])}")
        else:
            print(f"\n  {model_id}: FAILED")


if __name__ == "__main__":
    main()
