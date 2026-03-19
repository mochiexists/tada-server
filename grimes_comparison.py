#!/usr/bin/env python3
"""Grimes voice clone comparison — PyTorch MPS baseline.

Regenerates the full Grimes transcript using 10s, 16s, and 28s reference
clips on both tada-1b and tada-3b-ml. Captures time-series memory profiles
(MPS allocated, MPS driver, RSS) at 20ms intervals for visualization.

Usage:
    python grimes_comparison.py
    python grimes_comparison.py --model tada-1b
    python grimes_comparison.py --model tada-3b-ml
"""

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import torchaudio

# Add eval dir to path for profiler import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "eval"))
from profiler import MemoryProfiler

from tada.modules.encoder import Encoder
from tada.modules.tada import TadaForCausalLM

# Patch MPS lm_head issue
def _fixed_lm_head_forward(self, hidden_states):
    return self.lm_head(hidden_states)
TadaForCausalLM._lm_head_forward = _fixed_lm_head_forward

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SAMPLES_DIR = Path(__file__).parent.parent.parent / "samples"
OUTPUT_DIR = Path(__file__).parent / "audio_outputs" / "grimes_comparison"
PROFILE_DIR = OUTPUT_DIR / "profiles"

# Full Whisper transcription of grimes-communism.mp4 (46.4s)
FULL_TEXT = (
    "I have a proposition for the Communists. So typically most of the Communists "
    "I know are not big fans of AI. But if you think about it, AI is actually the "
    "fastest path to Communism. So if implemented correctly, AI could actually "
    "theoretically solve for abundance. Like we could totally get to a situation "
    "where nobody has to work. Everybody is provided for with a comfortable state "
    "of being comfortable living. AI could automate all the farming, weed out "
    "systematic corruption. Thereby bringing us to as close as possible to genuine "
    "equality. So basically everything that everybody loves about Communism. But "
    "without the collective farm. Because let's be real, enforced farming is really "
    "not a vibe."
)

# Same reference configs as both MLX repos
REFS = {
    "ref-10s": {
        "path": SAMPLES_DIR / "grimes-10s.wav",
        "transcript": (
            "I have a proposition for the communists. So typically most of the "
            "communists I know are not big fans of AI. But if you think about it,"
        ),
    },
    "ref-16s": {
        "path": SAMPLES_DIR / "grimes-16s.wav",
        "transcript": (
            "I have a proposition for the communists. So typically most of the "
            "communists I know are not big fans of AI. But if you think about it, "
            "AI is actually the fastest path to communism. So if implemented correctly,"
        ),
    },
    "ref-28s": {
        "path": SAMPLES_DIR / "grimes-28s.wav",
        "transcript": FULL_TEXT,
    },
}

ALL_MODELS = {
    "tada-1b": "HumeAI/tada-1b",
    "tada-3b-ml": "HumeAI/tada-3b-ml",
}


def main():
    parser = argparse.ArgumentParser(description="Grimes voice clone comparison (PyTorch)")
    parser.add_argument("--model", default=None, choices=list(ALL_MODELS.keys()),
                        help="Run only this model (default: all)")
    args = parser.parse_args()

    models_to_run = {args.model: ALL_MODELS[args.model]} if args.model else ALL_MODELS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    # Copy original for easy comparison
    orig_full = SAMPLES_DIR / "grimes-full.wav"
    if orig_full.exists():
        shutil.copy2(str(orig_full), str(OUTPUT_DIR / "00_original_grimes-full.wav"))

    print(f"Device: {DEVICE}")
    print(f"Target text ({len(FULL_TEXT)} chars):")
    print(f'  "{FULL_TEXT[:80]}..."')
    print()

    # Start session-wide profiler
    session_profiler = MemoryProfiler(backend="pytorch", interval_s=0.02)
    session_profiler.start()
    session_profiler.mark("session_start")

    # Load encoder
    print("Loading encoder (HumeAI/tada-codec)...")
    session_profiler.mark("encoder_load_start")
    encoder = Encoder.from_pretrained("HumeAI/tada-codec").to(DEVICE)
    session_profiler.mark("encoder_load_done")

    # Pre-encode all references
    prompts_cache = {}
    for ref_name, ref_info in REFS.items():
        ref_path = ref_info["path"]
        if not ref_path.exists():
            print(f"  SKIP {ref_name}: {ref_path} not found")
            continue
        ref_wav, ref_sr = torchaudio.load(str(ref_path))
        ref_dur = ref_wav.shape[-1] / ref_sr
        ref_wav = ref_wav.to(DEVICE)
        print(f"  Encoding {ref_name} ({ref_dur:.1f}s)...", end="", flush=True)
        session_profiler.mark(f"encode_{ref_name}_start")
        t0 = time.time()
        prompts_cache[ref_name] = encoder(ref_wav, text=[ref_info["transcript"]], sample_rate=ref_sr)
        session_profiler.mark(f"encode_{ref_name}_done")
        print(f" {time.time()-t0:.1f}s")
        del ref_wav

    del encoder
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    session_profiler.mark("encoder_freed")

    results = []

    for model_short, model_id in models_to_run.items():
        print(f"\n{'='*70}")
        print(f"Model: {model_id}")
        print(f"{'='*70}")

        session_profiler.mark(f"model_load_{model_short}_start")
        t0 = time.time()
        model = TadaForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(DEVICE)
        session_profiler.mark(f"model_load_{model_short}_done")
        print(f"  Loaded in {time.time()-t0:.1f}s\n")

        for ref_name in sorted(prompts_cache.keys()):
            prompt = prompts_cache[ref_name]
            out_path = OUTPUT_DIR / f"{model_short}_{ref_name}.wav"
            label = f"{model_short}_{ref_name}"

            print(f"  {ref_name} -> generating full clip text...")

            # Per-generation profiler (separate trace file)
            gen_profiler = MemoryProfiler(backend="pytorch", interval_s=0.02)

            gc.collect()
            if DEVICE == "mps":
                torch.mps.empty_cache()
                torch.mps.synchronize()

            gen_profiler.start()
            gen_profiler.mark("generate_start")
            session_profiler.mark(f"gen_{label}_start")

            t0 = time.time()
            with torch.no_grad():
                output = model.generate(prompt=prompt, text=FULL_TEXT)
            if DEVICE == "mps":
                torch.mps.synchronize()
            gen_time = time.time() - t0

            gen_profiler.mark("generate_done")
            session_profiler.mark(f"gen_{label}_done")

            # Snapshot memory post-generation
            if DEVICE == "mps":
                mps_after = torch.mps.current_allocated_memory()
                mps_driver = torch.mps.driver_allocated_memory()
            else:
                mps_after = 0
                mps_driver = 0

            gen_profiler.mark("audio_write_start")
            audio = output.audio[0].cpu()
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            duration = audio.shape[-1] / 24000
            rtf = gen_time / duration if duration > 0 else float("inf")
            torchaudio.save(str(out_path), audio, 24000)
            gen_profiler.mark("audio_write_done")
            gen_profiler.stop()

            # Extract peak from profiler time-series
            peak_mps_driver = max(
                (s.mps_driver_bytes for s in gen_profiler.samples), default=0
            )
            peak_mps_alloc = max(
                (s.mps_allocated_bytes for s in gen_profiler.samples), default=0
            )
            peak_rss = max(
                (s.rss_bytes for s in gen_profiler.samples), default=0
            )

            # Per-stage timing from generate() output
            # On MPS (non-CUDA), llm_time and diffusion_time are in seconds (per-step average)
            llm_avg = float(output.llm_time) if output.llm_time is not None else 0.0
            diff_avg = float(output.diffusion_time) if output.diffusion_time is not None else 0.0
            # Decoder time is not instrumented in the installed package;
            # estimate as total - (llm + diffusion per-step totals)
            # Note: llm_avg and diff_avg are per-step averages in seconds

            result = {
                "model": model_short,
                "ref": ref_name,
                "duration_s": round(duration, 2),
                "gen_time_s": round(gen_time, 2),
                "rtf": round(rtf, 2),
                "stage_llm_avg_ms": round(llm_avg * 1000, 1) if llm_avg else None,
                "stage_diffusion_avg_ms": round(diff_avg * 1000, 1) if diff_avg else None,
                "peak_rss_gb": round(peak_rss / 1e9, 2),
                "peak_mps_alloc_gb": round(peak_mps_alloc / 1e9, 2),
                "peak_mps_driver_gb": round(peak_mps_driver / 1e9, 2),
                "mps_allocated_gb": round(mps_after / 1e9, 2),
                "mps_driver_gb": round(mps_driver / 1e9, 2),
                "num_profile_samples": len(gen_profiler.samples),
                "file": out_path.name,
            }
            results.append(result)
            print(
                f"    -> {duration:.2f}s audio | RTF: {rtf:.2f}x | gen: {gen_time:.2f}s"
                f" | LLM avg: {llm_avg*1000:.1f}ms | FM avg: {diff_avg*1000:.1f}ms"
                f" | Peak MPS driver: {peak_mps_driver/1e9:.1f}GB | {out_path.name}"
            )

            # Save per-generation profile
            prof_data = gen_profiler.to_dict()
            prof_data["label"] = f"pytorch-mps / {model_short} / {ref_name}"
            gen_profiler.export_json(PROFILE_DIR / f"{label}.json")
            gen_profiler.export_chrome_trace(PROFILE_DIR / f"{label}.trace.json")
            gen_profiler.export_plotly(
                PROFILE_DIR / f"{label}.html",
                title=f"PyTorch MPS — {model_short} / {ref_name}",
            )

            del output, audio

        del model
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()
        session_profiler.mark(f"model_freed_{model_short}")

    # Stop session profiler and save
    session_profiler.stop()
    session_profiler.export_json(PROFILE_DIR / "session.json")
    session_profiler.export_chrome_trace(PROFILE_DIR / "session.trace.json")
    session_profiler.export_plotly(
        PROFILE_DIR / "session.html",
        title="PyTorch MPS — Full Session Memory Profile",
    )

    # Save results (merge with existing to preserve other model runs)
    results_path = OUTPUT_DIR / "comparison_results.json"
    existing_results = []
    if results_path.exists():
        try:
            with open(results_path) as f:
                existing_results = json.load(f).get("results", [])
        except (json.JSONDecodeError, KeyError):
            pass
    # Remove old entries for models we just ran, keep others
    models_ran = {r["model"] for r in results}
    merged = [r for r in existing_results if r.get("model") not in models_ran] + results
    with open(results_path, "w") as f:
        json.dump({"target_text": FULL_TEXT, "runtime": "pytorch-mps", "results": merged}, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print("GRIMES COMPARISON — PyTorch MPS RESULTS")
    print(f"{'='*70}")
    print(f"Original clip: 28s | Text: {len(FULL_TEXT)} chars")
    print()
    print(f"{'Model':<12} {'Ref':<10} {'Duration':>8} {'GenTime':>8} {'RTF':>6} {'PkDriver':>9} {'PkAlloc':>8} {'Samples':>8}")
    print("-" * 85)
    for r in results:
        print(
            f"{r['model']:<12} {r['ref']:<10} {r['duration_s']:>7.2f}s {r['gen_time_s']:>7.2f}s"
            f" {r['rtf']:>5.2f}x {r['peak_mps_driver_gb']:>8.1f}GB {r['peak_mps_alloc_gb']:>7.1f}GB"
            f" {r['num_profile_samples']:>7}"
        )

    print(f"\nFiles saved to: {OUTPUT_DIR}/")
    print(f"Profiles saved to: {PROFILE_DIR}/")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
