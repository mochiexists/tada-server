#!/usr/bin/env python3
"""
TADA Model Benchmark Suite

Comprehensive memory and performance profiling for HumeAI TADA TTS models.
Tests multiple prompt lengths, devices, and model sizes.

Usage:
    # Run full benchmark (all models, all devices)
    python benchmark.py

    # Single model, single device
    python benchmark.py --models HumeAI/tada-1b --devices mps

    # Custom prompts
    python benchmark.py --models HumeAI/tada-3b-ml --devices cpu --prompts short,medium

    # Save results to JSON
    python benchmark.py --output results.json

Environment:
    HF_TOKEN - HuggingFace token (required for gated model access)
"""

import argparse
import gc
import io
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import importlib

import psutil
import torch
import torchaudio
from huggingface_hub import hf_hub_download

from tada.modules.encoder import Encoder, EncoderOutput
from tada.modules.tada import TadaForCausalLM

# MPS fix: patch lm_head_forward to avoid CPU/MPS device mismatch
def _fixed_lm_head_forward(self, hidden_states):
    return self.lm_head(hidden_states)

TadaForCausalLM._lm_head_forward = _fixed_lm_head_forward


# ---------------------------------------------------------------------------
# Test prompts — varying lengths to measure scaling behaviour
# ---------------------------------------------------------------------------
TEST_PROMPTS = {
    "short": "Hello, how are you today?",
    "medium": (
        "The quick brown fox jumps over the lazy dog. "
        "This is a test of the text to speech system with a moderate length input "
        "that should produce roughly ten to fifteen seconds of audio output."
    ),
    "long": (
        "In the beginning, the universe was created. This has made a lot of people "
        "very angry and been widely regarded as a bad move. Many were increasingly of "
        "the opinion that they had all made a big mistake in coming down from the trees "
        "in the first place. And some said that even the trees had been a bad move, and "
        "that no one should ever have left the oceans. The story so far: in the beginning "
        "the universe was created. This has made a lot of people very angry and has been "
        "widely regarded as a bad move."
    ),
    "very_long": (
        "Four score and seven years ago our fathers brought forth on this continent, "
        "a new nation, conceived in Liberty, and dedicated to the proposition that all "
        "men are created equal. Now we are engaged in a great civil war, testing whether "
        "that nation, or any nation so conceived and so dedicated, can long endure. We "
        "are met on a great battle-field of that war. We have come to dedicate a portion "
        "of that field, as a final resting place for those who here gave their lives that "
        "that nation might live. It is altogether fitting and proper that we should do "
        "this. But, in a larger sense, we can not dedicate, we can not consecrate, we "
        "can not hallow this ground. The brave men, living and dead, who struggled here, "
        "have consecrated it, far above our poor power to add or detract. The world will "
        "little note, nor long remember what we say here, but it can never forget what "
        "they did here."
    ),
}


# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------
@dataclass
class MemorySnapshot:
    rss_mb: float = 0.0
    vms_mb: float = 0.0
    mps_allocated_mb: float = 0.0
    mps_driver_mb: float = 0.0
    cuda_allocated_mb: float = 0.0
    cuda_reserved_mb: float = 0.0


@dataclass
class StageResult:
    stage: str
    duration_s: float
    memory: MemorySnapshot


@dataclass
class GenerationResult:
    prompt_name: str
    prompt_chars: int
    prompt_words: int
    audio_duration_s: float
    generation_time_s: float
    rtf: float  # realtime factor (generation_time / audio_duration)
    memory_before: MemorySnapshot
    memory_after: MemorySnapshot
    memory_peak_rss_mb: float
    voice_ref: str = "default"  # voice reference label
    voice_ref_duration_s: float = 0.0


@dataclass
class ModelBenchmark:
    model_id: str
    device: str
    dtype: str
    total_params: int
    param_memory_mb: float
    implementation: str = "baseline"
    voice_ref: str = "default"
    voice_ref_duration_s: float = 0.0
    stages: list = field(default_factory=list)
    generations: list = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class BenchmarkReport:
    timestamp: str
    system: dict = field(default_factory=dict)
    benchmarks: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_system_info() -> dict:
    """Collect system hardware and software info."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
    }

    if torch.backends.mps.is_available():
        info["mps_available"] = True
    if torch.cuda.is_available():
        info["cuda_available"] = True
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = round(torch.cuda.get_device_properties(0).total_mem / 1024**3, 1)

    # macOS: get chip name
    if platform.system() == "Darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            info["chip"] = chip
        except Exception:
            pass

    return info


def take_snapshot(device: str) -> MemorySnapshot:
    """Capture current memory usage."""
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    snap = MemorySnapshot(
        rss_mb=round(mem.rss / 1024**2, 1),
        vms_mb=round(mem.vms / 1024**2, 1),
    )
    if device == "mps":
        snap.mps_allocated_mb = round(torch.mps.current_allocated_memory() / 1024**2, 1)
        snap.mps_driver_mb = round(torch.mps.driver_allocated_memory() / 1024**2, 1)
    elif device == "cuda":
        snap.cuda_allocated_mb = round(torch.cuda.memory_allocated() / 1024**2, 1)
        snap.cuda_reserved_mb = round(torch.cuda.memory_reserved() / 1024**2, 1)
    return snap


def cleanup(device: str):
    """Force garbage collection and cache clearing."""
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def fmt_mb(mb: float) -> str:
    if mb >= 1024:
        return f"{mb/1024:.2f} GB"
    return f"{mb:.0f} MB"


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Voice reference presets
# ---------------------------------------------------------------------------
VOICE_REFS = {
    "ljspeech": {
        "label": "ljspeech",
        "source": "hf",  # download from HuggingFace
        "repo_id": "HumeAI/tada",
        "repo_type": "space",
        "filename": "samples/en/ljspeech.wav",
        "transcript": "The examination and testimony of the experts enabled the commission to conclude that five shots may have been fired.",
    },
    "grimes-10s": {
        "label": "grimes-10s",
        "source": "local",
        "path": "samples/grimes-10s.wav",
        "transcript": (
            "I have a proposition for the communists. So typically most of the "
            "communists I know are not big fans of AI. But if you think about it."
        ),
    },
    "grimes-16s": {
        "label": "grimes-16s",
        "source": "local",
        "path": "samples/grimes-16s.wav",
        "transcript": (
            "I have a proposition for the communists. So typically most of the "
            "communists I know are not big fans of AI. But if you think about it, "
            "AI is actually the fastest path to communism."
        ),
    },
    "grimes-28s": {
        "label": "grimes-28s",
        "source": "local",
        "path": "samples/grimes-28s.wav",
        "transcript": (
            "I have a proposition for the communists. So typically most of the "
            "communists I know are not big fans of AI. But if you think about it, "
            "AI is actually the fastest path to communism. Because if we get AI and "
            "robots and stuff, eventually everything can be automated, nobody has to "
            "work, and the government can just pay everyone a universal basic income."
        ),
    },
}

MLX_IMPLEMENTATIONS = {
    "claude": Path(__file__).parent.parent / "tada-claude" / "src",
    "codex": Path(__file__).parent.parent / "tada-codex" / "src",
}


def _clear_tada_mlx_modules():
    for module_name in list(sys.modules):
        if module_name == "tada_mlx" or module_name.startswith("tada_mlx."):
            del sys.modules[module_name]


def _import_tada_mlx_from(src_path: Path):
    _clear_tada_mlx_modules()
    importlib.invalidate_caches()
    src_str = str(src_path)
    sys.path = [entry for entry in sys.path if entry != src_str]
    sys.path.insert(0, src_str)


def benchmark_model_mlx(
    model_id: str,
    prompt_names: list[str],
    voice_ref_name: str = "ljspeech",
    implementation: str = "claude",
) -> ModelBenchmark:
    """Run full benchmark for one model using the MLX backend."""
    model = None
    prompt = None
    encoder = None
    aligner_model = None
    prompt_encoder_model = None

    try:
        import mlx.core as mx
        import mlx.nn
    except ImportError as exc:
        raise RuntimeError(
            "MLX benchmarking requires `mlx` and `mlx-lm` in the active environment. "
            "Install them here or run benchmark.py from an MLX-capable venv."
        ) from exc
    import numpy as np
    import soundfile as sf

    src_path = MLX_IMPLEMENTATIONS.get(implementation)
    if src_path is None:
        raise ValueError(f"Unknown MLX implementation: {implementation}")
    if not src_path.exists():
        raise FileNotFoundError(f"MLX implementation not found: {src_path}")
    _import_tada_mlx_from(src_path)

    voice_ref = VOICE_REFS[voice_ref_name]

    result = ModelBenchmark(
        model_id=model_id,
        device="mlx",
        implementation=implementation,
        dtype="float32",
        total_params=0,
        param_memory_mb=0.0,
        voice_ref=voice_ref_name,
    )

    print(f"\n{'='*70}")
    print(f"  {model_id} on MLX/{implementation}")
    print(f"  Voice ref: {voice_ref_name}")
    print(f"{'='*70}")

    gc.collect()

    def mlx_snapshot() -> MemorySnapshot:
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        # Force pending ops to complete for accurate memory reading
        mx.synchronize()
        snap = MemorySnapshot(
            rss_mb=round(mem.rss / 1024**2, 1),
            vms_mb=round(mem.vms / 1024**2, 1),
            mps_allocated_mb=round(mx.get_active_memory() / 1024**2, 1),
            mps_driver_mb=round(mx.get_peak_memory() / 1024**2, 1),
        )
        return snap

    if voice_ref["source"] == "hf":
        ref_path = hf_hub_download(
            repo_id=voice_ref["repo_id"],
            repo_type=voice_ref.get("repo_type", "model"),
            filename=voice_ref["filename"],
        )
    else:
        ref_path = str(Path(__file__).parent / voice_ref["path"])

    if implementation == "claude":
        from tada_mlx.model import TadaForCausalLM as TadaMLX, InferenceOptions
        from tada_mlx.encoder import Encoder as EncoderMLX

        print("  Loading MLX encoder...", end="", flush=True)
        t0 = time.time()
        encoder = EncoderMLX.from_pretrained("HumeAI/tada-codec")
        dt = time.time() - t0
        snap = mlx_snapshot()
        result.stages.append(asdict(StageResult(stage="encoder_load", duration_s=round(dt, 1), memory=snap)))
        print(f" {dt:.1f}s  (RSS: {fmt_mb(snap.rss_mb)})")

        print(f"  Loading {model_id} (MLX/{implementation})...", end="", flush=True)
        t0 = time.time()
        model = TadaMLX.from_pretrained(model_id)
        dt = time.time() - t0
        snap = mlx_snapshot()
        result.stages.append(asdict(StageResult(stage="model_load", duration_s=round(dt, 1), memory=snap)))

        param_count = sum(p.size for k, p in mlx.nn.utils.tree_flatten(model.parameters()))
        param_bytes = sum(p.size * p.itemsize for k, p in mlx.nn.utils.tree_flatten(model.parameters()))
        result.total_params = param_count
        result.param_memory_mb = round(param_bytes / 1024**2, 1)
        print(f" {dt:.1f}s  (RSS: {fmt_mb(snap.rss_mb)}, params: {param_count:,})")

        wav_np, ref_sr = sf.read(ref_path)
        ref_dur = len(wav_np) / ref_sr
        result.voice_ref_duration_s = round(ref_dur, 1)

        print(f"  Encoding voice prompt ({voice_ref_name}, {ref_dur:.1f}s)...", end="", flush=True)
        t0 = time.time()
        audio_mx = mx.array(wav_np.reshape(1, -1).astype(np.float32))
        prompt = encoder(audio_mx, text=[voice_ref["transcript"]], sample_rate=ref_sr)
        mx.synchronize()
        dt = time.time() - t0
        snap = mlx_snapshot()
        result.stages.append(asdict(StageResult(stage="prompt_encode", duration_s=round(dt, 1), memory=snap)))
        print(f" {dt:.1f}s  (RSS: {fmt_mb(snap.rss_mb)})")

        def run_generation(text: str):
            return model.generate(
                prompt=prompt,
                text=text,
                inference_options=InferenceOptions(
                    num_flow_matching_steps=10,
                    acoustic_cfg_scale=1.6,
                    noise_temperature=0.9,
                    text_temperature=0.6,
                ),
            )

        def get_audio_duration(output) -> float:
            if output.audio and output.audio[0] is not None:
                audio_np = np.array(output.audio[0].astype(mx.float32))
                return len(audio_np) / 24000
            return 0.0

    elif implementation == "codex":
        from tada_mlx import TadaForCausalLM as TadaMLX
        from tada_mlx.model import TadaAlignerEncoderModel, TadaPromptEncoderModel
        from tada_mlx.modules import InferenceOptions, LogitAligner
        from tada_mlx.utils.text import normalize_text

        repo_root = src_path.parent
        aligner_artifacts_dir = repo_root / "artifacts" / "aligner_frontend"
        prompt_encoder_artifacts_dir = repo_root / "artifacts" / "prompt_encoder"
        flow_artifacts_dir = repo_root / "artifacts" / "flow_matching"
        decoder_artifacts_dir = repo_root / "artifacts" / "decoder"
        for artifact_dir in (
            aligner_artifacts_dir,
            prompt_encoder_artifacts_dir,
            flow_artifacts_dir,
            decoder_artifacts_dir,
        ):
            if not artifact_dir.exists():
                raise FileNotFoundError(f"Missing required Codex artifact directory: {artifact_dir}")

        print("  Loading MLX encoder...", end="", flush=True)
        t0 = time.time()
        aligner_model, _ = TadaAlignerEncoderModel.from_artifacts(aligner_artifacts_dir)
        prompt_encoder_model, _ = TadaPromptEncoderModel.from_artifacts(prompt_encoder_artifacts_dir)
        dt = time.time() - t0
        snap = mlx_snapshot()
        result.stages.append(asdict(StageResult(stage="encoder_load", duration_s=round(dt, 1), memory=snap)))
        print(f" {dt:.1f}s  (RSS: {fmt_mb(snap.rss_mb)})")

        print(f"  Loading {model_id} (MLX/{implementation})...", end="", flush=True)
        t0 = time.time()
        model = TadaMLX.from_pretrained(
            model_id,
            flow_artifacts_dir=flow_artifacts_dir,
            decoder_artifacts_dir=decoder_artifacts_dir,
        )
        dt = time.time() - t0
        snap = mlx_snapshot()
        result.stages.append(asdict(StageResult(stage="model_load", duration_s=round(dt, 1), memory=snap)))

        param_count = sum(p.size for k, p in mlx.nn.utils.tree_flatten(model.parameters()))
        param_bytes = sum(p.size * p.itemsize for k, p in mlx.nn.utils.tree_flatten(model.parameters()))
        result.total_params = param_count
        result.param_memory_mb = round(param_bytes / 1024**2, 1)
        print(f" {dt:.1f}s  (RSS: {fmt_mb(snap.rss_mb)}, params: {param_count:,})")

        audio_torch, ref_sr = torchaudio.load(str(ref_path))
        audio_torch = audio_torch[:1]
        audio_len = torch.tensor([audio_torch.shape[-1]], dtype=torch.long)
        ref_dur = float(audio_torch.shape[-1] / ref_sr)
        result.voice_ref_duration_s = round(ref_dur, 1)

        print(f"  Encoding voice prompt ({voice_ref_name}, {ref_dur:.1f}s)...", end="", flush=True)
        t0 = time.time()
        if ref_sr != 24000:
            audio_24k = torchaudio.functional.resample(audio_torch, ref_sr, 24000)
            audio_len_24k = (audio_len * 24000 / ref_sr).long()
        else:
            audio_24k = audio_torch
            audio_len_24k = audio_len
        audio_16k = torchaudio.functional.resample(audio_24k, 24000, 16000)
        input_lengths = mx.array(np.ceil(audio_len_24k.numpy() / 24000 * 50).astype(np.int32))
        prompt_text = normalize_text(voice_ref["transcript"])
        logits, _ = aligner_model(mx.array(audio_16k.numpy(), dtype=mx.float32))
        align_output = LogitAligner(model.tokenizer).align(
            logits,
            text=[prompt_text],
            input_lengths=input_lengths,
            return_logits=False,
        )
        prompt_output = prompt_encoder_model.encoder.encode_aligned(
            mx.array(audio_24k.numpy(), dtype=mx.float32),
            align_output.token_positions,
            align_output.token_masks,
            sample=False,
        )
        prompt = SimpleNamespace(
            text=[prompt_text],
            audio_len=audio_len_24k.numpy(),
            token_positions=np.array(align_output.token_positions),
            token_values=np.array(prompt_output.token_values),
            sample_rate=24000,
        )
        mx.synchronize()
        dt = time.time() - t0
        snap = mlx_snapshot()
        result.stages.append(asdict(StageResult(stage="prompt_encode", duration_s=round(dt, 1), memory=snap)))
        print(f" {dt:.1f}s  (RSS: {fmt_mb(snap.rss_mb)})")

        def run_generation(text: str):
            return model.generate(
                prompt=prompt,
                text=text,
                inference_options=InferenceOptions(
                    num_flow_matching_steps=10,
                    acoustic_cfg_scale=1.6,
                    noise_temperature=0.9,
                    text_temperature=0.6,
                ),
            )

        def get_audio_duration(output) -> float:
            if output.audio and output.audio[0] is not None:
                return len(output.audio[0]) / 24000
            return 0.0

    else:
        raise ValueError(f"Unsupported MLX implementation: {implementation}")

    # --- Generations ---
    print()
    print(f"  {'Prompt':<12} {'Chars':>6} {'Words':>6} {'Audio':>7} {'Gen Time':>9} {'RTF':>7} {'Peak Mem':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*7} {'-'*9} {'-'*7} {'-'*10}")

    for name in prompt_names:
        text = TEST_PROMPTS[name]
        mem_before = mlx_snapshot()

        t0 = time.time()
        output = run_generation(text)
        gen_time = time.time() - t0

        mem_after = mlx_snapshot()
        audio_dur = get_audio_duration(output)

        rtf = gen_time / audio_dur if audio_dur > 0 else float("inf")

        gen_result = GenerationResult(
            prompt_name=name,
            prompt_chars=len(text),
            prompt_words=len(text.split()),
            audio_duration_s=round(audio_dur, 2),
            generation_time_s=round(gen_time, 2),
            rtf=round(rtf, 2),
            memory_before=mem_before,
            memory_after=mem_after,
            memory_peak_rss_mb=mem_after.rss_mb,
            voice_ref=voice_ref_name,
            voice_ref_duration_s=round(ref_dur, 1),
        )
        result.generations.append(asdict(gen_result))

        print(
            f"  {name:<12} {len(text):>6} {len(text.split()):>6} "
            f"{audio_dur:>6.1f}s {gen_time:>8.1f}s {rtf:>6.1f}x "
            f"{fmt_mb(mem_after.mps_allocated_mb):>10}"
        )

        del output
        gc.collect()

    # Cleanup
    model = None
    encoder = None
    aligner_model = None
    prompt_encoder_model = None
    prompt = None
    gc.collect()

    final = mlx_snapshot()
    result.stages.append(asdict(StageResult(stage="cleanup", duration_s=0, memory=final)))

    return result


def benchmark_model(
    model_id: str,
    device: str,
    prompt_names: list[str],
    voice_ref_name: str = "ljspeech",
) -> ModelBenchmark:
    """Run full benchmark for one model on one device with a given voice reference."""
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    voice_ref = VOICE_REFS[voice_ref_name]

    result = ModelBenchmark(
        model_id=model_id,
        device=device,
        dtype=str(dtype),
        total_params=0,
        param_memory_mb=0.0,
        voice_ref=voice_ref_name,
    )

    print(f"\n{'='*70}")
    print(f"  {model_id} on {device} ({dtype})")
    print(f"  Voice ref: {voice_ref_name}")
    print(f"{'='*70}")

    cleanup(device)
    baseline = take_snapshot(device)

    # --- Stage: Load encoder ---
    print("  Loading encoder...", end="", flush=True)
    t0 = time.time()
    encoder = Encoder.from_pretrained("HumeAI/tada-codec", subfolder="encoder").to(device)
    dt = time.time() - t0
    snap = take_snapshot(device)
    result.stages.append(asdict(StageResult(stage="encoder_load", duration_s=round(dt, 1), memory=snap)))
    print(f" {dt:.1f}s  (RSS: {fmt_mb(snap.rss_mb)})")

    # --- Stage: Load model ---
    print(f"  Loading {model_id}...", end="", flush=True)
    t0 = time.time()
    model = TadaForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    dt = time.time() - t0
    snap = take_snapshot(device)
    result.stages.append(asdict(StageResult(stage="model_load", duration_s=round(dt, 1), memory=snap)))

    result.total_params = sum(p.numel() for p in model.parameters())
    result.param_memory_mb = round(
        sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2, 1
    )
    print(f" {dt:.1f}s  (RSS: {fmt_mb(snap.rss_mb)}, params: {result.total_params:,})")

    # --- Stage: Create voice prompt ---
    if voice_ref["source"] == "hf":
        ref_path = hf_hub_download(
            repo_id=voice_ref["repo_id"],
            repo_type=voice_ref.get("repo_type", "model"),
            filename=voice_ref["filename"],
        )
    else:
        ref_path = str(Path(__file__).parent / voice_ref["path"])

    ref_wav, ref_sr = torchaudio.load(ref_path)
    ref_dur = ref_wav.shape[-1] / ref_sr
    result.voice_ref_duration_s = round(ref_dur, 1)
    ref_wav = ref_wav.to(device)

    print(f"  Encoding voice prompt ({voice_ref_name}, {ref_dur:.1f}s)...", end="", flush=True)
    t0 = time.time()
    prompt = encoder(ref_wav, text=[voice_ref["transcript"]], sample_rate=ref_sr)
    dt = time.time() - t0
    snap = take_snapshot(device)
    result.stages.append(asdict(StageResult(stage="prompt_encode", duration_s=round(dt, 1), memory=snap)))
    print(f" {dt:.1f}s  (RSS: {fmt_mb(snap.rss_mb)})")

    del ref_wav  # free reference audio

    # --- Generations ---
    print()
    print(f"  {'Prompt':<12} {'Chars':>6} {'Words':>6} {'Audio':>7} {'Gen Time':>9} {'RTF':>7} {'Peak RSS':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*7} {'-'*9} {'-'*7} {'-'*10}")

    for name in prompt_names:
        text = TEST_PROMPTS[name]
        mem_before = take_snapshot(device)

        t0 = time.time()
        with torch.no_grad():
            output = model.generate(prompt=prompt, text=text)
        gen_time = time.time() - t0

        mem_after = take_snapshot(device)

        audio = output.audio[0].cpu()
        audio_dur = audio.shape[-1] / 24000
        rtf = gen_time / audio_dur if audio_dur > 0 else float("inf")

        gen_result = GenerationResult(
            prompt_name=name,
            prompt_chars=len(text),
            prompt_words=len(text.split()),
            audio_duration_s=round(audio_dur, 2),
            generation_time_s=round(gen_time, 2),
            rtf=round(rtf, 2),
            memory_before=mem_before,
            memory_after=mem_after,
            memory_peak_rss_mb=mem_after.rss_mb,
            voice_ref=voice_ref_name,
            voice_ref_duration_s=round(ref_dur, 1),
        )
        result.generations.append(asdict(gen_result))

        print(
            f"  {name:<12} {len(text):>6} {len(text.split()):>6} "
            f"{audio_dur:>6.1f}s {gen_time:>8.1f}s {rtf:>6.1f}x "
            f"{fmt_mb(mem_after.rss_mb):>10}"
        )

        del output, audio
        cleanup(device)

    # Cleanup
    del model, encoder, prompt
    cleanup(device)
    time.sleep(1)

    final = take_snapshot(device)
    result.stages.append(asdict(StageResult(stage="cleanup", duration_s=0, memory=final)))

    return result


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------
def _bm_get(bm, key, default=None):
    """Access benchmark data whether it's a dict or dataclass."""
    if isinstance(bm, dict):
        return bm.get(key, default)
    return getattr(bm, key, default)


def _runtime_label(bm) -> str:
    device = _bm_get(bm, "device", "?")
    implementation = _bm_get(bm, "implementation", "baseline")
    if device == "mlx":
        return f"mlx/{implementation}"
    if implementation and implementation != "baseline":
        return f"{device}/{implementation}"
    return f"{device}/baseline"


def print_summary(report: BenchmarkReport):
    """Print a markdown-formatted summary table."""
    print(f"\n\n{'#'*70}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'#'*70}")
    sys_info = report.system if isinstance(report.system, dict) else asdict(report.system) if hasattr(report, '__dataclass_fields__') else report.system
    print(f"\n  System: {sys_info.get('chip', sys_info.get('processor', 'unknown'))}")
    print(f"  RAM: {sys_info.get('ram_total_gb', '?')} GB")
    print(f"  PyTorch: {sys_info.get('torch_version', '?')}")
    print(f"  Date: {report.timestamp}")

    for bm in report.benchmarks:
        error = _bm_get(bm, 'error')
        if error:
            print(f"\n  {_bm_get(bm, 'model_id')} ({_runtime_label(bm)}): FAILED — {error}")
            continue

        model_id = _bm_get(bm, 'model_id')
        runtime = _runtime_label(bm)
        total_params = _bm_get(bm, 'total_params', 0)
        param_memory_mb = _bm_get(bm, 'param_memory_mb', 0)
        stages = _bm_get(bm, 'stages', [])
        generations = _bm_get(bm, 'generations', [])

        print(f"\n  ## {model_id} on {runtime}")
        print(f"  Parameters: {total_params:,} | Weight memory: {fmt_mb(param_memory_mb)}")

        # Stages
        for stage in stages:
            mem = stage["memory"]
            label = stage["stage"]
            dur = stage["duration_s"]
            rss = fmt_mb(mem["rss_mb"])
            extra = ""
            if mem.get("mps_driver_mb", 0) > 0:
                extra = f" | MPS driver: {fmt_mb(mem['mps_driver_mb'])}"
            print(f"    {label:<20} {dur:>6.1f}s  RSS: {rss:>10}{extra}")

        # Generations
        if generations:
            print()
            print(f"    {'Prompt':<12} {'Chars':>6} {'Words':>6} {'Audio':>7} {'Time':>8} {'RTF':>7} {'RSS':>10}")
            print(f"    {'-'*12} {'-'*6} {'-'*6} {'-'*7} {'-'*8} {'-'*7} {'-'*10}")
            for gen in generations:
                print(
                    f"    {gen['prompt_name']:<12} {gen['prompt_chars']:>6} {gen['prompt_words']:>6} "
                    f"{gen['audio_duration_s']:>6.1f}s {gen['generation_time_s']:>7.1f}s "
                    f"{gen['rtf']:>6.1f}x {fmt_mb(gen['memory_peak_rss_mb']):>10}"
                )


def generate_markdown(report: BenchmarkReport) -> str:
    """Generate a markdown report suitable for a PR or README."""
    lines = []
    lines.append("# TADA Benchmark Results\n")
    lines.append(f"**Date:** {report.timestamp}  ")
    lines.append(f"**System:** {report.system.get('chip', report.system.get('processor', 'unknown'))}  ")
    lines.append(f"**RAM:** {report.system.get('ram_total_gb', '?')} GB  ")
    lines.append(f"**PyTorch:** {report.system.get('torch_version', '?')}  ")
    lines.append(f"**Python:** {report.system.get('python_version', '?')}  ")
    lines.append(f"**OS:** {report.system.get('platform', '?')}\n")

    # Build comparison table
    lines.append("## Generation Performance\n")
    lines.append("| Model | Device | Impl | Voice Ref | Prompt | Chars | Audio | Gen Time | RTF | Peak RSS |")
    lines.append("|-------|--------|------|-----------|--------|------:|------:|---------:|----:|---------:|")

    for bm in report.benchmarks:
        if _bm_get(bm, 'error'):
            lines.append(
                f"| {_bm_get(bm, 'model_id')} | {_bm_get(bm, 'device')} | "
                f"{_bm_get(bm, 'implementation', 'baseline')} | {_bm_get(bm, 'voice_ref', '?')} | — | — | — | — | — | FAILED |"
            )
            continue
        voice_ref = _bm_get(bm, 'voice_ref', 'default')
        ref_dur = _bm_get(bm, 'voice_ref_duration_s', 0)
        ref_label = f"{voice_ref} ({ref_dur:.0f}s)" if ref_dur else voice_ref
        for gen in _bm_get(bm, 'generations', []):
            model_short = _bm_get(bm, 'model_id', '').split("/")[-1]
            lines.append(
                f"| {model_short} | {_bm_get(bm, 'device')} | {_bm_get(bm, 'implementation', 'baseline')} | "
                f"{ref_label} | {gen['prompt_name']} | {gen['prompt_chars']} | "
                f"{gen['audio_duration_s']:.1f}s | {gen['generation_time_s']:.1f}s | "
                f"{gen['rtf']:.1f}x | {fmt_mb(gen['memory_peak_rss_mb'])} |"
            )

    lines.append("\n## Memory Profile\n")
    lines.append("| Model | Device | Impl | Dtype | Params | Weight Mem | Encoder Load | Model Load | Peak MPS Driver | Peak RSS |")
    lines.append("|-------|--------|------|-------|-------:|-----------:|-------------:|-----------:|----------------:|---------:|")

    for bm in report.benchmarks:
        if _bm_get(bm, 'error'):
            continue
        model_short = _bm_get(bm, 'model_id', '').split("/")[-1]
        stages = _bm_get(bm, 'stages', [])
        generations = _bm_get(bm, 'generations', [])
        encoder_stage = next((s for s in stages if s["stage"] == "encoder_load"), None)
        model_stage = next((s for s in stages if s["stage"] == "model_load"), None)

        peak_rss = max(gen["memory_peak_rss_mb"] for gen in generations) if generations else 0
        peak_mps = max(
            (s["memory"]["mps_driver_mb"] for s in stages if s["memory"].get("mps_driver_mb", 0) > 0),
            default=0,
        )

        # Also check generation snapshots for peak MPS
        for gen in generations:
            for key in ("memory_before", "memory_after"):
                mps = gen[key].get("mps_driver_mb", 0)
                if mps > peak_mps:
                    peak_mps = mps

        dtype_short = _bm_get(bm, 'dtype', '').replace("torch.", "")
        lines.append(
            f"| {model_short} | {_bm_get(bm, 'device')} | {_bm_get(bm, 'implementation', 'baseline')} | "
            f"{dtype_short} | {_bm_get(bm, 'total_params', 0):,} | "
            f"{fmt_mb(_bm_get(bm, 'param_memory_mb', 0))} | "
            f"{encoder_stage['duration_s']:.1f}s | "
            f"{model_stage['duration_s']:.1f}s | "
            f"{fmt_mb(peak_mps) if peak_mps > 0 else 'N/A'} | "
            f"{fmt_mb(peak_rss)} |"
        )

    lines.append("\n## Prompt Scaling\n")
    lines.append("Shows how generation time and audio length scale with input text length.\n")

    for bm in report.benchmarks:
        if _bm_get(bm, 'error') or not _bm_get(bm, 'generations'):
            continue
        model_short = _bm_get(bm, 'model_id', '').split("/")[-1]
        lines.append(f"### {model_short} ({_runtime_label(bm)})\n")
        lines.append("| Prompt | Input Chars | Input Words | Output Audio | Gen Time | RTF |")
        lines.append("|--------|------------:|------------:|-------------:|---------:|----:|")
        for gen in _bm_get(bm, 'generations', []):
            lines.append(
                f"| {gen['prompt_name']} | {gen['prompt_chars']} | {gen['prompt_words']} | "
                f"{gen['audio_duration_s']:.1f}s | {gen['generation_time_s']:.1f}s | {gen['rtf']:.1f}x |"
            )
        lines.append("")

    lines.append("\n---\n")
    lines.append("*Generated by `benchmark.py` from [tada-server](https://github.com/mochiexists/tada-server)*\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="TADA TTS Benchmark Suite")
    p.add_argument(
        "--models",
        default="HumeAI/tada-1b,HumeAI/tada-3b-ml",
        help="Comma-separated model IDs (default: both 1b and 3b)",
    )
    p.add_argument(
        "--devices",
        default=None,
        help="Comma-separated devices: mps,cpu,cuda (default: auto-detect available)",
    )
    p.add_argument(
        "--prompts",
        default="short,medium,long,very_long",
        help="Comma-separated prompt names (default: all)",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Save JSON results to file",
    )
    p.add_argument(
        "--markdown", "-m",
        default=None,
        help="Save markdown report to file",
    )
    p.add_argument(
        "--voice-refs",
        default="ljspeech",
        help=f"Comma-separated voice refs (default: ljspeech). Available: {','.join(VOICE_REFS.keys())}",
    )
    p.add_argument(
        "--mlx-impls",
        default=None,
        help=f"Comma-separated MLX implementations (default: all available). Available: {','.join(MLX_IMPLEMENTATIONS.keys())}",
    )
    return p.parse_args()


def main():
    args = parse_args()

    models = [m.strip() for m in args.models.split(",")]
    prompt_names = [p.strip() for p in args.prompts.split(",")]

    if args.devices:
        devices = [d.strip() for d in args.devices.split(",")]
    else:
        devices = []
        if torch.backends.mps.is_available():
            devices.append("mps")
        if torch.cuda.is_available():
            devices.append("cuda")
        # Check for MLX availability
        try:
            import mlx.core
            devices.append("mlx")
        except ImportError:
            pass
        devices.append("cpu")

    if args.mlx_impls:
        mlx_impls = [impl.strip() for impl in args.mlx_impls.split(",")]
    else:
        mlx_impls = [name for name, path in MLX_IMPLEMENTATIONS.items() if path.exists()]
        if not mlx_impls:
            mlx_impls = ["claude"]

    voice_ref_names = [v.strip() for v in args.voice_refs.split(",")]

    # Validate prompts
    for name in prompt_names:
        if name not in TEST_PROMPTS:
            print(f"Unknown prompt: {name}. Available: {list(TEST_PROMPTS.keys())}")
            sys.exit(1)
    for name in voice_ref_names:
        if name not in VOICE_REFS:
            print(f"Unknown voice ref: {name}. Available: {list(VOICE_REFS.keys())}")
            sys.exit(1)

    report = BenchmarkReport(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        system=get_system_info(),
    )

    print(f"TADA Benchmark Suite")
    print(f"{'='*70}")
    print(f"  Models:     {', '.join(models)}")
    print(f"  Devices:    {', '.join(devices)}")
    if "mlx" in devices:
        print(f"  MLX impls:  {', '.join(mlx_impls)}")
    print(f"  Voice refs: {', '.join(voice_ref_names)}")
    print(f"  Prompts:    {', '.join(prompt_names)} ({sum(len(TEST_PROMPTS[p]) for p in prompt_names)} total chars)")
    print(f"  System:     {report.system.get('chip', 'unknown')} / {report.system.get('ram_total_gb', '?')} GB RAM")

    for model_id in models:
        for device in devices:
            for voice_ref_name in voice_ref_names:
                try:
                    if device == "mlx":
                        for implementation in mlx_impls:
                            bm = benchmark_model_mlx(model_id, prompt_names, voice_ref_name, implementation=implementation)
                            report.benchmarks.append(asdict(bm))
                    else:
                        bm = benchmark_model(model_id, device, prompt_names, voice_ref_name)
                        report.benchmarks.append(asdict(bm))
                except Exception as e:
                    print(f"\n  FAILED: {model_id} on {device} with {voice_ref_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    report.benchmarks.append(asdict(ModelBenchmark(
                        model_id=model_id,
                        device=device,
                        implementation="baseline" if device != "mlx" else "unknown",
                        dtype=str(torch.bfloat16 if device != "cpu" else torch.float32),
                        total_params=0,
                        param_memory_mb=0.0,
                        voice_ref=voice_ref_name,
                        error=str(e),
                    )))

    # Print summary
    # Convert back for printing (report is still a dataclass)
    print_summary(report)

    # Save JSON
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(asdict(report), indent=2))
        print(f"\n  JSON results saved to {output_path}")

    # Save markdown
    if args.markdown:
        md_path = Path(args.markdown)
        md_path.write_text(generate_markdown(report))
        print(f"  Markdown report saved to {md_path}")

    # Always save to benchmarks/ directory
    bench_dir = Path(__file__).parent / "benchmarks"
    bench_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chip = report.system.get("chip", "unknown").replace(" ", "_").replace("(", "").replace(")", "")
    ram = report.system.get("ram_total_gb", "unknown")

    json_path = bench_dir / f"{timestamp}_{chip}_{ram}GB.json"
    json_path.write_text(json.dumps(asdict(report), indent=2))

    md_path = bench_dir / f"{timestamp}_{chip}_{ram}GB.md"
    md_path.write_text(generate_markdown(report))

    print(f"\n  Results archived to benchmarks/")
    print(f"    {json_path.name}")
    print(f"    {md_path.name}")


if __name__ == "__main__":
    main()
