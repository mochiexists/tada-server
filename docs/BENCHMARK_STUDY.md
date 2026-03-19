# TADA TTS Benchmark Study

> Performance and memory profiling of HumeAI TADA text-to-speech models on Apple Silicon.

**Date:** 2026-03-18
**System:** Apple M4 Pro, 48 GB unified memory
**PyTorch:** 2.7.1 (MPS backend)
**Models tested:** `HumeAI/tada-1b`, `HumeAI/tada-3b-ml`

---

## Key Terms

- **RTF (Real-Time Factor):** Wall-clock seconds per second of audio output. RTF 1.0x = realtime. RTF 0.5x = 2x faster than realtime. Lower is better.
- **MPS:** Metal Performance Shaders — Apple's GPU compute backend for PyTorch on Apple Silicon.
- **MPS Driver:** Total GPU memory allocated by the Metal driver (includes allocator overhead beyond model weights).

---

## 1. Model Comparison: 1B vs 3B

Both models use bfloat16 on MPS. The encoder (`HumeAI/tada-codec`) is shared and uses ~3 GB MPS.

### Model Specifications

| | tada-1b | tada-3b-ml |
|---|------:|------:|
| Parameters | 1,753,588,834 | 4,225,756,258 |
| Weight memory (bf16) | 3.49 GB | 8.09 GB |
| Weight memory (f32/CPU) | 6.53 GB | 15.74 GB |
| Architecture | LlamaForCausalLM | LlamaForCausalLM |

### Generation Speed (MPS, default LJSpeech voice)

| Prompt | Chars | tada-1b Audio | tada-1b RTF | tada-3b Audio | tada-3b RTF |
|--------|------:|--------:|----:|--------:|----:|
| short | 25 | 1.4s | 1.9x | 1.5s | 2.5x |
| medium | 185 | 13.5s | **0.5x** | 12.3s | 1.0x |
| long | 500 | 30.6s | **0.7x** | 30.6s | 1.0x |
| very_long | 880 | 53.9s | **0.7x** | 53.9s | 1.7x |

**Finding:** The 1B model is faster than realtime on medium+ prompts (0.5-0.7x RTF). The 3B hits ~1.0x on medium/long but degrades on very long text. Audio output length scales linearly with input text length for both models.

### Peak Memory (MPS)

| | tada-1b | tada-3b-ml |
|---|------:|------:|
| MPS Allocated | 6.30 GB | 10.90 GB |
| MPS Driver (peak) | 10.64 GB | 14.25 GB |
| Process RSS | ~2-4 GB | ~1-3 GB |

### CPU Fallback Performance

| | tada-1b CPU | tada-3b CPU |
|---|------:|------:|
| Peak RSS | 4.34 GB | 9.39 GB |
| Generation (medium) | 20.3s | 224.5s |
| RTF (medium) | 4.75x | **55.3x** |

**Finding:** CPU mode uses float32 (2x weight memory). The 1B is usable on CPU (4.75x RTF). The 3B on CPU is impractical at 55x realtime.

---

## 2. Voice Reference Duration Impact

Tested with Grimes voice samples at 10s, 16s, and 28s durations on MPS.

### tada-1b — Voice Ref Duration Scaling

| Voice Ref | Prompt Encode | MPS Driver Peak | Medium RTF | Long RTF |
|-----------|------:|------:|----:|----:|
| grimes-10s | 1.6s | 11.21 GB | 0.6x | 0.8x |
| grimes-16s | 2.1s | 14.24 GB | **0.5x** | **0.6x** |
| grimes-28s | 3.2s | 18.98 GB | 0.7x | 3.2x |

### tada-3b-ml — Voice Ref Duration Scaling

| Voice Ref | Prompt Encode | MPS Driver Peak | Medium RTF | Long RTF |
|-----------|------:|------:|----:|----:|
| grimes-10s | 1.3s | 16.16 GB | 1.4x | 1.2x |
| grimes-16s | 4.4s | 19.23 GB | 1.2x | 1.4x |
| grimes-28s | 11.4s | **23.89 GB** | 1.7x | 1.6x |

### Memory Cost per Second of Voice Reference

| | MPS Driver Increase |
|---|------:|
| 10s → 16s | +3.0 GB (~0.5 GB/s) |
| 16s → 28s | +4.7 GB (~0.4 GB/s) |
| 10s → 28s | +7.7 GB (~0.4 GB/s) |

**Findings:**
- Each additional second of voice reference costs ~0.4-0.5 GB of MPS driver memory.
- **16s is the sweet spot** for tada-1b: best RTF (0.5x), manageable memory (14.2 GB).
- 28s reference causes significant slowdown on tada-3b (prompt encoding jumps from 1.3s to 11.4s).
- For 16 GB machines, the maximum safe voice reference is ~10s with tada-1b.

---

## 3. Prompt Length Scaling

Generation time and audio output scale roughly linearly with input text length. Memory usage during generation is flat — it's dominated by model weights and voice prompt, not by input text.

### tada-1b (MPS, grimes-10s voice)

| Prompt | Input Chars | Input Words | Output Audio | Gen Time | RTF |
|--------|------:|------:|------:|------:|------:|
| short | 25 | 5 | 1.4s | 2.6s | 1.9x |
| medium | 185 | 35 | 13.5s | 6.2s | 0.5x |
| long | 500 | 99 | 30.6s | 20.9s | 0.7x |
| very_long | 880 | 163 | 53.9s | 34.8s | 0.7x |

**Finding:** Short prompts have higher RTF due to fixed overhead. At medium+ lengths, the 1B model consistently runs faster than realtime.

---

## 4. Device Recommendations

### By Hardware

| Machine | RAM | Recommended Config |
|---------|----:|---|
| Mac Mini M4 (16 GB) | 16 GB | tada-1b, MPS, 10s voice ref (peak ~11 GB) |
| MacBook Pro M4 Pro (48 GB) | 48 GB | tada-3b-ml, MPS, 16s voice ref (peak ~19 GB) |
| Any Mac (conservative) | 16 GB | tada-1b, CPU (peak ~4.3 GB RSS, 4.75x RTF) |
| Linux/CUDA GPU | 12+ GB | tada-1b bf16; 24+ GB for tada-3b |

### By Use Case

| Use Case | Config | Expected Performance |
|----------|--------|---------------------|
| Realtime voice chat | tada-1b, MPS, 10s ref | 0.5-0.7x RTF |
| High quality, offline | tada-3b-ml, MPS, 10s ref | 1.0-1.4x RTF |
| Voice cloning (best quality) | tada-3b-ml, MPS, 16s ref | 1.2-1.4x RTF |
| Low-memory server | tada-1b, CPU | 4.75x RTF, 4.3 GB RAM |

---

## 5. MLX Conversion Status

As of March 2026, **no MLX port of TADA exists**. Key findings:

- The Llama backbone would convert easily via `mlx-lm` (standard architecture).
- **4 custom components** need porting: tada-codec encoder, temporal aligner, flow-matching acoustic head, acoustic decoder.
- The flow-matching head is the hardest — requires iterative ODE solving. However, F5-TTS (also flow-matching) has been ported to MLX successfully.
- [Voicebox](https://github.com/jamiepine/voicebox) claims TADA+MLX support but actually runs TADA through PyTorch (CPU on macOS) and only uses MLX for Qwen3-TTS.
- MLX supports iOS/iPadOS/visionOS via [mlx-swift](https://github.com/ml-explore/mlx-swift), so a successful port would enable on-device TTS on iPhone.
- Expected improvement from MLX: 4-5x faster inference on Apple Silicon based on comparable model ports.

### Quantization Path

TADA inherits from `LlamaForCausalLM` → `PreTrainedModel`, so HuggingFace quantization (bitsandbytes int8/int4, GPTQ, AWQ) is supported on CUDA. However:
- `bitsandbytes` is CUDA-only (no MPS support).
- The theoretical path: quantize on CUDA → export → convert to MLX format. This would roughly halve memory usage.

---

## 6. Methodology

All benchmarks use `benchmark.py` from this repository. Each run:

1. Loads the encoder (`HumeAI/tada-codec`) and measures memory
2. Loads the target model and measures memory + parameter count
3. Encodes a voice reference prompt and measures memory + time
4. Generates speech for each test prompt, measuring wall-clock time, output duration, and memory
5. Computes RTF (generation_time / audio_duration)
6. Archives results as JSON + Markdown in `benchmarks/`

Memory is measured via `psutil` (RSS) and PyTorch's MPS/CUDA allocator APIs. Runs are sequential (no concurrent generation). Model weights are cached locally after first download.

### Reproducibility

```bash
# Setup
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install psutil

# Full benchmark
python benchmark.py

# Specific configuration
python benchmark.py --models HumeAI/tada-1b --devices mps --voice-refs grimes-10s,grimes-16s --prompts medium,long
```

### Test Prompts

| Name | Chars | Content |
|------|------:|---------|
| short | 25 | "Hello, how are you today?" |
| medium | 185 | Quick brown fox + TTS system test |
| long | 500 | Hitchhiker's Guide excerpt |
| very_long | 880 | Gettysburg Address excerpt |

---

## Raw Benchmark Data

Full JSON results with per-stage memory snapshots are in `benchmarks/`.

---

*Generated from [tada-server](https://github.com/mochiexists/tada-server) benchmark suite.*
