#!/usr/bin/env python3
"""Generate and save audio samples for voice cloning quality comparison.

Generates all combinations of:
  - Models: tada-1b, tada-3b-ml
  - Voice refs: grimes-10s, grimes-16s, grimes-28s
  - Prompts: short, medium, long

Files are named for easy sorting:
  {model}_{voice_ref}_{prompt}.wav
  e.g. 1b_ref-10s_medium.wav, 3b_ref-28s_short.wav
"""

import gc
import json
import time
from pathlib import Path

import torch
import torchaudio
from huggingface_hub import hf_hub_download

from tada.modules.encoder import Encoder
from tada.modules.tada import TadaForCausalLM

# Patch MPS lm_head issue
def _fixed_lm_head_forward(self, hidden_states):
    return self.lm_head(hidden_states)
TadaForCausalLM._lm_head_forward = _fixed_lm_head_forward

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_DIR = Path(__file__).parent / "output_samples"

MODELS = [
    ("HumeAI/tada-1b", "1b"),
    ("HumeAI/tada-3b-ml", "3b"),
]

PROMPTS = {
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
        "that no one should ever have left the oceans."
    ),
}

VOICE_REFS = {
    "ref-10s": {
        "path": "../../samples/grimes-10s.wav",
        "transcript": (
            "Usually people think of an AI as like an evil robot or, you know, "
            "like when you think of an artificial general intelligence, "
            "I think most people immediately jump to Terminator."
        ),
    },
    "ref-16s": {
        "path": "../../samples/grimes-16s.wav",
        "transcript": (
            "Usually people think of an AI as like an evil robot or, you know, "
            "like when you think of an artificial general intelligence, "
            "I think most people immediately jump to Terminator. "
            "But I actually think AI could be the most"
        ),
    },
    "ref-28s": {
        "path": "../../samples/grimes-28s.wav",
        "transcript": (
            "I have a proposition for the communists. So typically most of the "
            "communists I know are not big fans of AI. But if you think about it, "
            "AI is actually the fastest path to communism. Because if we get AI and "
            "robots and stuff, eventually everything can be automated, nobody has to "
            "work, and the government can just pay everyone a universal basic income."
        ),
    },
}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}\n")

    # Load encoder once (shared across models)
    print("Loading encoder (HumeAI/tada-codec)...")
    encoder = Encoder.from_pretrained("HumeAI/tada-codec").to(DEVICE)

    # Pre-encode all voice prompts (shared across models)
    prompts_cache = {}
    for ref_name, ref_info in VOICE_REFS.items():
        ref_path = str(Path(__file__).parent / ref_info["path"])
        ref_wav, ref_sr = torchaudio.load(ref_path)
        ref_dur = ref_wav.shape[-1] / ref_sr
        ref_wav = ref_wav.to(DEVICE)
        print(f"  Encoding {ref_name} ({ref_dur:.1f}s)...", end="", flush=True)
        t0 = time.time()
        prompts_cache[ref_name] = encoder(ref_wav, text=[ref_info["transcript"]], sample_rate=ref_sr)
        print(f" {time.time()-t0:.1f}s")
        del ref_wav

    results = []

    for model_id, model_label in MODELS:
        print(f"\n{'='*60}")
        print(f"Loading model: {model_id}")
        print(f"{'='*60}")
        model = TadaForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(DEVICE)

        for ref_name in VOICE_REFS:
            prompt = prompts_cache[ref_name]
            print(f"\n  Voice ref: {ref_name}")
            print(f"  {'Prompt':<8} {'Audio':>7} {'Gen Time':>9} {'RTF':>7}")
            print(f"  {'-'*8} {'-'*7} {'-'*9} {'-'*7}")

            for prompt_name, text in PROMPTS.items():
                fname = f"{model_label}_{ref_name}_{prompt_name}.wav"
                out_path = OUTPUT_DIR / fname

                t0 = time.time()
                with torch.no_grad():
                    output = model.generate(prompt=prompt, text=text)
                dt = time.time() - t0

                audio = output.audio[0].cpu()
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                audio_dur = audio.shape[-1] / 24000
                torchaudio.save(str(out_path), audio, 24000)

                rtf = dt / audio_dur if audio_dur > 0 else float("inf")
                print(f"  {prompt_name:<8} {audio_dur:>6.1f}s {dt:>8.1f}s {rtf:>6.2f}x  → {fname}")

                results.append({
                    "file": fname,
                    "model": model_label,
                    "voice_ref": ref_name,
                    "prompt": prompt_name,
                    "audio_duration_s": round(audio_dur, 2),
                    "generation_time_s": round(dt, 2),
                    "rtf": round(rtf, 2),
                })

                del output, audio

        # Free model memory before loading next
        del model
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

    # Save results index
    index_path = OUTPUT_DIR / "_results.json"
    with open(index_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! {len(results)} samples saved to: {OUTPUT_DIR}")
    print(f"Results index: {index_path}")

    # Print comparison table
    print(f"\n{'File':<35} {'Audio':>7} {'Gen':>7} {'RTF':>7}")
    print(f"{'-'*35} {'-'*7} {'-'*7} {'-'*7}")
    for r in results:
        print(f"{r['file']:<35} {r['audio_duration_s']:>6.1f}s {r['generation_time_s']:>6.1f}s {r['rtf']:>6.2f}x")


if __name__ == "__main__":
    main()
