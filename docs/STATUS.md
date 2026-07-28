# Implementation status

openmoss is a standalone C++/GGML runtime for the MOSS audio models — CUDA,
Vulkan and CPU, with the Qwen3 backbone driven through libllama and everything
else written as GGML graphs.

Three model families are supported, and one binary handles all of them: the
architecture is recorded as `moss.architecture` in the GGUF and dispatched on at
run time.

| Family | Arch id | Shape | Output |
|---|---|---|---|
| MOSS-TTS / v1.5 / VoiceGenerator | `moss_tts_delay` | Qwen3-8B, 32 RVQ codebooks on a delay pattern, MOSS-Audio-Tokenizer v1 | 24 kHz mono |
| MOSS-TTS-Local-Transformer-v1.5 | `moss_tts_local` | Qwen3-4B, 12 codebooks emitted per frame by a 1-layer depth transformer, MOSS-Audio-Tokenizer v2 | 48 kHz stereo |
| MOSS-SoundEffect-v2.0 | `moss_soundeffect` | **not autoregressive** — a 30-block Wan-style DiT sampled by flow matching, Qwen3-1.7B text encoder, continuous DAC VAE | 48 kHz mono |

Published weights: [ilintar/moss-tts-gguf](https://huggingface.co/ilintar/moss-tts-gguf),
[ilintar/moss-tts-local-gguf](https://huggingface.co/ilintar/moss-tts-local-gguf),
[ilintar/moss-soundeffect-gguf](https://huggingface.co/ilintar/moss-soundeffect-gguf).

## What works

| Area | Status |
|---|---|
| GGUF converter (all three families) | done — backbone GGUF + `moss.*` sidecar, optional `--sidecar-dtype q8_0` |
| Delay-pattern generation + codec v1 | done, validated |
| MOSS-TTS-Local: depth transformer + codec v2 | done, validated exactly against the reference |
| Voice cloning (codec encoder) | done — round-trip envelope correlation 0.995 |
| Continuation mode (`ref_text`) | done for `moss_tts_local`, prompt ids exact |
| MOSS-SoundEffect: DiT, DAC VAE, flow matching | done, verified seam by seam |
| CLI | one binary, dispatches on architecture |
| HTTP server | `/tts`, `/sfx`, `/v1/audio/speech`, `/v1/models`, `/v1/audio/voices`, wav + pcm |
| WebUI | served at `/` by the server |
| Quantization | backbone via `llama-quantize`; sidecar via the converter |

## Not done

- **Streaming.** The LM half is already incremental; the codec is not. See
  [STREAMING.md](STREAMING.md) — it needs a per-stage KV cache, and the
  measured reason why chunking cannot substitute is documented there.
- **Continuation mode for `moss_tts_delay`.** Different slot layout, no verified
  reference locally; `ref_text` is rejected for that family rather than silently
  ignored.
- **Server concurrency.** One mutex serialises generation. Multi-user would need
  `llama_seq_id` plumbed through the batch and a per-request allocator.

## How correctness is established

Generated audio is a poor test — sampling and 100 solver steps both amplify tiny
numerical differences — so everything is verified at *seams*, feeding both the
port and the PyTorch reference identical input bytes.

| Probe | Checks |
|---|---|
| `moss-prompt-probe` | prompt grid ids vs the reference processor. No tolerance: plain, voice-clone and continuation layouts all match **every integer** |
| `moss-local-probe` | the depth transformer from a fixed hidden state |
| `moss-sfx-probe` | the DiT at 8 seams, at 16 and 1500 frames |
| `moss-sfx-vae-probe` | the DAC decoder at 10 seams, plus a chunking check that must be bit-identical |
| `moss-codec-causality` | whether the codec decoder is causal, and how much history a chunk needs |
| `moss-codec-roundtrip` | encode→decode envelope correlation, compared first half vs second half |

Two lessons worth carrying to any similar port:

**Establish the f16 budget before judging any number.** Rerun the reference with
its weights round-tripped through f16 (`p.copy_(p.half().float())`). For the DiT
that floor is ~2e-4 at every seam and it does *not* accumulate with depth, so
anything much above it is a bug rather than "f16 being lossy".

**ggml's Vulkan backend accumulates matmuls in f16 by default.** Any node left at
`GGML_PREC_DEFAULT` costs ~5e-3 to 1e-2 relative — far more than f16 weights.
This was found three separate times: in the DiT, in libllama's own graph (fixed
with `--vk-f32`, which sets `GGML_VK_DISABLE_F16`), and in the codec. It hides
because matrix-*vector* products take a different, f32-accumulating path, so
anything computed one token at a time looks perfect while every batched seam is
uniformly wrong.

## Performance

Radeon 8060S (Strix Halo, Vulkan), 123 GB unified memory.

| | Rate |
|---|---|
| MOSS-TTS-Delay backbone, Q8 | ~22 tok/s |
| MOSS-TTS-Local backbone, Q8 | ~22 frames/s (1.74x real time) |
| Codec decode | ~30x real time (40 s of audio in 1.10 s) |
| MOSS-SoundEffect, 100 steps | ~90 s regardless of duration — the DiT always denoises 30 s |
| MOSS-SoundEffect VAE decode | 0.99 s for 3 s of audio, 7.7 s for 30 s |

`--cfg-scale 1` halves SoundEffect's solve by skipping the unconditional branch.
`--vk-f32` is recommended on Vulkan and costs nothing measurable.
