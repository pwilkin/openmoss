# openmoss-ggml

A standalone C++/GGML port of [MOSS-TTS-Delay](https://huggingface.co/OpenMOSS-Team/MOSS-TTS) — Qwen3-8B
language backbone + 32 RVQ audio codebooks + a 1.6B pure-transformer audio codec, all packaged as
a single **GGUF** file.

Two operating modes:

- **Client mode (`moss-tts-cli`)** — one-shot generation. Loads the model, synthesizes one
  utterance, writes a WAV, exits.
- **Server mode (`moss-tts-server`)** — keeps the model resident in VRAM and serves an HTTP API
  for repeated low-latency generations.

Both modes support the full MOSS-TTS-Delay feature set: plain TTS, voice cloning from a reference
WAV, voice generation from a text description, and continuation.

## Status

**Early scaffolding.** No working build yet. See `docs/STATUS.md` for the current task tracker
mapping; tasks live alongside this commit and progress incrementally:

1. ✅ Investigate codec architecture
2. ⏳ Project scaffolding (this commit)
3. ⏳ GGUF converter (HF → single GGUF)
4. ⏳ Core inference (Qwen3 + 32 audio heads + delay-pattern sampling) — emits codes only
5. ⏳ Audio codec decoder in GGML (codes → 24 kHz waveform)
6. ⏳ Voice cloning (codec encoder for reference audio)
7. ⏳ CLI binary
8. ⏳ Server binary + thin HTTP client
9. ⏳ Quantization validation (Q8\_0, Q4\_K\_M)

## Architecture

```
                  ┌────────────────────────────────────────────┐
   text  ────►    │ BPE tokenizer (Qwen3, 155 648 vocab)        │
   ref.wav ──►    │ Codec encoder (audio→32×T_a codes)         │ — voice cloning
                  └────────────────────────────────────────────┘
                            │
                  ┌─────────▼──────────────────────────────────┐
                  │ Embedding stack:                           │
                  │   text emb  +  Σᵢ audio_emb_i(code_i)      │  (33 tables, 4096 dim)
                  └─────────┬──────────────────────────────────┘
                            │ inputs_embeds (S, 4096)
                  ┌─────────▼──────────────────────────────────┐
                  │ Qwen3-8B backbone (libllama, GGUF)         │
                  │   36 layers, hidden=4096, GQA 32/8         │
                  └─────────┬──────────────────────────────────┘
                            │ hidden_state (4096)
                  ┌─────────▼──────────────────────────────────┐
                  │ 33 LM heads (1 text + 32 audio×1025)       │  (GGML matmul, GPU)
                  └─────────┬──────────────────────────────────┘
                            │ logits
                  ┌─────────▼──────────────────────────────────┐
                  │ Delay-pattern state machine + sampling     │  (CPU, deterministic)
                  └─────────┬──────────────────────────────────┘
                            │ codes (T_a × 32)
                  ┌─────────▼──────────────────────────────────┐
                  │ Codec decoder (4-stage pure transformer)   │  (GGML)
                  │   12.5 Hz → 24 000 Hz (×1920 upsample)     │
                  └─────────┬──────────────────────────────────┘
                            ▼
                       waveform (f32, 24 kHz)
```

The Qwen3 backbone reuses **libllama** (GGML-based, supports all standard quantizations and the
CUDA/CPU/Vulkan backends llama.cpp ships). Embeddings, LM heads, and the codec are pure GGML
graphs we build ourselves.

## Build

Requires a built llama.cpp tree (`libllama.so`, `libggml.so`, headers).

```bash
cmake -B build \
    -DLLAMA_CPP_DIR=/devel/tools/llama.cpp \
    -DGGML_CUDA=ON
cmake --build build -j
```

Outputs:
- `build/bin/moss-tts-cli`
- `build/bin/moss-tts-server`

## Convert weights

```bash
pip install safetensors numpy huggingface_hub
python scripts/convert_hf_to_gguf.py \
    --moss-tts OpenMOSS-Team/MOSS-TTS \
    --codec    OpenMOSS-Team/MOSS-Audio-Tokenizer \
    --output   weights/moss-tts.gguf \
    --dtype    f16
```

Produces a single GGUF that contains:
- The Qwen3-8B language backbone (all standard `blk.*.attn_*`, `blk.*.ffn_*`, etc.)
- 32 audio embedding tables: `moss.audio_embed.{0..31}.weight` (1025 × 4096)
- 33 LM heads: `output.weight` (text — typically tied to `token_embd`) and
  `moss.audio_head.{0..31}.weight` (1025 × 4096)
- Codec encoder, RVQ codebooks, codec decoder under `moss.codec.*`
- KV metadata: `moss.n_vq`, `moss.audio_vocab_size`, `moss.audio.*_token_id`,
  `moss.sampling_rate`, `moss.frame_rate`, `moss.downsample_rate`.

Quantize the same way you would any GGUF (`llama-quantize` from llama.cpp); the codec tensors are
flagged in metadata so they stay F16 even when the backbone is heavily quantized.

## Use

### CLI

```bash
moss-tts-cli --model moss-tts.gguf \
             --text "Hello, world!" \
             --reference voice.wav \
             --output out.wav
```

### Server

```bash
moss-tts-server --model moss-tts.gguf --host 0.0.0.0 --port 8080
# then:
curl -X POST http://localhost:8080/v1/tts \
     -H 'Content-Type: application/json' \
     -d '{"text":"Hello!","reference_id":"alice"}' \
     --output out.wav
```

## License

Apache-2.0, matching upstream MOSS-TTS.
