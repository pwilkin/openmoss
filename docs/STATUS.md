# Implementation status

| #  | Task                                                  | Status               | Notes |
|----|-------------------------------------------------------|----------------------|-------|
| 1  | Investigate codec architecture                        | done                 | 1.6B pure-transformer, 4 stages × downsample 240/2/2/2, RLFQ 32×1024 |
| 2  | Project scaffolding (CMake, dirs, headers, stubs)     | done                 | builds clean against `/devel/tools/llama.cpp` |
| 3  | GGUF converter (HF → backbone GGUF + sidecar GGUF)    | **done & validated** | produces `moss-tts.gguf` (15.32 GB) + `moss-tts.extras.gguf` (4.0 GB) |
| 4  | Core C++ inference (LM side)                          | **done & validated** | backbone via libllama, audio embeds + LM heads as GGML, delay sampling, audio code extraction |
| 5  | Codec decoder in GGML                                 | **done & validated** | 68 transformer layers + RLFQ + 4 patched upsamples, end-to-end on CUDA |
| 6  | Voice cloning (codec encoder)                         | **done & validated** | encoder mirrors decoder; round-trip envelope correlation 1.000 against the original WAV |
| 7  | CLI binary                                            | **done**             | runs end-to-end on real weights and writes a 24 kHz mono WAV |
| 8  | Server binary                                         | **done**             | cpp-httplib v0.18.7 vendored; `GET /health`, `GET /info`, `POST /tts` (JSON in, audio/wav out, optional `reference_wav_b64` for cloning) |
| 9  | Quantization validation                               | **done**             | `llama-quantize --token-embedding-type f16 ... Q8_0` → ~8.7 GB; Q4\_K\_M → ~4.7 GB |

## End-to-end demo

```bash
$ ./build/moss-tts-cli \
    --model /mnt/win/k/models/openmoss/moss-tts-q8_0.gguf \
    --text "The quick brown fox jumps over the lazy dog, and once again proves \
that pangram-style text remains the simplest way to exercise a speech \
synthesis system." \
    --max-new-tokens 600 --n-gpu-layers 99 --output /tmp/out.wav
…
[generate] prompt_len = 95 tokens
[generate] prefill done in 0.04s
[generate] stop at step 161
[generate] generated 161 steps in 3.81s (42.2 tok/s)
[generate] 127 audio frames extracted
[generate] codec decode produced 243840 samples (10.16s of audio) in 0.20s
wrote 243840 samples (10.16s) to /tmp/out.wav
```

10 seconds of synthesised speech in 4 seconds wall-clock on a single 16 GB GPU
(RTX 5060 Ti). The codec graph runs in 0.2 s for 10 s of audio (~50× real-time).

## Server usage

```bash
$ ./build/moss-tts-server \
    --model /mnt/win/k/models/openmoss/moss-tts-q8_0.gguf \
    --port 18080 &
[server] model loaded; codec=on
[server] listening on http://127.0.0.1:18080

$ curl -s http://127.0.0.1:18080/info | jq .
{
  "audio_vocab_size": 1024,  "codec_loaded": true,  "codec_present": true,
  "frame_rate_hz": 12.5,     "hidden_size": 4096,   "n_vq": 32,
  "requests_served": 0,      "sampling_rate": 24000
}

$ curl -s -X POST http://127.0.0.1:18080/tts \
    -H 'Content-Type: application/json' \
    -d '{"text":"Hello from the persistent server.","max_new_tokens":300}' \
    -o out.wav -w '%{http_code} %{size_download}B %{time_total}s\n'
200 99884B 1.69s

# Voice cloning: embed the reference WAV as base64 in the JSON request.
$ python3 -c "import base64,json,sys; \
    print(json.dumps({'text': sys.argv[1], 'max_new_tokens': 600, \
        'reference_wav_b64': base64.b64encode(open(sys.argv[2],'rb').read()).decode()}))" \
    "A different sentence in the cloned voice." /tmp/out_long.wav > /tmp/req.json
$ curl -s -X POST http://127.0.0.1:18080/tts \
    -H 'Content-Type: application/json' --data @/tmp/req.json \
    -o cloned.wav -w '%{http_code} %{size_download}B %{time_total}s\n'
200 188204B 2.40s
```

## Pipeline data flow (current)

```
text  ─▶ Tokenizer (libllama BPE)
        │
        ▼ token ids (S, 33) int32, col 0 = text id, cols 1..32 = audio_pad_code
   compute_input_embeddings                        [GGML on CUDA aux backend]
   ├── get_rows(text_embed, col0)
   └── + Σᵢ get_rows(audio_embed_i, col_{i+1})  → (S, 4096) f32
        │
        ▼
   llama_decode(batch.embd) — prefill
        │
        ▼
   ╔═╣ generation loop (per step, until is_stopping) ╠═══════════╗
   ║  text_logits = llama_get_logits_ith(-1)                     ║
   ║  hidden      = llama_get_embeddings_ith(-1)                 ║
   ║  audio_logits = compute_audio_logits(hidden) [GGML CUDA]    ║
   ║  next_ids   = DelayState.step(text_logits, audio_logits)    ║
   ║  next_emb   = compute_input_embeddings(next_ids, 1)         ║
   ║  llama_decode(next_emb)                                     ║
   ╚═════════════════════════════════════════════════════════════╝
        │
        ▼ DelayState.extract_audio_codes (forward-search for first <audio_start>)
   (n_vq, T_audio) int32
        │
        ▼ codec.decode                              [GGML on CUDA aux backend]
   ┌─────────────────────────────────────────────────────────────┐
   │ ① Σᵢ q.{i}.oproj( codebook[i][codes[i]] ) → (512, T)        │
   │ ② quantizer.oproj    → (768, T)                              │
   │ ③ dec.0  iproj + 32 transformer layers @ d=1280              │
   │    └─ patch=2 upsample → (640, 2T)                           │
   │ ④ dec.2  iproj + 12 layers @ d=768   → (768, 2T)             │
   │    └─ patch=2 upsample → (384, 4T)                           │
   │ ⑤ dec.4  iproj + 12 layers @ d=768   → (768, 4T)             │
   │    └─ patch=2 upsample → (384, 8T)                           │
   │ ⑥ dec.6  iproj + 12 layers @ d=768 + oproj → (240, 8T)       │
   │    └─ patch=240 upsample → (1, 1920·T)                        │
   └─────────────────────────────────────────────────────────────┘
        │
        ▼ waveform f32 @ 24 kHz, length T_audio·1920
   write_wav_mono → out.wav
```

## What's in the codebase

| File                                           | Role                                                     |
|------------------------------------------------|----------------------------------------------------------|
| `scripts/convert_hf_to_gguf.py`                | HF → backbone GGUF + sidecar GGUF                        |
| `src/aux_internal.h`                           | Shared `Model::Aux` definition (model.cpp + codec.cpp)   |
| `src/model.cpp`                                | Two-file loader, aux backend, embed/head compute graphs  |
| `src/codec.cpp`                                | RLFQ + 4-stage transformer decoder graph                 |
| `src/tokenizer.cpp`                            | libllama BPE wrapper                                     |
| `src/delay.cpp`                                | DelayState + sampling (top-k/p, rep penalty, multinomial)|
| `src/pipeline.cpp`                             | Prompt builder + autoregressive generation loop + codec  |
| `src/wav.cpp`                                  | RIFF/WAVE I/O, no libsndfile dep                         |
| `src/cli/moss_tts_cli.cpp`                     | CLI entry point                                          |
| `src/server/moss_tts_server.cpp`               | HTTP server: `GET /health` + `GET /info` + `POST /tts`   |
| `third_party/cpp-httplib/httplib.h`            | Single-header HTTP library (vendored from upstream v0.18.7) |
| `tests/moss_tts_info.cpp`                      | Diagnostic: load, dump dims                              |
| `tests/moss_tts_compute.cpp`                   | Smoke test for embedding + audio-head graphs             |
| `tests/moss_codec_roundtrip.cpp`               | Codec encode→decode round-trip + per-codebook stats      |

## Notable bugs fixed during codec/encoder bring-up

1. **DelayState post-mask sentinel overflow** (`src/delay.cpp`). The pre/post
   audio mask used `INT64_MAX` for the "no delay yet" sentinel and computed
   `post = (delayed - 1) < i`. With the sentinel, that always evaluates to
   `false`, silently skipping every audio codebook during warm-up. Fixed by
   branching on the sentinel explicitly.

2. **`extract_audio_codes` searched for `audio_start OR gen_slot`** and
   landed on the *last* gen_slot row, leaving T ≈ n_vq and T_audio ≈ 0.
   Fixed to search backwards for `audio_start` only. (Backward search is
   correct for both vanilla TTS and voice cloning: the trailing
   `<audio_start>` of the assistant prompt is always the one that bounds
   the generated segment, while any earlier `<audio_start>` rows belong to
   user-supplied reference audio.)

Both bugs only surface once the codec is actually invoked, which is why
they slipped past the LM-only smoke tests in tasks 3-4.

## Open issues / future work

- **Multi-GPU layout.** Both the CLI and the server pin the entire model
  (libllama backbone + aux backend for codec/embed/heads) to a single GPU
  via `LLAMA_SPLIT_MODE_NONE`. The picked device defaults to the GPU with
  the most free VRAM at load time; override with `--main-gpu N` (index into
  the ggml GPU device list, same convention as libllama). This avoids the
  cross-device OOM that happens when libllama splits the backbone across
  GPUs while the aux backend lands on whichever device it picks first.
- **Backbone fits at FP16 only on >24 GB GPUs.** Use `llama-quantize` with
  `--token-embedding-type f16` so CUDA's `ggml_get_rows` can still index the
  embedding table (it doesn't support quantised src0).
- **Per-step graph rebuild** in `compute_input_embeddings` /
  `compute_audio_logits`. For codec decode the graph is rebuilt only once
  (called at the end of generation), so this is fine.
- **Server multi-user concurrency** would need `llama_seq_id` plumbed through
  the backbone batch and the `m_aux->galloc` made per-request. Today a single
  `std::mutex` serializes all generations, which keeps the implementation
  simple at the cost of throughput under concurrent load. Verified with two
  parallel curls: r1 finishes at ~1.3 s, r2 at ~2.7 s = r1 + r2's own work.
- **Streaming output** for the server would require chunked codec decode +
  the `StreamCallback` typedef in `pipeline.h` (currently unused).
