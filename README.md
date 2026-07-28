# openmoss-ggml

A standalone C++/[GGML](https://www.github.com/ggml-org/ggml) runtime for the **MOSS audio
models** — speech and sound effects — from one self-contained binary. The Qwen3 backbone is
hosted by **libllama** (so you get all of llama.cpp's quantizations and CUDA / CPU / Vulkan /
ROCm backends for free); the embedding stack, LM heads, codec, DiT and VAE are GGML graphs we
build ourselves.

Three model families are supported, and one binary handles all of them — the architecture is
recorded in the GGUF and dispatched on at run time:

| Family | Arch id | Shape | Output |
|---|---|---|---|
| [MOSS-TTS](https://huggingface.co/OpenMOSS-Team/MOSS-TTS) / v1.5 / VoiceGenerator | `moss_tts_delay` | Qwen3-8B, 32 RVQ codebooks on a delay pattern, MOSS-Audio-Tokenizer v1 | 24 kHz mono |
| MOSS-TTS-Local-Transformer-v1.5 | `moss_tts_local` | Qwen3-4B, 12 codebooks emitted per frame by a 1-layer depth transformer, MOSS-Audio-Tokenizer v2 | 48 kHz **stereo** |
| MOSS-SoundEffect-v2.0 | `moss_soundeffect` | **not autoregressive** — a 30-block Wan-style DiT sampled by flow matching, Qwen3-1.7B text encoder, continuous DAC VAE | 48 kHz mono |

Everything is verified seam-by-seam against the PyTorch reference rather than by ear; see
[`docs/STATUS.md`](docs/STATUS.md) for the method and the numbers.

Two entry points:

- **`moss-tts-cli`** — one-shot synthesis. Loads the model, synthesizes one utterance, writes a
  WAV, exits.
- **`moss-tts-server`** — keeps the model resident and exposes an HTTP API for repeated
  generations (TTS, sound effects, voice cloning), with optional **streaming**. Also serves a
  small browser **WebUI** with a per-browser history of generated audio (IndexedDB). See
  [WebUI](#webui).

On a single 16 GB GPU (RTX 5060 Ti) the Q8\_0 backbone produces ~10 s of speech in ~4 s of
wall-clock; the codec runs at ~50× real-time. See [`docs/STATUS.md`](docs/STATUS.md) for a
detailed feature matrix, pipeline diagram, and benchmark numbers.

Prebuilt binaries for Linux and Windows (Vulkan, ROCm, CUDA) are published on the
[releases page](../../releases) — each archive bundles the llama.cpp libraries it was built
against, so no separate llama.cpp build is needed. Serves as the `openmoss` backend of
[Lemonade](https://github.com/lemonade-sdk/lemonade).

## What's new in 0.2.0

Two entire model families and streaming, on top of what 0.1.x shipped for MOSS-TTS-Delay.

**New model families**

- **MOSS-TTS-Local-Transformer-v1.5** (`moss_tts_local`) — 48 kHz stereo. A 1-layer depth
  transformer emits all 12 codebooks per frame instead of the delay pattern, backed by
  MOSS-Audio-Tokenizer v2 (6 stages per side, per-stage sliding windows). Validated exactly
  against the reference.
- **MOSS-SoundEffect-v2.0** (`moss_soundeffect`) — text → sound effects. A Wan-style DiT sampled
  by flow matching through a continuous DAC VAE, verified seam by seam. Published GGUFs:
  [`ilintar/moss-soundeffect-gguf`](https://huggingface.co/ilintar/moss-soundeffect-gguf).

**Streaming** — audio is emitted as it is generated rather than after the whole utterance. The
codec decoder keeps a per-stage K/V cache, so nothing is recomputed. `--stream` on the CLI
(`--stream-out -` pipes raw PCM to a player), `"stream": true` on the server. Cuts time to first
audio from 2.54 s to 0.57 s on a 4.8 s utterance. See [`docs/STREAMING.md`](docs/STREAMING.md).

**Also new**

- **Voice cloning** through the codec *encoder*, and **continuation mode** (`ref_text`) for
  MOSS-TTS-Local, where the model carries on speaking from the reference instead of imitating it.
- **Server**: `/sfx`, `pcm` responses, `/v1/models`, `/v1/audio/voices`, and full option parity
  between the native and OpenAI-compatible routes.
- **WebUI**: a MOSS-SoundEffect mode.
- **Q8\_0 sidecar quantization** for the projection matrices (`--sidecar-dtype q8_0`).
- `--min-audio-frames` / `--max-audio-frames` to bound a segment when a model will not stop.

**Fixes worth knowing about**

- **A long `reference_wav_b64` could kill the server.** libllama aborts the process when a batch
  exceeds `n_batch`, and the whole prompt was submitted as one; a voice-clone reference costs one
  token per codec frame, so ~5.5 minutes of it was enough. The prefill is now chunked.
- **The codec accumulated its matmuls in f16** on Vulkan, costing ~1e-2 relative on the decoded
  waveform — 16× worse than f32 accumulation. Now forced to f32 (`--vk-f32` covers the backbone).
- The tree did not build under **clang** at all, which also blocked `-DGGML_HIP=ON`.

## Quick start

```bash
# 1. Build (llama.cpp is bundled as a submodule and built in-tree)
git clone --recurse-submodules https://github.com/pwilkin/openmoss
cd openmoss
cmake -B build -DGGML_CUDA=ON     # or -DGGML_VULKAN=ON / -DGGML_HIP=ON / nothing for CPU
cmake --build build -j

# 2. Convert weights once (or download a pre-built GGUF — see "Convert weights")
python scripts/convert_hf_to_gguf.py \
    --moss-tts OpenMOSS-Team/MOSS-TTS \
    --codec    OpenMOSS-Team/MOSS-Audio-Tokenizer \
    --output   weights/moss-tts.gguf

# 3. (Optional) quantize the backbone — keep the embedding table as f16.
#    llama-quantize comes from any llama.cpp build/release (the in-tree
#    submodule build skips llama.cpp's tools).
llama-quantize \
    --token-embedding-type f16 \
    weights/moss-tts.gguf weights/moss-tts-q8_0.gguf Q8_0

# 4. Synthesize
./build/moss-tts-cli \
    --model weights/moss-tts-q8_0.gguf \
    --text  "Hello, world!" \
    --output out.wav
```

## Build

llama.cpp is **bundled as a git submodule** (`third_party/llama.cpp`) and built in-tree —
no separate llama.cpp build or `-DLLAMA_CPP_DIR` flag is needed. Pick the GPU backend with
the usual upstream flags (`-DGGML_CUDA=ON`, `-DGGML_VULKAN=ON`, `-DGGML_HIP=ON`, … — none
for CPU-only).

Prerequisites:

- The submodules checked out: `git clone --recurse-submodules …` (or, in an existing
  checkout, `git submodule update --init --recursive`).
- CMake ≥ 3.18, a C++17 compiler (GCC/Clang on Linux, MSVC on Windows), and the toolkit
  for your chosen backend (CUDA toolkit, Vulkan SDK, ROCm, …).
- `nlohmann/json.hpp` and `cpp-httplib` are vendored under `third_party/`; no system deps.

### Linux

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j
```

### Windows

Open a **Developer Command Prompt for VS** (or `vcvarsall.bat x64`), then:

```cmd
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

`ws2_32` (Winsock) is linked automatically for the HTTP server.

### Outputs

Executables land in the build root; the llama.cpp/ggml shared libraries in `build/bin/`
(they resolve via RPATH on Linux — with MSVC, copy the DLLs next to the executables or
add `build\bin\Release` to `PATH`).

Linux:
- `build/moss-tts-cli`
- `build/moss-tts-server`
- diagnostics — `moss-tts-info`, `moss-tts-compute-test`, `moss-prompt-probe`,
  `moss-local-probe`, `moss-codec-roundtrip`, `moss-codec-causality`, `moss-codec-stream`,
  `moss-sfx-probe`, `moss-sfx-vae-probe`. Each checks one seam against the PyTorch reference;
  [`docs/STATUS.md`](docs/STATUS.md) says what each one proves.
- `build/bin/libllama.so`, `build/bin/libggml*.so`

Windows (MSVC):
- `build/Release/moss-tts-cli.exe`, `build/Release/moss-tts-server.exe` (+ diagnostics)
- `build/bin/Release/llama.dll`, `ggml*.dll`

## Convert weights

**Pre-built GGUFs**, one repo per family — download the backbone *and* its `*.extras.gguf`
sidecar, then pass the backbone to `--model` (the sidecar is auto-located alongside it):

| Family | Repo |
|---|---|
| `moss_tts_delay` | [`ilintar/moss-tts-gguf`](https://huggingface.co/ilintar/moss-tts-gguf) |
| `moss_tts_local` | [`ilintar/moss-tts-local-gguf`](https://huggingface.co/ilintar/moss-tts-local-gguf) |
| `moss_soundeffect` | [`ilintar/moss-soundeffect-gguf`](https://huggingface.co/ilintar/moss-soundeffect-gguf) |

Or convert your own. The converter produces **two** GGUF files alongside each other:

- `moss-tts.gguf` — the Qwen3 backbone, pure Qwen3 layout that libllama can load directly.
- `moss-tts.extras.gguf` — the sidecar: the audio embedding tables and LM heads (32 + 33 for the
  delay family, 12 for MOSS-TTS-Local), the codec encoder / RVQ codebooks / decoder, and the
  `moss.*` KV namespace (architecture, frame rate, special token ids, …). Loaded by us, not
  libllama. For MOSS-SoundEffect the sidecar instead carries the DiT, the DAC VAE and the
  scheduler constants — there is no codec.

The family is detected from the model's `config.json`; `--arch` overrides it.

```bash
pip install safetensors numpy huggingface_hub gguf

# MOSS-TTS (delay family)
python scripts/convert_hf_to_gguf.py \
    --moss-tts OpenMOSS-Team/MOSS-TTS \
    --codec    OpenMOSS-Team/MOSS-Audio-Tokenizer \
    --output   weights/moss-tts.gguf

# MOSS-TTS-Local — 48 kHz stereo, codec v2
python scripts/convert_hf_to_gguf.py \
    --moss-tts OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
    --codec    OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 \
    --output   weights/moss-tts-local.gguf

# MOSS-SoundEffect — no codec; the DiT and VAE come from the same repo
python scripts/convert_hf_to_gguf.py \
    --moss-tts OpenMOSS-Team/MOSS-SoundEffect-v2.0 \
    --output   weights/moss-soundeffect.gguf
```

| flag | meaning |
|---|---|
| `--arch ID` | force the family instead of detecting it from `config.json` |
| `--backbone-dtype f16\|f32\|bf16` | backbone precision (default f16) |
| `--sidecar-dtype f16\|q8_0` | quantize the sidecar's projection matrices (default f16) |
| `--sidecar-only` | rewrite just the sidecar, reusing an existing backbone |
| `--llama-cpp-dir DIR` | a different llama.cpp source tree; defaults to the bundled submodule (no llama.cpp *build* required) |
| `--cache-dir` / `--scratch-dir` / `--keep-scratch` / `--skip-extract` | HF cache location, and reusing or inspecting the intermediate extracted backbone |

**Do not quantize the MOSS-SoundEffect backbone.** It is a text *encoder* whose hidden states are
consumed as a continuous conditioning vector, so there is no sampling stage to absorb the error:
Q8\_0 moves the conditioning to 2.7e-2 and the final latent to 1.8e-1. f16 against the source
bf16 is not a compromise — the measured round-trip across 2.03 B weights is 4.7e-9.

### Quantization

Quantize the backbone the same way you quantize any GGUF, but pass
`--token-embedding-type f16`: the embedding table is indexed via `ggml_get_rows`, which doesn't
support quantized `src0` on CUDA, so it must stay f16.

```bash
llama-quantize --token-embedding-type f16 \
    weights/moss-tts.gguf weights/moss-tts-q8_0.gguf Q8_0   # ~8.7 GB, validated end-to-end
llama-quantize --token-embedding-type f16 \
    weights/moss-tts.gguf weights/moss-tts-q4km.gguf Q4_K_M # ~4.7 GB
```

The sidecar is **not** quantized — keep `moss-tts.extras.gguf` next to whichever backbone you
chose. The CLI/server find the sidecar by replacing the backbone's `.gguf` suffix with
`.extras.gguf`.

## CLI usage

```bash
moss-tts-cli --model <gguf> --text "<text>" [options]
```

Common options:

| flag                  | meaning                                                            |
|-----------------------|--------------------------------------------------------------------|
| `--text "…"`          | utterance to synthesize (required)                                 |
| `--output PATH`       | output WAV (default `out.wav`, 16-bit; rate and channels come from the model — see below) |
| `--reference PATH`    | reference WAV for voice cloning — speaker is encoded and prepended |
| `--instruction "…"`   | voice / style description (voice-generation mode, no reference)    |
| `--language CODE`     | language hint (`en` / `zh` / …)                                     |
| `--tokens N`          | approximate audio length in tokens (1 s ≈ 12.5 tokens)             |
| `--max-new-tokens N`  | generation cap (default 4096)                                      |
| `--main-gpu N`        | pin model to GPU index N (default: auto-pick GPU with most free VRAM) |
| `--n-gpu-layers N`    | backbone GPU offload (default: all)                                |
| `--n-batch N`         | libllama batch size (default 512); raise if a long prompt exceeds it |
| `--n-ctx N`           | libllama context size (default 8192)                               |
| `--no-flash-attn`     | disable flash attention                                            |
| `--skip-codec`        | don't load codec tensors; emits codes only (debug, saves ~3.4 GB)  |
| `--aux-cpu`           | run audio embeds + codec on CPU (workaround for backends missing ops); backbone stays on GPU |
| `--stream [N]`        | decode incrementally, N frames per chunk (default 8; a frame is 80 ms) |
| `--stream-out PATH`   | also write each chunk as raw s16le as it arrives; `-` is stdout    |

Sample rate and channel count are properties of the loaded model, never hardcoded — they
come from `ModelDims::sampling_rate` / `n_channels`, which the converter writes into the
GGUF:

| family             | rate   | channels | codebooks (`n_vq`) |
|--------------------|--------|----------|--------------------|
| `moss_tts_delay`   | 24 kHz | mono     | 32                 |
| `moss_tts_local`   | 48 kHz | stereo   | 12                 |
| `moss_soundeffect` | 48 kHz | mono     | — (not an RVQ model) |

### Plain TTS

```bash
./build/moss-tts-cli \
    --model weights/moss-tts-q8_0.gguf \
    --text  "The quick brown fox jumps over the lazy dog." \
    --max-new-tokens 600 \
    --output out.wav
```

### Voice cloning

Pass a reference WAV with `--reference`. The codec encoder converts it to codes which get
spliced into the user-side audio block of the chat prompt, and the model continues in that
voice. The codebook count is the model's `n_vq` — 32 for the delay family, 12 for
MOSS-TTS-Local.

```bash
./build/moss-tts-cli \
    --model     weights/moss-tts-q8_0.gguf \
    --text      "A different sentence in the cloned voice." \
    --reference samples/alice.wav \
    --max-new-tokens 600 \
    --output    cloned.wav
```

Any WAV works — it is linearly resampled to the model's rate internally, and its channel count
is adapted to match (a mono file is duplicated for the stereo codec, a stereo one downmixed for
the mono codec). A few seconds of clean speech is usually enough.

### Streaming

`--stream` decodes the codec alongside generation instead of once at the end, so audio
exists long before the utterance is finished. The WAV written at exit is the same either
way; what changes is when you can hear it.

```bash
# play it as it is generated (48 kHz stereo — use -r 24000 -c 1 for the delay family)
./build/moss-tts-cli \
    --model weights/moss-tts-local-1.5-q8_0.gguf \
    --text  "Streaming means the first words arrive long before the last ones do." \
    --stream 8 --stream-out - --output out.wav \
  | aplay -f S16_LE -r 48000 -c 2
```

Smaller chunks mean lower latency and more overhead — the decode runs inline with
generation, so it stalls the LM. See [docs/STREAMING.md](docs/STREAMING.md) for the
measured curve and for why the codec needs a per-stage KV cache rather than simple
chunking.

## Server usage

```bash
./build/moss-tts-server --model weights/moss-tts-q8_0.gguf --host 0.0.0.0 --port 8080
```

`moss-tts-server` shares the same `--main-gpu`, `--n-gpu-layers`, `--n-batch`, `--n-ctx`,
`--no-flash-attn`, `--skip-codec`, and `--aux-cpu` flags as the CLI. Generations are serialized through a single mutex (libllama
state is not reentrant); concurrent requests queue cleanly.

Additional flags:

| flag                | meaning                                                              |
|---------------------|----------------------------------------------------------------------|
| `--webui-dir DIR`   | serve a static WebUI from `DIR` at `/` (overrides auto-detection)    |
| `--no-webui`        | disable WebUI even if a `webui/` directory is found                  |

By default the server auto-detects `./webui` or `<binary_dir>/webui` and mounts it at `/`.
The CMake build stages the source tree's `webui/` next to the server binary, so a fresh build
+ `./build/moss-tts-server --model …` is enough — opening http://127.0.0.1:8080/ shows the UI.

### Endpoints

- `GET /health` — readiness probe, returns `ok`.
- `GET /info` — JSON with model dims, codec status, and the request counter. For
  MOSS-SoundEffect it also reports `max_seconds` and the solver defaults.
- `GET /v1/models` — OpenAI-compatible model list. Always exactly one entry, named after
  the loaded architecture; the model cannot be switched at runtime.
- `GET /v1/audio/voices` — voices registry, vLLM-Omni shaped. Reports the single `default`
  entry, or an empty list for MOSS-SoundEffect, which has no notion of a voice. Cloning is
  driven by reference audio rather than a named voice.
- `POST /tts` — native TTS for the autoregressive families. JSON in, `audio/wav` or
  `audio/pcm` out per `response_format` (or `text/plain` with the error message on failure).
- `POST /sfx` — native MOSS-SoundEffect. Same response formats.
- `POST /v1/audio/speech` — OpenAI-compatible; dispatches on the loaded architecture.

One model is loaded at startup, so exactly one of `/tts` and `/sfx` applies — the other
answers `400` with a pointer to the right one.

#### `POST /tts` request body

| field                 | type     | required | notes                                                    |
|-----------------------|----------|----------|----------------------------------------------------------|
| `text`                | string   | yes      | utterance to synthesize. May start with a `${token:N}` prefix as an inline duration hint |
| `instruction`         | string   | no       | voice / style description (voice-generation mode)        |
| `language`            | string   | no       | language hint                                            |
| `quality`             | string   | no       | upstream quality hint, passed through into the prompt    |
| `token_count`         | int      | no       | duration hint, 1 s ≈ 12.5 tokens. Wins over the `${token:N}` prefix |
| `tokens`              | int      | no       | legacy alias for `token_count`                           |
| `max_new_tokens`      | int      | no       | default 4096                                             |
| `reference_wav_b64`   | string   | no       | base64-encoded WAV bytes for voice cloning               |
| `ref_text`            | string   | no       | transcript of the reference — selects continuation mode (`moss_tts_local` only); needs `reference_wav_b64` alongside it |
| `references`          | array    | no       | array form of the two above: `[{"ref_audio": …, "ref_text": …}]`. At most one entry |
| `response_format`     | string   | no       | `"wav"` (default) or `"pcm"` — raw 16-bit little-endian, no container |
| `stream`              | bool     | no       | default false; chunked transfer, requires `"pcm"` (see below) |
| `stream_chunk_frames` | int      | no       | default 8; one frame is 80 ms of audio                   |
| `sampling`            | object   | no       | `text_temperature`, `text_top_p`, `text_top_k`, `audio_temperature`, `audio_top_p`, `audio_top_k`, `audio_repetition_penalty`, `seed` |

Response headers carry timing info: `X-MOSS-Audio-Frames`, `X-MOSS-Generate-Seconds`,
`X-MOSS-Decode-Seconds`, and `X-MOSS-Channels`. A streamed response carries
`X-MOSS-Sample-Rate`, `X-MOSS-Channels`, and `X-MOSS-Stream-Chunk-Frames` instead — the
timings are not known when the headers go out.

#### Streaming

`"stream": true` answers with chunked transfer, writing each codec chunk as it is
decoded instead of buffering the whole utterance. On MOSS-TTS-Local this cuts time to
first audio from 2.54 s to 0.57 s for a 4.8 s utterance, at +31% total time — the codec
now runs inline with generation. Larger `stream_chunk_frames` recovers the throughput:
16 frames costs +7% and still answers in 0.83 s. Audio arrives faster than real time at
every setting, so playback never starves. Full measurements in
[docs/STREAMING.md](docs/STREAMING.md).

It requires `response_format: "pcm"`. A RIFF header states its own data length in its
first 44 bytes and that is unknown until generation ends, so the endpoint rejects the
`wav` combination rather than emit a header that lies; `X-MOSS-Sample-Rate` and
`X-MOSS-Channels` carry what a player needs. MOSS-SoundEffect rejects the flag outright —
flow matching refines every latent frame at once, so no prefix of the audio exists until
the last solver step.

The payload limit is 64 MB, which fits a ~60 s reference WAV after base64 encoding.

#### `POST /sfx` request body

MOSS-SoundEffect only — a diffusion transformer rather than an autoregressive model, so
none of the TTS sampling knobs apply.

| field                    | type   | required | notes                                                  |
|--------------------------|--------|----------|--------------------------------------------------------|
| `prompt`                 | string | yes      | sound description (`input` and `text` are also accepted) |
| `seconds`                | float  | no       | default 10.0, capped by the model. Must be > 0         |
| `steps`                  | int    | no       | default 100; each costs two DiT passes. Must be ≥ 1    |
| `cfg_scale`              | float  | no       | default 4.0; 1.0 skips the unconditional pass and halves the work |
| `sigma_shift`            | float  | no       | default 5.0                                            |
| `negative_prompt`        | string | no       | default empty, which is what it was trained on         |
| `seed`                   | uint64 | no       | default 0 — a valid seed, not a sentinel               |
| `append_duration_suffix` | bool   | no       | default true; appends ` duration: <X>s` to the prompt, as trained |
| `response_format`        | string | no       | `"wav"` (default) or `"pcm"`                           |

`seconds` is purely textual: the DiT always denoises `max_seconds` worth of latent and the
waveform is cropped afterwards. `stream` is rejected outright — flow matching refines every
latent frame at once, so no prefix exists until the last solver step.

Response headers: `X-MOSS-Prompt` (the prompt as actually sent, duration suffix included),
`X-MOSS-Latent-Frames`, `X-MOSS-Solve-Seconds`, `X-MOSS-Decode-Seconds`.

```bash
curl -s -X POST http://localhost:8080/sfx \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"a distant thunderclap rolling over a field","seconds":8}' \
    -o sfx.wav
```

#### `POST /v1/audio/speech` — OpenAI-compatible endpoint

Accepts the OpenAI TTS JSON schema, so any client or SDK targeting the OpenAI API works
drop-in, plus a few native extensions for what that schema has no field for.

| field                | type   | required | notes                                                     |
|----------------------|--------|----------|-----------------------------------------------------------|
| `input`              | string | yes      | text to synthesize (maps to native `text`), or the sound description for MOSS-SoundEffect |
| `model`              | string | no       | ignored (the model is already loaded at server startup)     |
| `voice`              | string | no       | passed as an instruction hint to the model                  |
| `response_format`    | string | no       | `"wav"` (default) or `"pcm"`; anything else is a 400        |
| `stream`             | bool   | no       | default false; same semantics as `/tts`                     |
| `stream_chunk_frames`| int    | no       | default 8; one frame is 80 ms of audio                      |
| `speed`              | float  | no       | 0.25–4.0, scales the token budget (default 1.0). This sets `max_new_tokens`; the field itself is not read here |
| `token_count`        | int    | no       | duration hint, as on `/tts`; the `${token:N}` prefix works too |
| `reference_wav_b64` / `ref_text` / `references` | — | no | voice cloning and continuation, exactly as on `/tts`        |
| `sampling`           | object | no       | as on `/tts`. The same keys are also accepted flattened at the top level, which is what the cookbooks send — flat keys win over the nested form |

When the loaded model is MOSS-SoundEffect this route dispatches to the `/sfx` path instead:
`input` becomes the prompt, the [`/sfx`](#post-sfx-request-body) solver fields are accepted
as extensions, and `voice` and `speed` are ignored.

Validation errors (missing/non-string `input`, undecodable or non-WAV `reference_wav_b64`,
out-of-range `speed`, unsupported `response_format`, `stream` with `wav`) return `400`
with a `text/plain` message.

Response: `audio/wav` or `audio/pcm` per `response_format`, 16-bit either way, same as the
native endpoint. Rate and channel count come from the loaded model — 24 kHz mono for the
delay family, 48 kHz stereo for MOSS-TTS-Local, 48 kHz mono for MOSS-SoundEffect — never
hardcoded.

### Examples

```bash
# Plain TTS
curl -s -X POST http://localhost:8080/tts \
    -H 'Content-Type: application/json' \
    -d '{"text":"Hello from the persistent server.","max_new_tokens":300}' \
    -o out.wav

# Voice cloning
python3 -c "import base64,json,sys; \
    print(json.dumps({'text': sys.argv[1], 'max_new_tokens': 600, \
        'reference_wav_b64': base64.b64encode(open(sys.argv[2],'rb').read()).decode()}))" \
    "A different sentence in the cloned voice." samples/alice.wav > req.json
curl -s -X POST http://localhost:8080/tts \
    -H 'Content-Type: application/json' --data @req.json -o cloned.wav

# OpenAI-compatible endpoint (works with the OpenAI Python SDK)
curl -s -X POST http://localhost:8080/v1/audio/speech \
    -H 'Content-Type: application/json' \
    -d '{"model":"tts-1","input":"Hello from the OpenAI-compatible endpoint!","voice":"alloy"}' \
    -o openai_out.wav
```

#### Using with the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # the server does not check authentication
)

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello! This is a test of the OpenAI-compatible TTS endpoint.",
    response_format="wav"
)
response.stream_to_file("output.wav")
```

## WebUI

A small single-page WebUI ships with the server. It is plain HTML / CSS / vanilla JS — no build
step, no dependencies — and is mounted at `/` whenever the server can find a `webui/` directory.

Features:

- One-shot **TTS** form (text, optional voice instruction, language hint, max-new-tokens, advanced
  sampling, optional reference WAV for voice cloning).
- Per-browser **history** stored in IndexedDB — every successful generation is appended together
  with its metadata and original WAV blob.
- In-page **playback**, per-item **download**, **reuse text** (re-populates the form), per-item
  **delete**, and a **Clear all** button.
- Live `/info` ping in the header showing model dims, codec status, and requests served.

### Launcher

The provided wrapper finds the server binary and webui directory, then starts the server bound
to localhost:

```bash
# Linux / macOS
scripts/launch-webui.sh weights/moss-tts-q8_0.gguf

# Windows
scripts\launch-webui.bat weights\moss-tts-q8_0.gguf
```

Then open <http://127.0.0.1:8080/>.

Override the bound host/port or paths with `MOSS_HOST`, `MOSS_PORT`, `MOSS_SERVER`, `MOSS_WEBUI`,
or pass any extra `moss-tts-server` flag after the model path (e.g. `--main-gpu 0`,
`--skip-codec`).

### Running it manually

```bash
./build/moss-tts-server --model weights/moss-tts-q8_0.gguf --host 0.0.0.0 --port 8080
# WebUI: http://localhost:8080/
# API:   http://localhost:8080/tts, /v1/audio/speech, /info, /health
```

The WebUI directory is plain files under `webui/` — edit `index.html`, `style.css`, or `app.js`
and refresh the page; no rebuild needed.

## GPU placement

Both binaries pin the full model — libllama backbone **and** the aux GGML backend (embeddings,
LM heads, codec) — to a single device via `LLAMA_SPLIT_MODE_NONE`. With no flag, they auto-pick
the GPU that has the most free VRAM at load time:

```
Model::load: pinning to GPU 1 (CUDA1, 15703/15850 MiB free)
```

Override with `--main-gpu N` (index into the ggml GPU device list — same convention as
libllama's own `main_gpu`). Splitting the backbone across GPUs is *not* currently supported,
because the aux backend would still need to land on one device and would then race the backbone
fragment on that device for VRAM. At Q8\_0 (~9 GB backbone + ~5 GB aux ≈ 14 GB) anything ≥ 16 GB
fits.

## Architecture

The **delay family** (`moss_tts_delay`), which is the most involved of the three on the LM side.
MOSS-TTS-Local replaces the delay-pattern state machine with a depth transformer that emits all
12 codebooks per frame, and MOSS-SoundEffect is not autoregressive at all — see
[`docs/STATUS.md`](docs/STATUS.md) for those.

```
                  ┌────────────────────────────────────────────┐
   text  ────►    │ BPE tokenizer (Qwen3, 155 648 vocab)       │
   ref.wav ──►    │ Codec encoder (audio → 32×T_a codes)       │ — voice cloning
                  └────────────────────────────────────────────┘
                            │
                  ┌─────────▼──────────────────────────────────┐
                  │ Embedding stack:                           │
                  │   text emb  +  Σᵢ audio_emb_i(code_i)      │  (33 tables, 4096 dim)
                  └─────────┬──────────────────────────────────┘
                            │ inputs_embeds (S, 4096) — fed via batch.embd
                  ┌─────────▼──────────────────────────────────┐
                  │ Qwen3-8B backbone (libllama)               │
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
                            │ codes (32 × T_a)
                  ┌─────────▼──────────────────────────────────┐
                  │ Codec decoder (4-stage pure transformer)   │  (GGML)
                  │   12.5 Hz → 24 000 Hz (×1920 upsample)     │
                  └─────────┬──────────────────────────────────┘
                            ▼
                       waveform (f32, 24 kHz)  →  16-bit WAV
```

The codec transformers use *sliding-window* causal attention: each stage attends to at most
10 s of context (`causal_transformer_context_duration` upstream — 125 keys at 12.5 Hz up to
1000 keys at 100 Hz). The window is part of the trained model, not an optimization; decoding
with unbounded attention degrades audio progressively past the 10 s mark (issue #7).

For the full per-stage tensor shapes, list of bugs hit during bring-up, and benchmark numbers,
see [`docs/STATUS.md`](docs/STATUS.md).

## Repo layout

| path                                  | role                                                       |
|---------------------------------------|------------------------------------------------------------|
| `scripts/convert_hf_to_gguf.py`       | HF → backbone GGUF + sidecar GGUF                          |
| `src/model.cpp`, `src/aux_internal.h` | Two-file loader, aux backend, embed / LM-head graphs       |
| `src/codec.cpp`                       | RLFQ + transformer encoder/decoder graphs (4 stages per side in v1, 6 in v2), batch and incremental |
| `src/tokenizer.cpp`                   | libllama BPE wrapper                                       |
| `src/frame_decoder.cpp`               | Family-agnostic seam between the backbone loop and the code sampler |
| `src/delay.cpp`                       | DelayState + sampling (top-k/p, repetition penalty)        |
| `src/local_transformer.cpp`           | MOSS-TTS-Local's depth transformer + its per-frame KV cache |
| `src/dit.cpp`, `src/dac_vae.cpp`, `src/soundeffect.cpp` | MOSS-SoundEffect: the Wan-style DiT, the DAC VAE decoder, and the flow-matching sampler |
| `src/pipeline.cpp`                    | Prompt builder + autoregressive loop + codec dispatch      |
| `src/wav.cpp`                         | RIFF/WAVE I/O (no libsndfile dep)                          |
| `src/cli/moss_tts_cli.cpp`            | CLI entry point                                            |
| `src/server/moss_tts_server.cpp`      | HTTP server + static WebUI mount                           |
| `webui/`                              | Browser WebUI (vanilla HTML / CSS / JS, IndexedDB history) |
| `scripts/launch-webui.{sh,bat}`       | Server + WebUI launcher                                    |
| `tests/`                              | Seam probes — see [`docs/STATUS.md`](docs/STATUS.md) for what each proves |
| `docs/STATUS.md`, `docs/STREAMING.md` | Feature matrix + how correctness is established; streaming design and measurements |
| `third_party/cpp-httplib/httplib.h`   | Vendored single-header HTTP library                        |

## License

Apache-2.0, matching upstream MOSS-TTS.
