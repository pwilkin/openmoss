# Handoff: MOSS-SoundEffect-v2.0 support

Branch: `feat/moss-tts-local` (off `main`). Everything below is committed.

## TL;DR

**Done and shipped.** All three GGML pieces are written and verified against the
PyTorch reference, MOSS-SoundEffect generates end to end from both the CLI and the
server, and the weights are published at
<https://huggingface.co/ilintar/moss-soundeffect-gguf>.

Phase 6 (the server option surface) is done, including `ref_text` /
continuation mode.

## Commits on this branch

```
5d77b51  feat(webui): MOSS-SoundEffect mode
5e515d2  docs: streaming design pass, and refresh STATUS.md
726c811  perf(sfx): cache the graphs on Model, and decode only the latent needed
fc6cf47  fix(codec): accumulate the codec's matmuls in f32
7810fce  feat(tts): continuation mode (ref_text) for MOSS-TTS-Local
c68148b  feat(server): Phase 6 — SoundEffect endpoint and option-surface parity
a1fae1c  docs: refresh the MOSS-SoundEffect handoff
ce1fdcb  feat(sfx): dump the text conditioning, and --vk-f32
7cd33b1  feat(convert): optional Q8_0 for the sidecar's projection matrices
ba8744b  feat(sfx): flow-matching sampler and end-to-end generation
a056bbb  feat(sfx): DAC VAE decoder
ddcdb4c  feat(sfx): WanAudioModel DiT graph for MOSS-SoundEffect-v2.0
92ec1e9  docs: handoff notes for the remaining MOSS-SoundEffect work
e8f0467  feat(convert): support MOSS-SoundEffect-v2.0
db7bfe4  feat(cli,tests): sampling flags, code dump, and a depth-transformer probe
747b3a6  feat(convert): support the moss_tts_local family
ff26e54  feat(codec): MOSS-Audio-Tokenizer-v2 + chunked sliding-window attention
21ff13f  feat(core): model-family abstraction + MOSS-TTS-Local generation
c7409a9  feat(wav): channel-aware WAV I/O
```

Build with `cmake -B build -DGGML_VULKAN=ON && cmake --build build -j`.

## Running it

```bash
./build/moss-tts-cli \
  --model /devel/models/models/openmoss/moss-soundeffect-2.0.gguf \
  --text "a dog barking twice" --seconds 5 --steps 100 --seed 42 \
  --vk-f32 --output out.wav
```

**Use `--vk-f32` on Vulkan.** It sets `GGML_VK_DISABLE_F16`, which stops the
backend accumulating *libllama's* matmuls in f16. Without it the text encoder
lands 3.3e-2 from the fp32 reference and the DiT cross-attends to that on all
~200 forward passes; with it the final latent goes from 1.7e-2 to 1.1e-3,
correlation 0.999855 to 1.000000. It costs nothing measurable (6.76 s of solve
against 6.77 s) because the weights stay f16 in memory either way — only shader
compute precision changes. The graphs in this repo already force f32
accumulation per node, so this only affects the backbone; the DiT probe is
bit-identical with and without it.

It is opt-in because the env var is process-global and would silently change the
two autoregressive families' numerics too. Auto-enabling it for this
architecture alone means reading `moss.architecture` from the sidecar *before*
libllama initialises — a deliberate loader reorder, not a one-liner.

`moss-tts-cli` dispatches on `moss.architecture`; the sampling flags do not
apply to this family. SoundEffect-only flags: `--seconds`, `--steps`,
`--cfg-scale`, `--sigma-shift`, `--negative-prompt`, `--no-duration-suffix`,
`--dump-latent`, `--init-latent`.

5 s at the default 100 steps takes ~88 s on a Radeon 8060S: 0.15 s text,
80 s solve (200 DiT passes), 8 s VAE. Cost is independent of `--seconds`,
because the DiT always denoises the full 30 s and the waveform is cropped.
`--cfg-scale 1` halves the solve by skipping the unconditional branch.

### Over HTTP

```bash
./build/moss-tts-server --model .../moss-soundeffect-2.0.gguf --port 8080
curl -X POST localhost:8080/sfx -H 'Content-Type: application/json' \
     -d '{"prompt":"a dog barking twice","seconds":5,"steps":100}' -o out.wav
```

One model is loaded per server, so exactly one of `/tts` and `/sfx` applies and
the other answers 400 pointing at the right one. `/v1/audio/speech` dispatches on
the loaded architecture and answers for either. Also: `GET /v1/models`,
`GET /v1/audio/voices`, `response_format` of `"wav"` or `"pcm"`, `token_count` or
an inline `${token:N}` prefix, flattened sampling keys, and `references[]` with
`ref_audio` plus optional `ref_text` (which selects continuation mode). Set
`GGML_VK_DISABLE_F16=1` in the environment — the server has no `--vk-f32` flag
of its own.

## Files on disk

| What | Where |
|---|---|
| HF sources (all three models) | `/devel/models/models/openmoss-src/` |
| GGUFs | `/devel/models/models/openmoss/` |
| SoundEffect backbone | `moss-soundeffect-2.0.gguf` (4.07 GB, Qwen3-1.7B text encoder) |
| SoundEffect sidecar, f16 | `moss-soundeffect-2.0.extras.gguf` (2.98 GB) |
| SoundEffect sidecar, Q8_0 | `moss-soundeffect-2.0-q8.extras.gguf` (1.67 GB; backbone is a hardlink) |
| Reference dumps | scratchpad `ref/`, `ref_vae/`, `ref_pipe/`, `ref_f16w/` — **regenerate if gone**, the scripts are next to them |
| `llama-quantize` | `/devel/tools/llama.cpp/build/bin/llama-quantize` (not built in-tree) |

## The verification method — read this before touching the numerics

Comparing generated audio is useless here: 100 solver steps amplify any
numerical difference. Everything was verified at *seams*, feeding both sides
identical input bytes. Two things had to be established first, and they are what
make the numbers interpretable.

**1. The f16 budget.** Storing weights as f16 costs ~2e-4 relative at every
seam. Measured by rerunning the fp32 reference with its weights round-tripped
through f16 (`p.copy_(p.half().float())`). It does *not* accumulate over the 30
blocks — it is flat end to end. So ~2e-4 is the floor, and anything much above
it is a bug rather than noise. Do not skip this step; without it, a 25x-too-large
error looks like a plausible "f16 is lossy" hand-wave.

**2. The head amplifies.** Residual channel 421 grows to ~50x its neighbours by
the last block, so the head's non-affine LayerNorm collapses every frame onto
nearly one direction — only ~7e-5 of the normalised energy survives demeaning.
The DiT output is therefore a per-channel DC offset of ±150 carrying only ~2.5
std of real signal. **A naive relative error against `dit_out` reads a
comfortable 1e-3 while the per-frame structure is destroyed.** Compare it
demeaned. Frame-constant error passes the head at ~0.8x; frame-varying error is
amplified ~50x.

That framing immediately exposed the one real bug in this work — see the traps
below.

### The probes

```bash
# DiT: t, t_mod, ctx_emb, x_patched, blk0 self/cross/out, blk29, dit_out
./build/moss-sfx-probe   <model.gguf> <scratchpad>/ref [_t16]

# VAE: post_quant, dec_in, blk1 up/res0/out, blk2..5, audio, + a chunking check
./build/moss-sfx-vae-probe <model.gguf> <scratchpad>/ref_vae [_t40]
```

Both gate on the f16 budget and print every seam. Env knobs:
`MOSS_AUX_CPU=1` reruns on the CPU backend, which separates a graph error from a
backend numerics difference; `MOSS_SEAM_TOL=F` raises the per-seam budget for a
quantised sidecar.

The VAE probe's chunking check is worth keeping honest: it re-decodes chunked and
diffs against single-shot, and they agree **bit for bit**, because convolution is
exactly shift-equivariant when each output's summation order is unchanged. An
overlap of 12 latent frames already suffices and 2 does not (1.5e-2), so the
chosen 32 is real margin — verified by temporarily shrinking it.

### Where things stand

| seam | f16 sidecar | Q8_0 sidecar | f16 budget |
|---|---|---|---|
| DiT `ctx_emb` | 3.1e-4 | 5.3e-3 | 2.0e-4 |
| DiT `blk29_out` | 4.2e-4 | 2.7e-3 | 1.6e-4 |
| DiT `dit_out` demeaned | 2.805e-2 | 2.805e-2 | ~1.1e-2 |
| VAE `audio` | 1.2e-3 | — | — |
| text conditioning vs fp32 ref | 1.7e-3 with `--vk-f32`, 3.3e-2 without | — | — |
| final latent vs fp32 ref, 8 steps | **1.07e-3 (corr 1.000000)** with `--vk-f32`; 1.70e-2 (corr 0.999855) without | 2.08e-2 (corr 0.999783) | — |

For scale on that last row: the same latent shifted by 0.74 s correlates at 0.90,
and unrelated noise at 0.06. Every configuration here is the same generation.

Note the Q8_0 column: quantisation makes every *seam* 6-10x worse yet leaves the
demeaned output identical, because weight-rounding error is almost entirely
frame-constant and the head discards exactly that component. End to end it costs
1.70e-2 → 2.08e-2, and the 100-step waveform envelope correlates at 0.999994
with the f16 one.

**A caveat about what `latent_final` proves.** Less than you would think. The
guidance signal `rms(vpos - vneg) / rms(vpos)` starts at 23% and decays to under
1% within a handful of steps, so a port with a badly wrong text encoder still
lands close on the final latent. Diff `--dump-context` against
`ref_pipe/context_pos.bin` to test that path directly — that is how the
libllama f16-accumulation problem was found, well after the latent comparison
had already "passed".

## Prompt construction is verified exactly, not approximately

`tests/moss_prompt_probe.cpp` dumps the grid and diffs it against the reference
processor. There is no numerical tolerance here — an id is right or it is not —
and the check needs neither the codec nor the backbone, which makes it the
cheapest thing in the repo. All three MOSS-TTS-Local layouts match every
integer: plain 69x13, voice clone 77x13, continuation 82x13.

```bash
./build/moss-prompt-probe <model.gguf> "<text>" [ref_codes.txt] [ref_text]
```

`ref_codes.txt` is the `(n_vq, T)` matrix `--dump-codes` writes. Note the
reference processor wants its codes the other way round, `[T, n_vq]` — transpose
before feeding it. The generator for the reference side is the scratchpad's
`ref_prompt.py`.

Continuation mode is what `ref_text` selects: the reference goes in the
*assistant* channel behind the trailing audio_start with no closing audio_end,
the reference-audio block is replaced by the literal text "None", and the
synthesis text becomes transcript + new material. Only moss_tts_local; the delay
family's layout differs and generate() rejects `ref_text` for it.

## Traps already hit (do not rediscover these)

1. **ggml's Vulkan backend accumulates matmuls in f16 by default.** Any node left
   at `GGML_PREC_DEFAULT` picks the f16acc pipeline, costing ~5e-3 relative over
   a 1536-long dot product — 25x the f16 weight budget. Call
   `ggml_mul_mat_set_prec(node, GGML_PREC_F32)`. It hides well: matrix-*vector*
   products take a different, f32-accumulating path, so `t` (computed one column
   at a time) was exact while every wide seam was off by the same flat amount.
   That flatness is the tell.

   This bit **twice** — the second time inside libllama's backbone graph, where
   there is no node to set. `GGML_VK_DISABLE_F16=1` (i.e. `--vk-f32`) is the
   answer there. Measure the speed cost rather than assuming one: it was zero
   here, because the weights stay f16 in memory either way and only shader
   compute precision moves.

2. **`ggml_set_output` on a *view* protects nothing** — the parent owns the
   buffer. The `t_mod` tap read whatever block 29 later wrote there. Flag the
   view's parent.

3. **`ggml_gallocr_needs_realloc` ignores output flags.** It compares node
   counts, shapes and sources, so a tapped graph silently reuses an untapped
   one's allocation plan and the taps read overwritten buffers. Invisible on
   small inputs, where there is enough slack that no reuse happens. Tapped runs
   now get their own allocator.

4. **`ggml_conv_transpose_1d` asserts `p0 == 0`, `d0 == 1`, a 2-D input, and
   Vulkan only has an f32 x f32 pipeline for it.** So the transposed convs run
   unpadded and crop afterwards: ggml gives `L = (T-1)s + k`, PyTorch gives
   `L = (T-1)s - 2p + k + op`, so `p` comes off the left and `p - op` off the
   right. With `k = 2s`, `p = ceil(s/2)`, `op = s%2` that is exactly `s*T`.

5. **`ggml_conv_1d` hardcodes an f16 im2col buffer** — a precision floor on the
   activations, and over a gigabyte at full length. `conv1d_()` in `dac_vae.cpp`
   does the im2col in f32 by hand, which needs f32 kernels; those are
   materialised once at construction with a cast-and-copy graph.

6. **Snake reciprocals must stay f32.** Three of the 15936 decoder alphas are
   small enough that `1/(a + 1e-9)` exceeds f16's 65504 ceiling, and **one of
   them is negative** (so a clamp-to-positive guard would be wrong). The f16
   failure is not merely inaccurate: the reciprocal goes to inf while
   `sin(a*x)^2` underflows to exactly 0 on those same channels, and `inf * 0` is
   NaN.

7. **A projection's absence must never be inferred from matching dimensions**
   (from the TTS-Local work; `src/codec.cpp` used to, and v2 would have silently
   dropped 10 learned square matrices).

## Things the reference does that look like bugs but are not

* **The unconditional context is exactly zeros.** The negative prompt is the
  empty string, which tokenizes to zero real tokens, and the encoder then zeroes
  every row past the real length. Qwen3 is skipped entirely for that branch.
* **Duration is textual.** The prompt gets `f" duration: {seconds:.1f}s"` and the
  DiT always denoises the full `max_seconds`; only the waveform is cropped.
* **Padding never needs to reach the backbone.** Upstream right-pads to 512 with
  an attention mask, but the encoder is causal, so a real token at position i
  attends only to 0..i — all real. Running just the real tokens is exact and
  skips up to 511 forwards. libllama's `t_embd` is set right after
  `result_norm`, the same tensor HF exposes as `hidden_states[-1]`.
* **The `chunk(3)` on the RoPE table is a no-op.** `precompute_freqs_cis_1d`
  splits 64 into 22/22/20 and `forward` concatenates them straight back. Do not
  mimic the uneven split; it is a plain 1-D RoPE, theta 10000, head_dim 128.
* **`norm_q`/`norm_k` are RMSNorm over the full 1536 axis, not per-head.** Apply
  before reshaping into heads.
* **Only `norm3` is affine** inside a block, and the diffusers export stores it
  under the name `norm2`. The converter already renames it.
* **The head takes the plain time embedding `t`, not `t_mod`**, with its own
  `[1, 2, dim]` modulation: index 0 = shift, 1 = scale.
* **FFN GELU is `approximate='tanh'`** — `ggml_gelu`, not `ggml_gelu_erf`.

## What is left

### Do not quantise the SoundEffect backbone

Now tested, and the handoff's suspicion was right. `llama-quantize` will process
it happily — it is a plain Qwen3-1.7B GGUF — but the model consumes the
encoder's *hidden state* as a continuous conditioning vector rather than turning
logits into tokens, so the error propagates instead of being absorbed by
sampling. Q8_0 moves the conditioning from 1.7e-3 to 2.7e-2 and the final latent
from 1.5e-2 to 1.8e-1, correlation 0.99988 → 0.984. That is a noticeably
different generation.

f16 rather than the source bf16 is *not* a compromise: bf16 has 7 mantissa bits
against f16's 10, so every bf16 weight is exactly representable as long as its
exponent fits. Across all 2.03 B text-encoder weights none overflow, 0.157% land
in f16's subnormal range and 1578 flush to zero, for a measured round-trip error
of **4.7e-9** — five orders of magnitude below what compute precision costs.

### Streaming — designed, not built

See [docs/STREAMING.md](../docs/STREAMING.md). The conclusion, with measured
numbers: the codec decoder *is* causal, but its six stages are banded at
10+10+8+4+2+1 s and those windows add, so a chunk needs 35 s of history. At
`W = 128` frames the error is larger than the signal (1.144); it only converges
past 512 frames (8.7e-3). Emitting 1 s chunks statelessly costs 36x the work,
which at the codec's ~30x real time cannot keep up. **Per-stage KV caching is the
architecture**; `src/local_transformer.cpp` is the pattern, the caches total
44.3 M floats, and the LM at 1.74x real time — not the codec at 30x — is the real
ceiling. MOSS-SoundEffect cannot stream at all: flow matching refines all 1500
latent frames simultaneously.

### Smaller things

* Continuation mode for `moss_tts_delay` (different slot layout, no verified
  reference locally; `ref_text` is rejected for it rather than ignored).
* Server concurrency — one mutex serialises generation.
* Progress over SSE for `/sfx`, since a 100-step run is ~90 s of silence and
  `generate_sound_effect()` already takes a progress callback.

## Listening to output

A Lemonade server on `:8000` with `gemma-4-E4B-it-GGUF-Q6_K` accepts audio.
Load with `POST /api/v1/load {"model_name": "gemma-4-E4B-it-GGUF-Q6_K"}` and
send `{"type": "input_audio", "input_audio": {"data": <b64 wav>, "format":
"wav"}}` to `/api/v1/chat/completions`. Convert to **16 kHz mono** first; there
is a hand-rolled converter in the scratchpad's `ask_audio.py` (`audioop` is gone
in Python 3.13+).

Two caveats, both hit again this session. It is a *reasoning* model, so a small
`max_tokens` is consumed entirely by `reasoning_content` and `content` comes back
empty — budget 1200+, and read `reasoning_content` when `content` is empty. And
it is genuinely unreliable: it intermittently claims "no audio was provided" for
clips it just processed, and misidentified a glass smash as a chainsaw. **Always
run a known-good control clip first.**

For sound effects the **envelope is a better judge than the model**. Bin the RMS
into 0.5 s slices and look at the shape: "a dog barking twice" gives two discrete
bursts separated by silence, "heavy rain on a tin roof" is flat and sustained,
"a single glass shattering on stone" is one sharp transient with a decay tail.
That matched the prompt in all three cases and needs no external model.
