# openmoss — handoff

Branch: `feat/moss-tts-local` (off `main`). Everything below is committed.

> **The branch has never been pushed.** All 25 commits exist only on this disk,
> and `origin` still has just `main` at 64eabe0 (July 11). That is the single
> most important operational fact here.

## TL;DR

All three MOSS families work end to end, from one binary, verified against the
PyTorch reference at every seam. MOSS-SoundEffect is published at
<https://huggingface.co/ilintar/moss-soundeffect-gguf>.

| Family | Arch id | Output | State |
|---|---|---|---|
| MOSS-TTS / v1.5 / VoiceGenerator | `moss_tts_delay` | 24 kHz mono | done |
| MOSS-TTS-Local-Transformer-v1.5 | `moss_tts_local` | 48 kHz stereo | done, incl. voice clone + continuation |
| MOSS-SoundEffect-v2.0 | `moss_soundeffect` | 48 kHz mono | done, published |

**Streaming now works too** — per-stage KV cache in the codec, `--stream` in the
CLI, chunked transfer in the server. See
[docs/STREAMING.md](../docs/STREAMING.md). Nothing substantial is outstanding;
what remains is the short list at the end of this document.

## Where everything lives

| What | Where |
|---|---|
| HF sources | `/devel/models/models/openmoss-src/` |
| GGUFs | `/devel/models/models/openmoss/` |
| **Reference dumps + harnesses** | **`/devel/models/models/openmoss-ref/`** |
| `llama-quantize` | `/devel/tools/llama.cpp/build/bin/llama-quantize` (not built in-tree) |

The reference directory was moved out of a session scratchpad deliberately — it
is 551 MB that cost real compute to produce, and every probe below depends on
it. It holds `ref/` (DiT), `ref_vae/`, `ref_pipe/` (full pipeline), the
f16-budget variants, `se2src/` (the upstream Python, which is *not* in the model
dirs and had to be fetched from `github.com/OpenMOSS/MOSS-TTS`), and the
generator scripts next to each set so any of it can be rebuilt.

Build: `cmake -B build -DGGML_VULKAN=ON && cmake --build build -j`

## Running it

```bash
# sound effects
./build/moss-tts-cli --model .../moss-soundeffect-2.0.gguf \
  --text "a dog barking twice" --seconds 5 --steps 100 --vk-f32 --output out.wav

# speech
./build/moss-tts-cli --model .../moss-tts-local-1.5-q8_0.gguf \
  --text "Hello there." --output out.wav

# server (one model per server; /tts or /sfx applies, the other 400s)
./build/moss-tts-server --model <gguf> --port 8080

# streaming: hear it arrive rather than waiting for the file
./build/moss-tts-cli --model .../moss-tts-local-1.5-q8_0.gguf \
  --text "..." --stream 8 --stream-out - --output out.wav | aplay -f S16_LE -r 48000 -c 2

# same over HTTP; requires pcm, since a RIFF header needs a length up front
curl -N -X POST localhost:8080/tts -H 'Content-Type: application/json' \
  -d '{"text":"...","response_format":"pcm","stream":true}'
```

**Use `--vk-f32` on Vulkan** for MOSS-SoundEffect. It sets
`GGML_VK_DISABLE_F16`, stopping the backend accumulating *libllama's* matmuls in
f16 — which costs the text conditioning 3.3e-2 against the fp32 reference, and
the DiT cross-attends to it on all ~200 forward passes. With it, the final
latent lands at 1.07e-3, correlation **1.000000**. It costs nothing measurable.
It is opt-in only because the env var is process-global; auto-enabling it for
this architecture means reading `moss.architecture` before libllama initialises,
which is a deliberate loader reorder. The server has no flag — set the env var.

Timings on a Radeon 8060S: SoundEffect ~90 s for 100 steps (independent of
duration — the DiT always denoises 30 s); TTS-Local ~22 frames/s (1.74x real
time) with codec decode at ~30x real time.

## The verification method — read this before touching numerics

Generated audio is a useless test: sampling and 100 solver steps both amplify
tiny differences. Everything is verified at *seams*, feeding both sides identical
input bytes. Two things make the numbers interpretable, and skipping either
wastes a day.

**1. Establish the f16 budget first.** Rerun the reference with its weights
round-tripped through f16 (`p.copy_(p.half().float())`) — `ref_f16w/` is that
run. For the DiT the floor is ~2e-4 at every seam and it does **not** accumulate
with depth. So ~2e-4 is the floor and anything much above it is a bug, not "f16
being lossy". Without this, a 25x-too-large error looks like a plausible
hand-wave.

**2. The DiT's head amplifies.** Residual channel 421 grows to ~50x its
neighbours by the last block, so the head's non-affine LayerNorm collapses every
frame onto nearly one direction — only ~7e-5 of the normalised energy survives
demeaning. The output is a per-channel DC offset of ±150 carrying ~2.5 std of
real signal, so **a naive relative error reads a comfortable 1e-3 while the
per-frame structure is destroyed.** Compare it demeaned. Frame-constant error
passes at ~0.8x; frame-varying error is amplified ~50x.

### The probes

```bash
R=/devel/models/models/openmoss-ref
./build/moss-sfx-probe       <sfx.gguf> $R/ref      [_t16]   # DiT, 8 seams
./build/moss-sfx-vae-probe   <sfx.gguf> $R/ref_vae  [_t40]   # DAC decoder, 10 seams
./build/moss-prompt-probe    <tts.gguf> "<text>" $R/pp_codes.txt ["<ref_text>"]
./build/moss-codec-causality <tts.gguf> $R/codes_causality.txt
./build/moss-codec-stream    <tts.gguf> $R/codes_long.txt [chunk ...]
./build/moss-local-probe     <tts.gguf> <hidden.f32>
./build/moss-codec-roundtrip <tts.gguf> <in.wav> <out.wav>
```

All verified working from the durable path. Env knobs: `MOSS_AUX_CPU=1` reruns
on the CPU backend, which separates a graph error from a backend numerics
difference; `MOSS_SEAM_TOL=F` raises the per-seam budget for a quantised sidecar
(Q8_0 sits around 5e-3).

`moss-codec-stream` has a structure worth copying for any cached-attention work:
it runs the *whole* utterance as one push first, which must come back
**bit-identical** to the batch decode. That pins the streaming graph to the batch
graph, so a chunked run that then drifts can only be the cache. Without that
control, a 1e-3 chunked error is uninterpretable — it could be either. And the
signal to watch in the chunked runs is not the magnitude but the *slope*: a
too-short cache is clean at the start and degrades once the window fills, so
flat error across 853 frames is the actual proof.

`moss-prompt-probe` is the cheapest check in the repo — prompt ids have no
numerical tolerance, and it needs neither the codec nor the backbone. All three
MOSS-TTS-Local layouts match the reference processor on **every integer**: plain
69x13, voice clone 77x13, continuation 82x13. Regenerate the reference side with
`$R/ref_prompt.py` — note it wants codes as `[T, n_vq]` while `--dump-codes`
writes `(n_vq, T)`, so transpose.

The VAE probe's chunking check must stay bit-identical: convolution is exactly
shift-equivariant when each output's summation order is unchanged. 12 latent
frames of overlap already suffice and 2 do not (1.5e-2), so the chosen 32 is
real margin.

### Where things stand

| seam | f16 sidecar | Q8_0 sidecar |
|---|---|---|
| DiT `ctx_emb` | 3.1e-4 | 5.3e-3 |
| DiT `blk29_out` | 4.2e-4 | 2.7e-3 |
| DiT `dit_out` demeaned | 2.805e-2 | 2.805e-2 |
| VAE `audio` | 1.2e-3 | — |
| text conditioning | 1.7e-3 (`--vk-f32`) / 3.3e-2 without | — |
| final latent, 8 steps | **1.07e-3, corr 1.000000** | 2.08e-2, corr 0.999783 |

For scale: the same latent shifted 0.74 s correlates at 0.90, unrelated noise at
0.06. Quantisation makes every *seam* 6-10x worse yet leaves the demeaned output
identical — weight-rounding error is almost entirely frame-constant, and the
head discards exactly that component.

Streaming, against a batch decode of the same codes:

| check | result |
|---|---|
| one single push, both codec generations | **bit-identical** |
| chunked, 853 frames, chunk 1 / 8 / 32 / 128 | 1.44 / 1.60 / 1.35 / 1.23e-3, and *flat* along the utterance |
| end to end `moss_tts_local`, fixed seed | identical codes; 3.5e-3, corr 0.99999398 |
| end to end `moss_tts_delay`, fixed seed | identical codes; 6.7e-4, corr 0.99999978 |

**What `latent_final` does not prove.** The guidance signal
`rms(vpos-vneg)/rms(vpos)` starts at 23% and decays under 1% within a few steps,
so a badly wrong text encoder still lands close. Diff `--dump-context` against
`$R/ref_pipe/context_pos.bin` to test that path directly — that is how the
libllama f16 problem was found, long after the latent comparison had "passed".

## Traps already hit — do not rediscover these

1. **ggml's Vulkan backend accumulates matmuls in f16 by default.** Any node at
   `GGML_PREC_DEFAULT` costs 5e-3 to 1e-2 relative — far more than f16 weights.
   Call `ggml_mul_mat_set_prec(node, GGML_PREC_F32)`.
   **This bit three separate times**: the DiT, the codec (12 sites across 68
   layers, 16x improvement when fixed), and libllama's own graph, where there is
   no node to set and `GGML_VK_DISABLE_F16=1` is the answer. It hides because
   matrix-*vector* products take a different, f32-accumulating path — so
   anything computed one token at a time looks perfect while every batched seam
   is uniformly wrong. That flatness is the tell. Always measure the speed cost
   rather than assuming: it was zero.
2. **`ggml_set_output` on a *view* protects nothing** — the parent owns the
   buffer.
3. **`ggml_gallocr_needs_realloc` ignores output flags**, so a tapped graph
   silently reuses an untapped one's plan. Tapped runs get their own allocator.
   Invisible on small inputs, where nothing gets recycled.
4. **`ggml_conv_transpose_1d`** asserts `p0 == 0`, `d0 == 1`, a 2-D input, and
   Vulkan is f32-only for it. Run unpadded and crop: ggml gives `(T-1)s + k`,
   PyTorch `(T-1)s - 2p + k + op`.
5. **`ggml_conv_1d` hardcodes an f16 im2col** — a precision floor and over a
   gigabyte at full length. `conv1d_()` does it in f32 by hand.
6. **Snake reciprocals must stay f32.** Three decoder alphas overflow f16, and
   **one is negative** so a clamp-to-positive guard is wrong. The failure is
   NaN, not inaccuracy: the reciprocal goes to inf while `sin(a*x)^2` underflows
   to zero on those same channels.
7. **Never infer a projection's absence from matching dimensions.**
8. **A stream callback must not capture block-scoped locals by reference.**
   Three APIs here take one (`StreamCallback`, `SoundEffectProgress`, httplib's
   `DataSink`) and all are invoked *after* the block that built them. The failure
   is not a plausible garbage value: RVO makes the callee's return object alias
   the caller's dead stack slots, so writing to a dangling counter corrupts the
   `std::vector` being appended to and it surfaces as `std::bad_alloc` inside
   libstdc++, with a backtrace pointing at the wrong function entirely.
9. **Do not quantise the SoundEffect backbone.** Tested: Q8_0 moves the
   conditioning to 2.7e-2 and the latent to 1.8e-1, corr 0.984. A text encoder
   consumed as a continuous conditioning vector has no sampling stage to absorb
   the error. f16 vs the source bf16 is *not* a compromise — bf16 has 7 mantissa
   bits against f16's 10, and the measured round-trip across 2.03 B weights is
   **4.7e-9**.

## Things the reference does that look like bugs but are not

* The unconditional context is exactly zeros (empty negative prompt → 0 tokens →
  hard-zeroed), so Qwen3 is skipped for that branch.
* Duration is textual: `" duration: {seconds:.1f}s"`, and the DiT always
  denoises `max_seconds` regardless.
* Padding never needs to reach the backbone — the encoder is causal and the tail
  is zeroed anyway, so running just the real tokens is exact.
* The `chunk(3)` on the RoPE table is a no-op; it is a plain 1-D RoPE.
* `norm_q`/`norm_k` are RMSNorm over the full 1536 axis, **not** per-head.
* Only `norm3` is affine in a block, and diffusers stores it as `norm2`.
* The head takes the plain `t`, not `t_mod`.
* FFN GELU is `approximate='tanh'` → `ggml_gelu`, not `ggml_gelu_erf`.
* Continuation mode replaces the reference block with the literal text `"None"`
  and puts the codes at the end in the assistant channel, with no `audio_end`.

## Streaming, as built

[docs/STREAMING.md](../docs/STREAMING.md) is the full account. What matters here:

Each codec stage keeps the `context - 1` positions its next queries can reach,
on the host, handed to each push's graph as an input — `src/local_transformer.cpp`
is the same trade. The batch path is untouched: the attention plan just gained
`q_origin`/`k_origin`, both 0 there. Both codec generations verified.

Two numbers to hold onto. **Streaming buys latency and spends throughput**:
first audio 2.54 s → 0.57 s at the default 8-frame chunk, for +31% total time,
because the decode now runs inline with generation. The knee is at 16 frames
(+7%, 0.83 s) — below it the fixed per-push cost starts to dominate. And **the
delay family is the awkward one**: its 32-step ramp means no frame completes for
32 steps, and it generates at ~0.9x real time, so a client there should buffer
rather than play straight through.

MOSS-SoundEffect still cannot stream — flow matching refines all 1500 latent
frames simultaneously — and all three endpoints now say so rather than ignoring
the flag. Progress over SSE remains the only thing worth exposing there;
`generate_sound_effect()` already takes a callback.

## What is left

* Continuation mode for `moss_tts_delay` — different slot layout, no verified
  reference locally, so `ref_text` is rejected for it rather than ignored.
* Server concurrency — one mutex serialises generation. Streaming makes this
  more visible: a streamed request holds the lock for its whole duration.
* The WebUI does not use streaming; it still posts and waits.
* README drift predating this work — the endpoint list omits `/sfx`, and several
  option descriptions still assume 24 kHz mono.
* The `main` branch is 14 commits behind this one and this branch is unpushed.

## Listening to output

A Lemonade server on `:8000` with `gemma-4-E4B-it-GGUF-Q6_K` accepts audio; load
it with `POST /api/v1/load` and send `input_audio` parts to
`/api/v1/chat/completions`, at **16 kHz mono** (`$R/ask_audio.py` has the
converter — `audioop` is gone in 3.13+).

Two caveats, both hit repeatedly. It is a *reasoning* model, so a small
`max_tokens` is consumed entirely by `reasoning_content` and `content` comes
back empty — budget 1200+ and read `reasoning_content` when `content` is blank.
And it is genuinely unreliable: it intermittently claims "no audio was provided"
for clips it just processed, and called a glass smash a chainsaw. **Always run a
known-good control clip first.**

For sound effects the **envelope is a better judge than the model**. Bin RMS into
0.5 s slices: "a dog barking twice" gives two discrete bursts separated by
silence, "heavy rain on a tin roof" is flat and sustained, "a single glass
shattering on stone" is one transient with a decay tail. That matched the prompt
every time and needs no external model.
