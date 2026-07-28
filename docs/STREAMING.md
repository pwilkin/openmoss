# Streaming

Streaming works. The codec decoder keeps a per-stage K/V cache, so audio can be
emitted as codes arrive instead of after the whole utterance. This document
records why the cheap approach was rejected, what was built instead, and what it
costs — all measured on a Radeon 8060S (Strix Halo, Vulkan) with
MOSS-TTS-Local 1.5 Q8.

## What streams

`generate()` takes a `StreamCallback`. With one, the codec runs incrementally
alongside generation and the callback receives each chunk of PCM as it is
decoded; without one, nothing changes and the batch decode still happens at the
end. Both paths produce the same audio.

* CLI: `--stream [N]` (N frames per chunk, default 8) and `--stream-out PATH`,
  which writes raw s16le as it arrives — `-` for stdout, so you can pipe it to a
  player and hear the latency directly.
* Server: `"stream": true` on `/tts` or `/v1/audio/speech`, answered with
  chunked transfer. It requires `response_format: "pcm"`; see below.

MOSS-SoundEffect cannot stream and all three endpoints say so rather than
quietly ignoring the flag. Flow matching integrates the whole latent
simultaneously — every solver step refines all 1500 frames at once — so no
prefix of the audio exists until the last step finishes. What *is* worth
exposing there is progress, and `generate_sound_effect()` already takes a
callback for it.

## Why the codec cannot simply be chunked

The decoder is causal: its attention is banded, `(delta >= 0) & (delta < context)`.
Verified by decoding a prefix of a real 853-frame generation and comparing
against the same prefix of a full decode — `moss-codec-causality`:

| prefix | relative error |
|---|---|
| 213 of 853 frames | 1.05e-3 |
| 426 of 853 | 7.30e-4 |
| 639 of 853 | 2.41e-5 |

Those residuals are attention *blocking* differences (`ATTN_CHUNK` splits the
query axis differently at different lengths), not lookahead — a non-causal
decoder would show O(1) error. So audio can in principle be emitted as codes
arrive.

The problem is how much history a chunk needs. The six decoder stages are banded
at 10, 10, 8, 4, 2 and 1 seconds, and because they are sequential those windows
**add**: stage 5 needs 1 s of its input, which needs stage 4 over that span,
which needs 2 s before it, and so on — 35 s in total. Measured, by decoding the
last `W` frames alone and comparing the tail against the full decode:

| left context | relative error |
|---|---|
| 8 frames (0.6 s) | 1.245 |
| 16 (1.3 s) | 1.266 |
| 32 (2.6 s) | 1.194 |
| 64 (5.1 s) | 1.146 |
| 128 (10.2 s) | 1.144 |
| 256 (20.5 s) | 0.173 |
| 512 (41.0 s) | 8.70e-3 |

A cliff between 20 s and 41 s, converging only once `W` passes the theoretical
35 s (437 frames). Anything less is not approximate — at `W = 128` the error is
larger than the signal.

So a stateless chunked decoder pays `(437 + C) / C` times the work to emit `C`
frames: 36x at 1 s granularity, which at the codec's ~30x real time cannot keep
up. That is the option this design rejects.

## What was built

Per-stage K/V caches. Each stage holds the `context - 1` positions its next
queries can still reach — a query at p attends `(p - context, p]` — and every
push's graph is handed that prefix as an input, so nothing is recomputed. Cost
per frame is O(1) and granularity drops to a single frame (80 ms).

The attention plan already carved queries into blocks for the batch path, so it
only needed *global* coordinates: `q_origin` and `k_origin` record where each
tensor's index 0 sits, and both are 0 for a batch decode, which is therefore
bit-for-bit unchanged. The `PatchedPretransform` between stages multiplies the
frame rate, so one input frame becomes 2 at the next stage and so on up to 240
at the final unpatch; that is a fixed, causal expansion, so it only means each
stage's cache is a different length.

The cache lives on the host and is uploaded per push, which is the trade
`src/local_transformer.cpp` already makes for the depth transformer: it keeps
in-graph mutation out of the picture entirely. Sizes, at the trained window per
stage:

| stage | layers | d_model | window | floats |
|---|---|---|---|---|
| dec.0 | 32 | 1280 | 125 | 10.2 M |
| dec.2 | 12 | 768 | 250 | 4.6 M |
| dec.4/6/8/10 | 12 | 768 | 400 | 7.4 M each |
| **total** | | | | **44.3 M** |

177 MB at f32 — the same order as the codec weights themselves. If that upload
ever becomes the bottleneck, the fix is backend-resident ring buffers written
with `ggml_cpy`, the way libllama handles its own KV cache.

## Is it correct?

`moss-codec-stream` decodes the same codes both ways and compares. Two things
are checked, and the split matters:

**One single push is bit-identical to the batch decode** — rel 0.0, max|d| 0.0,
on both codec generations. That pins the streaming graph to the batch graph, so
anything else that shows up is the cache and only the cache.

**Chunked, the error is ~1.2e-3 and flat.** Over 853 frames (68 s) of real
generation:

| chunk | rel | max\|d\| |
|---|---|---|
| 1 (0.08 s) | 1.44e-3 | 1.33e-2 |
| 8 (0.64 s) | 1.60e-3 | 1.36e-2 |
| 32 (2.56 s) | 1.35e-3 | 1.38e-2 |
| 128 (10.2 s) | 1.23e-3 | 9.41e-3 |

The point is not the magnitude but that it does not grow: frames 768..776 sit at
1.7e-3, the same as frames 0..8, long after every stage's window has turned over
several times. A cache that were too short would be clean at the start and
degrade once the window filled. For scale, stateless chunking at these widths
sits at 1.144 — larger than the signal.

The residual is attention blocking, the same effect `moss-codec-causality`
measures at ~1e-3 for a prefix decode: the batch path splits queries at
`ATTN_CHUNK`, the stream splits them at chunk boundaries, and flash attention
sums a different partition of identical terms.

End to end, at a fixed seed, both families produce **identical codes** streamed
or not — only the codec path differs:

| family | frames | rel | correlation |
|---|---|---|---|
| `moss_tts_local` (48 kHz stereo, 6 stages) | 51 | 3.5e-3 | 0.99999398 |
| `moss_tts_delay` (24 kHz mono, 4 stages) | 44 | 6.7e-4 | 0.99999978 |

## The delay family streams, but not comfortably

Worth knowing before pointing a player at it. Two things work against
`moss_tts_delay` that do not affect MOSS-TTS-Local:

**The ramp delays the first frame.** Codebook *i* is staggered *i* steps behind
codebook 0, so no frame is complete until `n_vq` steps have run — 32 of them.
A measured run spent 78 steps producing 44 audio frames, and first audio landed
at 2.03 s against 3.55 s for the batch path. Still a win, but a much smaller one
than the local family's 0.57 s against 2.54 s.

**It generates at roughly real time.** That same run produced 3.52 s of audio in
3.93 s — 0.90x. A player started at first audio would slowly fall behind and
eventually starve, so a client for this family should buffer rather than play
straight through. MOSS-TTS-Local has real headroom by comparison, delivering
4.8 s of audio in 3.3 s.

`drain()` handles the ramp without knowing about it: polling is driven off
generation steps, and `extract_audio_codes()` reports only frames whose last
codebook has actually been sampled.

## What it costs

Streaming trades throughput for latency: the codec now runs inline with
generation instead of once at the end, so it stalls the LM. Medians over 3–6
server requests, 60 frames (4.8 s of audio):

| chunk | granularity | codec | first audio | total | vs batch |
|---|---|---|---|---|---|
| batch | — | 0.13 s | 2.54 s | 2.54 s | baseline |
| 4 | 0.32 s | 0.96 s | **0.33 s** | 3.06 s | +20% |
| 8 | 0.64 s | 1.03 s | 0.57 s | 3.31 s | +31% |
| 16 | 1.28 s | 0.36 s | 0.83 s | 2.72 s | +7% |
| 32 | 2.56 s | 0.24 s | 1.46 s | 2.63 s | +4% |
| 64 | 5.12 s | 0.17 s | 2.60 s | 2.61 s | +3% |

The knee is at 16 frames. Below it the fixed per-push cost — graph build,
allocator, the 177 MB cache upload, and roughly 3700 kernel dispatches — starts
to dominate the ~2.3 s the LM itself takes; above it the overhead disappears but
so does most of the latency win. The default is 8: latency is the point of the
feature, and 0.57 s against 2.54 s is the whole reason to turn it on.

Note that audio still arrives **faster than real time at every chunk size** —
4.8 s of audio delivered in 3.3 s at the default — so playback never starves
once it starts. The cost is server throughput, not smoothness.

Measure before assuming these transfer: the per-push overhead is dominated by
dispatch count and driver behaviour, and the single-sample runs that produced
these tables varied by ±10% until averaged.

## Streaming responses are PCM, not WAV

A RIFF header states its own data length in its first 44 bytes, and that is not
known until generation stops. The server rejects `stream=true` with
`response_format=wav` rather than emit a header that lies, and reports
`X-MOSS-Sample-Rate` and `X-MOSS-Channels` instead.

One other consequence of answering before the audio exists: the status line goes
out before generation starts, so a mid-stream failure truncates the response
rather than becoming a 500. Requests are fully validated before the server
commits to streaming, which leaves model-level failures as the only case, and
those are logged server-side.
