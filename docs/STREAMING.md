# Streaming: design pass

Conclusion up front: streaming is **achievable**, the codec decoder is the only
missing piece, and the right architecture is a **per-stage KV cache**. Stateless
chunked decoding — the cheap option — is not viable at any useful granularity,
and this document shows why with measured numbers rather than an estimate.

Nothing here is implemented yet. It is written so the next person does not have
to re-derive the constraint.

## What already streams

`generate()` produces audio frames one at a time: the backbone decodes a token,
the frame decoder turns it into `n_vq` codes, and the loop repeats.
`pipeline.h` has carried a `StreamCallback` typedef since the beginning, unused.
So the LM half is already incremental — only the codec is batch.

The measured rates on a Radeon 8060S, MOSS-TTS-Local Q8:

| stage | throughput | vs real time |
|---|---|---|
| backbone + depth transformer | 21.8 frames/s | 1.74x |
| codec decode | ~30x real time | 30x |

Two consequences. First, **the LM is the constraint**, not the codec — a
streaming design that made the codec incremental but still waited for the whole
utterance from the LM would gain nothing. Second, for a *complete* utterance
you already get audio faster than you can play it (8.5 s of speech in 5.2 s), so
streaming buys **latency, not throughput**. It matters for interactive use and
not at all for batch.

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
frames:

| chunk | multiplier | effective speed | time to first audio |
|---|---|---|---|
| 1 s | 36x | 1.2x real time — **too slow** | — |
| 2 s | 18.5x | 0.62x | ~2.5 s |
| 5 s | 8.4x | 0.28x | ~4.2 s |

1 s chunks cannot keep up. 5 s chunks can, but 5 s of granularity and ~4 s to
first audio is barely better than just waiting for the whole utterance. This is
the option to reject.

## The architecture to build

Keep per-stage K/V caches and push new frames through, so nothing is
recomputed. Cost per frame becomes O(1) and granularity drops to one frame
(80 ms).

`src/local_transformer.cpp` is the pattern to copy — it already does exactly
this for the depth transformer, holding the cache on the host and handing the
prefix to each step's graph as an input, which avoids in-graph cache mutation.
The codec needs the same shape, six times over, with one wrinkle: the
`PatchedPretransform` between stages multiplies the frame rate, so one input
frame becomes 2 at the next stage, 4 at the one after, and so on up to 240 at
the final unpatch. That is a fixed expansion and stays causal, so it does not
change the design — it only means each stage's cache is a different length.

Cache size, at the trained window per stage:

| stage | layers | d_model | window | floats |
|---|---|---|---|---|
| dec.0 | 32 | 1280 | 125 | 10.2 M |
| dec.2 | 12 | 768 | 250 | 4.6 M |
| dec.4/6/8/10 | 12 | 768 | 400 | 7.4 M each |
| **total** | | | | **44.3 M** |

177 MB at f32, 89 MB at f16 — the same order as the codec weights themselves,
and unremarkable next to the backbone.

Expected result: first audio after roughly one frame of LM work plus one
incremental decode, so ~0.5 s rather than the full 5.2 s, with the LM's 1.74x
real-time rate as the steady-state ceiling.

## MOSS-SoundEffect cannot stream

Not a limitation of the port. Flow matching integrates the whole latent
simultaneously — every solver step refines all 1500 frames at once, and no
prefix of the latent is usable until the last step finishes. There is no
incremental structure to exploit.

What *is* worth exposing is progress: `generate_sound_effect()` already takes a
`SoundEffectProgress` callback, and the server could surface it over
server-sent events so a caller can show a bar rather than waiting blind through
~90 s. The waveform itself only exists at the end.

The one real speedup available is already taken: the VAE now decodes only the
latent frames the requested duration needs, which is why a 3 s request spends
0.99 s in the decoder instead of 7.8 s.
