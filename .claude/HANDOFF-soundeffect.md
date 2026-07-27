# Handoff: MOSS-SoundEffect-v2.0 support

Branch: `feat/moss-tts-local` (off `main`). Everything below is committed.

## TL;DR

Two of the three model families are done. **MOSS-TTS-Local-Transformer-v1.5 is
finished, verified and published**; **MOSS-SoundEffect-v2.0 has its converter
done** and needs three GGML pieces written: the DiT graph, the DAC VAE decoder,
and the flow-matching sampler.

Everything structural has already been verified against the real checkpoints —
none of it needs re-deriving. Take the numbers in this document as ground truth
unless the code disagrees, in which case trust the checkpoint.

## Commits on this branch

```
e8f0467  feat(convert): support MOSS-SoundEffect-v2.0
db7bfe4  feat(cli,tests): sampling flags, code dump, and a depth-transformer probe
747b3a6  feat(convert): support the moss_tts_local family
ff26e54  feat(codec): MOSS-Audio-Tokenizer-v2 + chunked sliding-window attention
21ff13f  feat(core): model-family abstraction + MOSS-TTS-Local generation
c7409a9  feat(wav): channel-aware WAV I/O
```

Each was checked to build individually — the history is bisectable. Build with
`cmake -B build -DGGML_VULKAN=ON && cmake --build build -j`.

## Files on disk

| What | Where |
|---|---|
| HF sources (all three models) | `/devel/models/models/openmoss-src/` |
| GGUFs | `/devel/models/models/openmoss/` |
| SoundEffect backbone GGUF | `moss-soundeffect-2.0.gguf` (4.07 GB, Qwen3-1.7B text encoder) |
| SoundEffect sidecar | `moss-soundeffect-2.0.extras.gguf` (2.98 GB, 825 DiT + 148 VAE tensors) |
| `llama-quantize` | `/devel/tools/llama.cpp/build/bin/llama-quantize` (not built in-tree) |
| Reference Python sources | scratchpad `se2src/` — may be gone; re-fetch from `github.com/OpenMOSS/MOSS-TTS/moss_soundeffect_v2` |

Published: <https://huggingface.co/ilintar/moss-tts-local-gguf> (MOSS-TTS-Local, Q8).

## What is done

### Family abstraction (`include/openmoss/frame_decoder.h`, `src/model.cpp`)

`moss.architecture` is a string KV read at load, falling back to
`general.architecture` and then to `moss_tts_delay`, so **GGUFs published before
this work still load untouched**. `Arch` is an enum on `ModelDims`.

`IFrameDecoder::step()` takes the raw backbone hidden state, not pre-computed
audio logits. That is forced by MOSS-TTS-Local: it must sample code *k* before it
can compute logits for *k+1*, so head invocation belongs to the decoder.

### MOSS-TTS-Local — complete

Verified four independent ways, all exact: prompt ids vs the reference processor
(69/69), depth transformer on a synthetic hidden state (12/12), on the
reference's real hidden state (12/12), and full-pipeline greedy Q8 vs fp32
reference (12/12). Codec round-trip 0.995 at 5/40/120 s. Gemma 4 E4B transcribed
three generated samples word-for-word.

### MOSS-SoundEffect converter — complete (`e8f0467`)

Produces both GGUFs. See the commit message for the details; the load-bearing
ones are repeated under "Traps" below.

## What is left

Tracked as SFX-2 … SFX-5. Roughly in order:

### SFX-2 — WanAudioModel DiT graph

30 blocks, dim 1536, 12 heads, head_dim 128, ffn 8960, eps 1e-6. Per block:

```
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    = (block.mod[1,6,1536] + t_mod[B,6,1536]).chunk(6, dim=1)

x = x + gate_msa * self_attn( norm1(x)*(1+scale_msa) + shift_msa )
x = x + cross_attn( norm3(x), context )          # NOTE: no gate here
x = x + gate_mlp * ffn( norm2(x)*(1+scale_mlp) + shift_mlp )
```

Verified against the reference source:

* `norm1` and `norm2` are `LayerNorm(eps, elementwise_affine=False)` — **no
  parameters, and absent from the checkpoint**. Only `norm3` is affine, and it
  is stored on disk as `norm2` (an off-by-one rename in the diffusers export).
  The converter already maps it to `moss.dit.blk.N.norm3.*`.
* Self-attention: `q = norm_q(W_q x)` then RoPE; same for k; v is neither
  normed nor rotated. `norm_q`/`norm_k` are **RMSNorm over the full 1536 axis,
  not per-head** — apply them before reshaping into heads.
* RoPE is interleaved-pair → `GGML_ROPE_TYPE_NORMAL` (mode 0), theta 10000,
  n_dims 128, positions 0..f-1. The reference chunks `freqs_cis` into 3 and
  concatenates them back, which is an exact identity for `vae_type="dac"`.
* Cross-attention: no RoPE, **no mask** — all 512 text slots are attended,
  including padding.
* FFN is `Linear → GELU(approximate='tanh') → Linear`. ggml's `ggml_gelu` *is*
  the tanh approximation, so use it, not `ggml_gelu_erf`.
* `head(x, t)` takes the plain time embedding **`t`, not `t_mod`**, with its own
  `[1, 2, 1536]` modulation: `shift, scale = (mod + t.unsqueeze(1)).chunk(2)`,
  index 0 = shift, 1 = scale.

Time embedding (compute host-side, it is one 256-vector per step):

```
freqs   = 10000 ** (-i/128)  for i in 0..127      # accumulate in float64
sinusoid = outer(timestep, freqs)
emb     = cat([cos(sinusoid), sin(sinusoid)])     # COS FIRST, then sin
t       = t_emb.2( silu( t_emb.1(emb) ) )         # [1536]
t_mod   = t_proj( silu(t) ).unflatten(6, 1536)    # [6, 1536]
```

`time_projection` is `Sequential(SiLU(), Linear(...))` — the SiLU comes first, so
the stored weight is index `.1`. The converter names it `moss.dit.t_proj.*`.

Patchify is `patch_size=[1]`, i.e. `Conv1d(128, 1536, k=1)` = a per-frame affine
map; unpatchify is a no-op reshape.

### SFX-3 — DAC VAE decoder

`continuous=True`: **no quantizer, no codebooks** (verified — zero quantizer keys
in the checkpoint). Mono. Total upsample 960 → 50 latent frames/s at 48 kHz.

```
post_quant_conv  Conv1d(128->128, k=1)        # plain, NOT weight-normed
dec.in           WNConv1d(128->2048, k=7, p=3)
dec.{1..5}       Snake -> WNConvTranspose1d -> 3x ResidualUnit(dil 1,3,9)
dec.out.snake    Snake(64)
dec.out          WNConv1d(64->1, k=7, p=3)
                 tanh
```

Per block B=1..5: `(C_in, C_out, K, stride, pad, out_pad)` =
`(2048,1024,16,8,4,0) (1024,512,10,5,3,1) (512,256,8,4,2,0) (256,128,6,3,2,1)
(128,64,4,2,1,0)`. `ResidualUnit(dim, d)` is
`Snake -> WNConv1d(k=7, dilation=d, pad=3d) -> Snake -> WNConv1d(k=1)`, added to
its input.

Snake, per-channel alpha: `y = x + sin(a*x)^2 / (a + 1e-9)`. The converter ships
both `alpha` and a precomputed `inv` = `1/(a+1e-9)`, **both f32** (see Traps).
Emitting exactly `MUL -> SIN -> SQR -> MUL -> ADD` with 2-D x and a separate inv
tensor may hit llama.cpp's fused Snake CUDA kernel.

**`ggml_conv_transpose_1d` asserts `p0 == 0`, `d0 == 1`, and a 2-D (batch-1)
input.** Run it unpadded and crop: ggml gives `L = (T-1)s + k`, PyTorch gives
`L = (T-1)s - 2p + (k-1) + 1 + op`, so crop `p` from the left and `p - op` from
the right. Worked out per block:

| block | s | k | p | op | ggml L | crop L/R | final L |
|---|---|---|---|---|---|---|---|
| 1 | 8 | 16 | 4 | 0 | 12008 | 4 / 4 | 12000 |
| 2 | 5 | 10 | 3 | 1 | 60005 | 3 / 2 | 60000 |
| 3 | 4 | 8 | 2 | 0 | 240004 | 2 / 2 | 240000 |
| 4 | 3 | 6 | 2 | 1 | 720003 | 2 / 1 | 720000 |
| 5 | 2 | 4 | 1 | 0 | 1440002 | 1 / 1 | 1440000 |

(for the full 30 s / 1500-latent case).

Two memory problems to plan for, both like the codec's mask issue:

1. Block 5 outputs `64 x 1,440,000` fp32 = 368 MB, and each ResidualUnit needs
   several live temporaries that size. **Chunk along time** with an overlap of
   at least the receptive field.
2. `ggml_conv_1d` goes through `ggml_im2col` with `GGML_TYPE_F16` hardcoded. For
   k=7, IC=64, T=1.44M that buffer is 1.29 GB *and* silently drops activations
   to fp16 near the final tanh. Prefer `ggml_im2col(..., GGML_TYPE_F32)` +
   `ggml_mul_mat` by hand, or chunk.

### SFX-4 — flow matching + pipeline wiring

```
sigmas    = linspace(1, 0, N+1)[:-1]                 # extra_one_step=True
sigmas    = shift*s / (1 + (shift-1)*s)              # shift = 5.0
timesteps = sigmas * 1000
x        += v * (sigma_next - sigma)                 # Euler; last step -> 0
```

Initial latent is pure `randn` (sigma_0 is exactly 1.0, so no scaling).
Defaults: `num_inference_steps=100`, `cfg_scale=4.0`, `sigma_shift=5.0`,
`seconds=10.0`.

CFG: `pred = neg + scale * (pos - neg)`, both branches in fp32. Two full DiT
forwards per step, so 200 for the default.

**Two shortcuts worth taking, both verified:**

* The negative prompt is the empty string, which tokenizes to *zero* real tokens
  and is then hard-zeroed — so the unconditional context is **exactly
  `zeros([1, 512, 2048])`**. Skip Qwen3 entirely for that branch, and precompute
  its cross-attention K/V once for the whole run (the text embedding does not
  change across the 100 steps — cache the positive branch's K/V too).
* **Duration is purely textual.** The prompt gets `f" duration: {seconds:.1f}s"`
  appended (single ASCII space, one decimal, trailing `s`), and the DiT *always*
  generates 1500 latents = 30 s regardless; the waveform is cropped to
  `48000 * seconds` afterwards. Do not try to shorten the latent.

Text conditioning: Qwen3-1.7B run as a causal LM, `hidden_states[-1]` (i.e. after
the final RMSNorm), no pooling, no projection (text_dim 2048 already equals its
hidden size — the 2048→1536 projection lives in the DiT as `txt_emb`).
Tokenized with `padding='max_length', max_length=512, truncation=True`, **no BOS
and no EOS**, right-padded with 151643, then rows past the real length are
zeroed.

Wiring: `Model::load` currently *requires* a libllama backbone
(`src/model.cpp`, the `llama_model_load_from_file` throw). For SoundEffect the
backbone genuinely is a Qwen3 GGUF, so this happens to be fine — but the
generation entry point is completely different. Add a `generate_flow()` rather
than trying to bend `generate()`; they share only the model loader, the aux
backend, the tokenizer and the CLI/server shell.

### SFX-5 — verify, quantize, ship

Quantize the DiT to Q8 but **keep `t_emb`, `t_proj`, every `mod`
(`scale_shift_table`), the head, and the whole VAE decoder at F16/F32** — the
modulation path is low-rank and precision-sensitive.

Ship to a new `ilintar/moss-soundeffect-gguf`. **Confirm with the user before
uploading** — it is public and name-attributed.

## Traps already hit (do not rediscover these)

1. **Snake reciprocals must stay f32.** Three of the 15936 decoder alphas are
   ~8e-6, giving `1/(a+1e-9)` up to 1.2e5 — past f16's 65504 ceiling. Cast to
   f16 they become `inf` and silently poison the waveform. The sidecar writer
   now preserves f32 when a collector chooses it; keep it that way.
2. **Neither conv needs transposing.** `ggml_conv_1d` reads its kernel as
   ne=(K, IC, OC) and PyTorch's `[OC, IC, K]` gives that; `ggml_conv_transpose_1d`
   wants ne=(K, OC, IC) and PyTorch's `[IC, OC, K]` gives that. Confirmed in the
   produced GGUF.
3. **Weight-norm's `g` axis differs between the conv flavours** — output channels
   for Conv1d, *input* channels for ConvTranspose1d — but both put the indexed
   axis first, so one code path handles both.
4. **`resolve_model_dir` / `detect_family`** must handle a diffusers layout:
   SoundEffect has `model_index.json` and no top-level `config.json`.
5. From the TTS-Local work, still relevant: a projection's *absence* must never be
   inferred from matching dimensions (`src/codec.cpp` used to, and v2 would have
   silently dropped 10 learned square matrices).

## Verifying against the reference

`flash_attn` is not installed and MOSS configs request `flash_attention_2`. The
setting is captured **per-module at construction**, so overriding the config
before `from_pretrained` does nothing. Patch the instances:

```python
for mod in m.modules():
    if hasattr(mod, "attn_implementation"):
        mod.attn_implementation = "sdpa"
```

Compare at seams, not on generated audio — greedy output diverges from f16
numerics alone (several codebooks sit on ties as tight as 0.009). Good seams:
prompt ids, one block forward given a fixed input, the final latent, the decoded
waveform.

`tests/moss_local_probe.cpp` is the pattern to copy: feed a fixed tensor from a
raw f32 file into one isolated subgraph so a mismatch cannot be blamed on
upstream drift.

## Listening to output

A Lemonade server runs on `:8000` with `gemma-4-E4B-it-GGUF-Q6_K`, which accepts
audio and transcribes accurately. Load it with
`POST /api/v1/load {"model_name": "gemma-4-E4B-it-GGUF-Q6_K"}` and send
`{"type": "input_audio", "input_audio": {"data": <b64 wav>, "format": "wav"}}`
to `/api/v1/chat/completions`. Convert to **16 kHz mono** first (`audioop` is
gone in Python 3.13+; there is a hand-rolled converter in the scratchpad's
`ask_audio.py`).

Caveats: it intermittently claims "no audio was provided" for clips it just
transcribed — **always run a known-good control clip first** so a refusal is not
mistaken for a bad sample. It is also unreliable at describing *voice
characteristics* (it tends to transcribe instead), so it could not settle
whether voice cloning transfers speaker identity. Multiple `input_audio` parts
in one request do work, and an A/B "SAME or DIFFERENT" prompt passes its own
sanity check, but the judge is too noisy to rely on.

## Also still open

**Phase 6, the server option surface** — deliberately deferred to last at the
user's direction. The CLI half is done (per-channel sampling flags, `--seed`,
`--dump-codes`); the HTTP half is untouched. It needs, per the sglang-omni
cookbooks: `references[]` with `ref_audio`/`ref_text` (the transcript path needs
*continuation mode* — reference audio in the assistant channel, slot 151656, no
`audio_end`), flattened per-channel sampling on `/v1/audio/speech`,
`token_count` plus the inline `${token:N}` prefix, `response_format: pcm`,
`GET /v1/models`, and a voices registry at `/v1/audio/voices` (no upstream
standard exists — `voice` is always `"default"` in every cookbook; the vLLM-Omni
convention is the closest thing).

Streaming needs its own design pass: the codec's summed receptive field is 35 s,
so block decode with warm-up is not viable and it wants per-stage KV caching.
