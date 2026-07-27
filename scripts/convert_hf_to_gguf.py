#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Convert a MOSS TTS model (+ its MOSS-Audio-Tokenizer) from HuggingFace
safetensors into GGUF files consumable by openmoss-ggml.

Supported families (auto-detected from config.json's ``model_type``, or forced
with ``--arch``):

    moss_tts_delay   MOSS-TTS / MOSS-TTS-v1.5 / MOSS-VoiceGenerator
                     Qwen3 backbone, 32 RVQ codebooks scheduled with a delay
                     pattern, MOSS-Audio-Tokenizer v1 (24 kHz mono).

    moss_tts_local   MOSS-TTS-Local-Transformer-v1.5
                     Qwen3-4B backbone, 12 RVQ codebooks emitted per frame by a
                     1-layer local ("depth") transformer, MOSS-Audio-Tokenizer
                     v2 (48 kHz stereo).

    moss_soundeffect MOSS-SoundEffect-v2.0
                     Not autoregressive at all: a 30-block Wan-style Diffusion
                     Transformer sampled with flow matching, conditioned on a
                     Qwen3-1.7B text encoder, decoded by a continuous DAC KL-VAE
                     (48 kHz mono). Detected from model_index.json rather than
                     config.json, since it ships as a diffusers pipeline.

Layout produced
---------------
The output GGUF is a *valid Qwen3 GGUF for libllama* (so the backbone loads via
`llama_model_load_from_file` without any patching) **plus** extra tensors and
KV entries under the `moss.*` namespace that libllama ignores but our C++
reader picks up:

    Qwen3 backbone (loaded by libllama)
        token_embd.weight, blk.{N}.*, output_norm.weight, output.weight, ...

    MOSS audio extension
        moss.audio_embed.{0..n_vq-1}.weight       (audio_vocab_size+1, hidden)
        moss.audio_head.{0..n_vq-1}.weight        (audio_vocab_size+1, hidden)

    Optional codec (added when --codec is supplied)
        moss.codec.encoder.*                      verbatim from MossAudioTokenizer
        moss.codec.decoder.*
        moss.codec.quantizer.*

    KV metadata
        moss.architecture           "moss_tts_delay"
        moss.n_vq                   32
        moss.audio_vocab_size       1024
        moss.audio_pad_code         1024
        moss.sampling_rate          24000
        moss.frame_rate             12.5  (sampling_rate / downsample_rate)
        moss.downsample_rate        1920
        moss.token.audio_start      151652
        moss.token.audio_end        151653
        moss.token.audio_user_slot  151654
        moss.token.audio_gen_slot   151656
        moss.token.audio_delay_slot 151662
        moss.token.im_start         151644
        moss.token.im_end           151645
        moss.codec.present          (bool)

Pipeline (high level)
---------------------
1. snapshot-download MOSS-TTS (and codec, if requested)
2. extract a Qwen3-shaped checkpoint (rename `language_model.*` → `model.*`,
   `lm_heads.0.weight` → `lm_head.weight`) into a scratch dir
3. run llama.cpp's `convert_hf_to_gguf.py` on that scratch dir → vanilla Qwen3 GGUF
4. read that GGUF back via gguf-py
5. read `emb_ext.*` and `lm_heads.{1..n_vq}` from the original shards
6. (optional) read the codec safetensors, prefix every tensor with `moss.codec.`
7. write a combined GGUF (vanilla Qwen3 tensors + ours + moss.* KV)

Usage
-----
    python scripts/convert_hf_to_gguf.py \\
        --moss-tts OpenMOSS-Team/MOSS-TTS \\
        --output   weights/moss-tts.gguf

    # With the codec included (will be much larger; ~13 GB):
    python scripts/convert_hf_to_gguf.py \\
        --moss-tts OpenMOSS-Team/MOSS-TTS \\
        --codec    OpenMOSS-Team/MOSS-Audio-Tokenizer \\
        --output   weights/moss-tts.gguf

    # Local source dirs (skip download):
    python scripts/convert_hf_to_gguf.py \\
        --moss-tts /path/to/moss-tts \\
        --codec    /path/to/moss-audio-tokenizer \\
        --output   weights/moss-tts.gguf

Quantization
------------
This script always emits F16 (or whatever you pass via --backbone-dtype). To
quantize the backbone after conversion, use llama.cpp's `llama-quantize` on the
output file — the moss.* tensors stay untouched because the quantizer only
recognises Qwen3 tensor names.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

log = logging.getLogger("moss-convert")


# ────────────────────────────────────────────────────────────────────────────
# Backbone extraction (MossTTSDelay → Qwen3)
# ────────────────────────────────────────────────────────────────────────────

def remap_backbone_delay(name: str) -> str | None:
    """MossTTSDelay tensor name → Qwen3ForCausalLM convention. Returns None
    when the tensor is not a backbone tensor.
    """
    if name.startswith("language_model."):
        # language_model.embed_tokens.weight → model.embed_tokens.weight
        # language_model.layers.N.*          → model.layers.N.*
        # language_model.norm.weight         → model.norm.weight
        return "model." + name[len("language_model."):]
    if name == "lm_heads.0.weight":
        return "lm_head.weight"
    return None


def remap_backbone_local(name: str) -> str | None:
    """MossTTSLocal tensor name → Qwen3ForCausalLM convention.

    `text_lm_head.weight` is deliberately dropped. MossTTSLocalModel.tie_weights
    binds it to `transformer.embed_tokens.weight` (and the two blobs are
    byte-identical in the checkpoint), while the Qwen3 config carries
    tie_word_embeddings=true — so llama.cpp reuses token_embd as the output
    projection and shipping the duplicate would waste 389 M params.
    """
    if name.startswith("transformer."):
        return "model." + name[len("transformer."):]
    return None


# ────────────────────────────────────────────────────────────────────────────
# Minimal self-contained safetensors reader.
#
# The safetensors file format is:
#   [u64 little-endian header_size]
#   [header_size bytes of JSON]    — { tensor_name: { dtype, shape, data_offsets } | "__metadata__": {...} }
#   [tensor data, contiguous, in the order described by the offsets]
#
# We mmap the file once per shard and serve numpy views over each tensor's
# byte range, doing dtype conversion in pure numpy (so no torch / ml_dtypes
# dependency, and bf16 is handled natively).
# ────────────────────────────────────────────────────────────────────────────

import mmap

_SAFETENSORS_DTYPE = {
    "F64":  ("float64", 8),
    "F32":  ("float32", 4),
    "F16":  ("float16", 2),
    "BF16": ("bfloat16", 2),         # not a real numpy dtype — we convert to f32 first
    "F8_E4M3": ("float8_e4m3", 1),   # we don't need these for MOSS-TTS but list for completeness
    "F8_E5M2": ("float8_e5m2", 1),
    "I64":  ("int64", 8),
    "I32":  ("int32", 4),
    "I16":  ("int16", 2),
    "I8":   ("int8", 1),
    "U64":  ("uint64", 8),
    "U32":  ("uint32", 4),
    "U16":  ("uint16", 2),
    "U8":   ("uint8", 1),
    "BOOL": ("bool", 1),
}


class STShard:
    """Read-only handle on a single safetensors shard."""

    def __init__(self, path: Path):
        self.path = path
        self._fp  = open(path, "rb")
        header_size = int.from_bytes(self._fp.read(8), "little")
        header_bytes = self._fp.read(header_size)
        self._header = json.loads(header_bytes.decode("utf-8"))
        self._data_start = 8 + header_size
        # mmap the whole file lazily — we'll only access slices.
        if sys.platform == "win32":
            self._mm = mmap.mmap(self._fp.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            self._mm = mmap.mmap(self._fp.fileno(), 0, prot=mmap.PROT_READ)

    def keys(self):
        return [k for k in self._header.keys() if k != "__metadata__"]

    def dtype_of(self, tname: str) -> str:
        return self._header[tname]["dtype"]

    def shape_of(self, tname: str) -> tuple[int, ...]:
        return tuple(int(d) for d in self._header[tname]["shape"])

    def read_raw(self, tname: str) -> tuple[bytes, str, tuple[int, ...]]:
        """Return (raw_bytes_copy, dtype_string, shape).

        Returns a *copy* (not a memoryview) so callers can pass the result
        around without keeping the mmap pinned — important because Python's
        ``mmap.close()`` raises ``BufferError`` if any view is alive.
        """
        info = self._header[tname]
        dtype = info["dtype"]
        shape = self.shape_of(tname)
        beg, end = info["data_offsets"]
        beg += self._data_start
        end += self._data_start
        # Slice via memoryview to avoid double-copying through bytes(self._mm),
        # then materialise a bytes object (which itself drops the memoryview).
        mv = memoryview(self._mm)[beg:end]
        try:
            return bytes(mv), dtype, shape
        finally:
            mv.release()

    def get_f16(self, tname: str) -> "np.ndarray":
        """Read a tensor and return it as a numpy float16 array.

        Handles bf16 via bit-shift; passes through f16; casts other floats.
        """
        raw, dtype, shape = self.read_raw(tname)
        if dtype == "F16":
            return np.frombuffer(raw, dtype=np.float16).reshape(shape).copy()
        if dtype == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16)
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
            return f32.astype(np.float16).reshape(shape)
        if dtype == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(shape).astype(np.float16)
        if dtype == "F64":
            return np.frombuffer(raw, dtype=np.float64).reshape(shape).astype(np.float16)
        raise RuntimeError(f"unhandled tensor dtype {dtype} for {tname} ({self.path})")

    def get_native(self, tname: str) -> "np.ndarray":
        """Read a tensor in its source dtype where numpy can represent it.

        bf16 is exposed as the underlying uint16 byte pattern (numpy has no
        native bf16); other dtypes round-trip exactly.
        """
        raw, dtype, shape = self.read_raw(tname)
        np_dtype, _ = _SAFETENSORS_DTYPE[dtype]
        if dtype == "BF16":
            return np.frombuffer(raw, dtype=np.uint16).reshape(shape).copy()
        return np.frombuffer(raw, dtype=np.dtype(np_dtype)).reshape(shape).copy()

    def close(self):
        # Belt-and-braces: drop any cached attribute that could hold a view.
        try:
            self._mm.close()
        finally:
            self._fp.close()

    def __enter__(self): return self
    def __exit__(self, *a): self.close()


def load_safetensors_index(model_dir: Path) -> dict[str, str]:
    """Return weight_map { tensor_name -> shard_filename }."""
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        with idx_path.open() as f:
            return json.load(f)["weight_map"]
    single = model_dir / "model.safetensors"
    if single.exists():
        with STShard(single) as sf:
            return {k: "model.safetensors" for k in sf.keys()}
    raise FileNotFoundError(f"No safetensors found in {model_dir}")


def _save_safetensors_torchfree(tensors: dict, path: Path) -> None:
    """Write a dict of {name -> (dtype_str, shape, raw_bytes)} as a safetensors file.

    Bypasses torch — uses our own minimal writer so the converter works on a
    system where torch is broken. The output is identical to what
    safetensors.torch.save_file would produce.
    """
    # Build the JSON header.
    header: dict = {}
    offset = 0
    blobs: list[bytes] = []
    for name in sorted(tensors.keys()):
        dtype, shape, blob = tensors[name]
        sz = len(blob)
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + sz]}
        offset += sz
        blobs.append(bytes(blob))

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # safetensors requires header to be a multiple of 8 bytes (some loaders
    # tolerate any size, but huggingface_hub wants 8-byte alignment).
    pad = (-len(header_bytes)) % 8
    if pad:
        header_bytes += b" " * pad

    with path.open("wb") as f:
        f.write(len(header_bytes).to_bytes(8, "little"))
        f.write(header_bytes)
        for blob in blobs:
            f.write(blob)


def extract_qwen3_backbone(moss_dir: Path, out_dir: Path,
                            fam: "Family", moss_config: dict) -> None:
    """Materialise the backbone part of a MOSS checkpoint into ``out_dir`` as a
    self-contained Qwen3ForCausalLM checkpoint.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_map = load_safetensors_index(moss_dir)
    shard_to_tensors: dict[str, list[str]] = defaultdict(list)
    for tname, shard in weight_map.items():
        shard_to_tensors[shard].append(tname)

    MAX_SHARD = 5 * 1024 ** 3  # 5 GB
    bucket: dict = {}                 # qwen_name -> (dtype_str, shape, raw_bytes)
    bucket_bytes = 0
    shard_idx = 0
    saved: list[str] = []
    new_weight_map: dict[str, str] = {}

    def flush():
        nonlocal bucket, bucket_bytes, shard_idx
        if not bucket:
            return
        shard_idx += 1
        shard_name = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
        log.info("  writing %s (%d tensors, %.2f GB)",
                 shard_name, len(bucket), bucket_bytes / 1e9)
        _save_safetensors_torchfree(bucket, out_dir / shard_name)
        for tname in bucket:
            new_weight_map[tname] = shard_name
        saved.append(shard_name)
        bucket = {}
        bucket_bytes = 0

    for shard in sorted(shard_to_tensors):
        log.info("scanning %s", shard)
        with STShard(moss_dir / shard) as sf:
            for tname in sorted(shard_to_tensors[shard]):
                qwen_name = fam.remap_backbone(tname)
                if qwen_name is None:
                    continue
                blob, dtype, shape = sf.read_raw(tname)
                size = len(blob)
                if bucket and bucket_bytes + size > MAX_SHARD:
                    flush()
                bucket[qwen_name] = (dtype, shape, blob)
                bucket_bytes += size
    flush()

    # Rename shards with the correct total count.
    total = len(saved)
    if total == 0:
        raise RuntimeError("no backbone tensors extracted — wrong source model?")
    for i, name in enumerate(saved, 1):
        new = f"model-{i:05d}-of-{total:05d}.safetensors"
        if name != new:
            target = out_dir / new
            if target.exists():
                target.unlink()
            (out_dir / name).rename(target)
            for tn in [k for k, v in new_weight_map.items() if v == name]:
                new_weight_map[tn] = new

    if total == 1:
        # No index needed for single-shard; rename to the canonical name.
        only = next(iter(new_weight_map.values()))
        target = out_dir / "model.safetensors"
        if target.exists():
            target.unlink()
        (out_dir / only).rename(target)
    else:
        total_size = sum((out_dir / s).stat().st_size for s in set(new_weight_map.values()))
        with (out_dir / "model.safetensors.index.json").open("w") as f:
            json.dump({"metadata": {"total_size": total_size},
                       "weight_map": new_weight_map}, f, indent=2, sort_keys=True)

    # Qwen3 config.
    lang = None
    for key in fam.backbone_config_keys:
        if key in moss_config:
            lang = dict(moss_config[key])
            break
    if lang is None:
        raise RuntimeError(
            f"{fam.arch}: none of {fam.backbone_config_keys} present in config.json")
    lang.pop("_name_or_path", None)
    lang["architectures"] = ["Qwen3ForCausalLM"]
    lang["model_type"] = "qwen3"
    lang.setdefault("torch_dtype", "bfloat16")
    with (out_dir / "config.json").open("w") as f:
        json.dump(lang, f, indent=2)

    # Tokenizer files (all of them — convert_hf_to_gguf reads several).
    for fn in ("tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "added_tokens.json",
               "merges.txt", "vocab.json", "chat_template.jinja"):
        src = moss_dir / fn
        if src.exists():
            shutil.copy2(src, out_dir / fn)


def assemble_backbone_dir(moss_dir: Path, out_dir: Path, fam: "Family") -> None:
    """For families whose backbone already *is* a standalone HF checkpoint in a
    subdirectory, just place it next to the tokenizer files.

    MOSS-SoundEffect keeps its Qwen3-1.7B text encoder in text_encoder/ with the
    tokenizer in a sibling tokenizer/ dir; llama.cpp's converter wants them in
    one directory.
    """
    src = moss_dir / fam.backbone_subdir
    if not (src / "config.json").exists():
        raise RuntimeError(f"{fam.arch}: no config.json in {src}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, out_dir / f.name)
    tok = moss_dir / "tokenizer"
    if tok.is_dir():
        for f in tok.iterdir():
            if f.is_file():
                shutil.copy2(f, out_dir / f.name)
    log.info("  assembled backbone dir from %s + %s", src.name, tok.name)


# ────────────────────────────────────────────────────────────────────────────
# Stage 2: invoke llama.cpp's converter on the prepared Qwen3 dir
# ────────────────────────────────────────────────────────────────────────────

def run_llama_cpp_converter(qwen3_dir: Path, gguf_path: Path, llama_cpp_dir: Path,
                             dtype: str) -> None:
    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not converter.exists():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found at {converter}")

    cmd = [
        sys.executable, str(converter), str(qwen3_dir),
        "--outfile", str(gguf_path),
        "--outtype", dtype,
    ]
    log.info("running: %s", " ".join(cmd))
    env = dict(os.environ)
    # Make sure we use the gguf-py shipped with llama.cpp for tensor mapping consistency.
    env["PYTHONPATH"] = str(llama_cpp_dir / "gguf-py") + os.pathsep + env.get("PYTHONPATH", "")
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"convert_hf_to_gguf.py exited with {res.returncode}")


# ────────────────────────────────────────────────────────────────────────────
# Stage 3: append moss.* tensors and KV
# ────────────────────────────────────────────────────────────────────────────

def numpy_dtype_for(out_dtype: str):
    return {
        "f16":  np.float16,
        "f32":  np.float32,
        "bf16": np.float32,  # gguf doesn't have bfloat16 in numpy; we promote
    }[out_dtype]


def _shards_by_name(model_dir: Path) -> dict[str, list[str]]:
    weight_map = load_safetensors_index(model_dir)
    by_shard: dict[str, list[str]] = defaultdict(list)
    for tname, shard in weight_map.items():
        by_shard[shard].append(tname)
    return by_shard


def collect_audio_extras_delay(moss_dir: Path, cfg: dict):
    """Yield (gguf_tensor_name, np.ndarray[f16]) for the audio embeddings + heads."""
    by_shard = _shards_by_name(moss_dir)
    for shard in sorted(by_shard):
        with STShard(moss_dir / shard) as sf:
            for tname in sorted(by_shard[shard]):
                if tname.startswith("emb_ext.") and tname.endswith(".weight"):
                    idx = int(tname.split(".")[1])
                    yield f"moss.audio_embed.{idx}.weight", sf.get_f16(tname)
                elif tname.startswith("lm_heads.") and tname.endswith(".weight"):
                    idx = int(tname.split(".")[1])
                    if idx == 0:
                        continue # already mapped to output.weight by stage 1
                    yield f"moss.audio_head.{idx-1}.weight", sf.get_f16(tname)


def collect_audio_extras_local(moss_dir: Path, cfg: dict):
    """Yield (gguf_tensor_name, np.ndarray[f16]) for MOSS-TTS-Local.

    Three differences from the delay family:

    * `audio_lm_heads.{i}` is dropped — tie_weights() binds it to
      `audio_embeddings.{i}`, and the blobs are byte-identical. The C++ side
      uses the embedding table as the head matrix (viewing the first
      audio_vocab_size rows).
    * Each embedding table gains one all-zero row at index `audio_pad_code`.
      Upstream masks the pad code out of the summed input embedding
      (`embedding(safe_ids) * valid_mask`); an explicit zero row reproduces
      that with a plain get_rows and no branch in the graph.
    * The 1-layer local ("depth") transformer and the 2-way stop head ship as
      `moss.local.*` / `moss.local_text_head.weight`. The stop head is
      independently trained — it is NOT a slice of text_lm_head — so it must
      be shipped even though everything else tied is dropped.
    """
    audio_vocab = int(cfg.get("audio_vocab_size", 1024))
    pad_code    = int(cfg.get("audio_pad_code", audio_vocab))
    if pad_code != audio_vocab:
        raise RuntimeError(
            f"moss_tts_local: expected audio_pad_code == audio_vocab_size, got "
            f"{pad_code} != {audio_vocab}; the zero-pad-row trick assumes the pad "
            f"code sits immediately after the codebook entries")

    by_shard = _shards_by_name(moss_dir)
    for shard in sorted(by_shard):
        with STShard(moss_dir / shard) as sf:
            for tname in sorted(by_shard[shard]):
                if tname.startswith("audio_embeddings.") and tname.endswith(".weight"):
                    idx = int(tname.split(".")[1])
                    arr = sf.get_f16(tname)          # (audio_vocab, hidden)
                    if arr.shape[0] != audio_vocab:
                        raise RuntimeError(
                            f"{tname}: expected {audio_vocab} rows, got {arr.shape[0]}")
                    padded = np.zeros((audio_vocab + 1, arr.shape[1]), dtype=np.float16)
                    padded[:audio_vocab] = arr
                    yield f"moss.audio_embed.{idx}.weight", padded
                elif tname.startswith("audio_lm_heads."):
                    continue                          # tied to audio_embeddings
                elif tname == "text_lm_head.weight":
                    continue                          # tied to transformer.embed_tokens
                elif tname.startswith("local_transformer."):
                    yield "moss.local." + tname[len("local_transformer."):], sf.get_f16(tname)
                elif tname == "local_text_lm_head.weight":
                    yield "moss.local_text_head.weight", sf.get_f16(tname)


_CODEC_RENAMES = (
    # (substring, replacement) — applied in order, repeatedly until stable.
    # GGUF tensor names are capped at 64 bytes by the C reader, so the upstream
    # PyTorch names (some up to 77 chars when prefixed) need shortening. Order
    # matters: longer/more-specific patterns first.
    ("parametrizations.weight.original0", "wp0"),
    ("parametrizations.weight.original1", "wp1"),
    ("transformer.layers",                "tr.l"),

    # MOSS-Audio-Tokenizer v2 renamed three modules relative to v1. Mapping them
    # back onto v1's GGUF names keeps src/codec.cpp's resolvers name-agnostic —
    # only the stage tables differ between codec generations.
    #
    # The plural forms MUST come first: "self_attn.in_proj" is a prefix of
    # "self_attn.in_projs", so applying the singular rule first would mangle v1
    # into "attn.inp.0s.0".
    ("self_attn.in_projs",                "attn.inp"),    # v1: ModuleList
    ("self_attn.out_projs",               "attn.outp"),   # v1: ModuleList
    ("self_attn.in_proj",                 "attn.inp.0"),  # v2: single Linear
    ("self_attn.out_proj",                "attn.outp.0"), # v2: single Linear
    ("ffn.0.weight",                      "linear1.weight"),  # v2 Sequential
    ("ffn.2.weight",                      "linear2.weight"),  # v2 Sequential

    ("encoder",                           "enc"),
    ("decoder",                           "dec"),
    ("self_attn",                         "attn"),
    ("in_projs",                          "inp"),
    ("out_projs",                         "outp"),
    ("quantizers",                        "q"),
    ("input_proj",                        "iproj"),
    ("output_proj",                       "oproj"),
    ("in_proj",                           "iproj"),
    ("out_proj",                          "oproj"),
)


def _shorten_codec_name(name: str) -> str:
    """Map a PyTorch codec tensor name → a ≤64-char GGUF-safe name.

    The mapping is deterministic and reversible (we ship the rules in
    ``moss.codec.name_table`` KV so the C++ reader can reconstruct the
    original PyTorch name if it ever needs to log them).
    """
    s = name
    for pat, rep in _CODEC_RENAMES:
        s = s.replace(pat, rep)
    return s


def collect_codec_tensors(codec_dir: Path):
    """Yield (gguf_tensor_name, np.ndarray) for every tensor in the codec
    safetensors. Integer tensors are passed through in their native dtype.

    Names are renamed according to ``_CODEC_RENAMES`` and prefixed with
    ``moss.codec.`` so the result is always under the 64-byte GGUF limit.
    """
    weight_map = load_safetensors_index(codec_dir)
    by_shard: dict[str, list[str]] = defaultdict(list)
    for tname, shard in weight_map.items():
        by_shard[shard].append(tname)
    seen: dict[str, str] = {}  # short → original (for collision detection)
    for shard in sorted(by_shard):
        with STShard(codec_dir / shard) as sf:
            for tname in sorted(by_shard[shard]):
                src_dtype = sf.dtype_of(tname)
                if src_dtype in ("F16", "BF16", "F32", "F64"):
                    arr = sf.get_f16(tname)
                else:
                    arr = sf.get_native(tname)
                short = "moss.codec." + _shorten_codec_name(tname)
                if len(short) > 63:
                    raise RuntimeError(
                        f"codec tensor name still over limit ({len(short)} chars): {short}\n"
                        f"  original: {tname}\n"
                        f"  add another rule to _CODEC_RENAMES")
                if short in seen:
                    raise RuntimeError(
                        f"codec tensor name collision after rename: {short}\n"
                        f"  {seen[short]}\n  {tname}")
                seen[short] = tname
                yield short, arr


def _fold_weight_norm(g: "np.ndarray", v: "np.ndarray") -> "np.ndarray":
    """PyTorch weight_norm reconstruction: w = g * v / ||v||.

    The norm runs over every dim except dim 0, and *dim 0 means different
    things for the two conv flavours*:

        Conv1d          weight is [out, in, k]  -> g is indexed by OUT channel
        ConvTranspose1d weight is [in, out, k]  -> g is indexed by IN  channel

    Both are handled identically here precisely because both put the indexed
    axis first; the caller does not need to care which it has.
    """
    v = v.astype(np.float32, copy=False)
    norm = np.sqrt((v.astype(np.float64) ** 2).sum(axis=tuple(range(1, v.ndim)),
                                                   keepdims=True))
    return (g.astype(np.float64) * v.astype(np.float64) / norm).astype(np.float32)


def collect_soundeffect_extras(moss_dir: Path, cfg: dict):
    """Yield (gguf_tensor_name, np.ndarray) for the DiT and the DAC VAE decoder.

    Both conv flavours are shipped in PyTorch's own row-major layout, which is
    already what ggml wants: ggml_conv_1d reads its kernel as ne=(K, IC, OC) and
    PyTorch's [OC, IC, K] gives exactly that, while ggml_conv_transpose_1d wants
    ne=(K, OC, IC) and PyTorch's [IC, OC, K] gives that. No transpose either way.
    """
    # ── DiT ────────────────────────────────────────────────────────────────
    dit_dir = moss_dir / "transformer"
    with STShard(dit_dir / "diffusion_pytorch_model.safetensors") as sf:
        for t in sorted(sf.keys()):
            arr = sf.get_f16(t)
            n = t
            n = n.replace("condition_embedder.time_embedder.linear_", "t_emb.")
            n = n.replace("condition_embedder.time_proj",             "t_proj")
            n = n.replace("condition_embedder.text_embedder.linear_", "txt_emb.")
            n = n.replace("patch_embedding",                          "patch")
            n = n.replace("proj_out",                                 "head")
            n = n.replace("blocks.",                                  "blk.")
            n = n.replace(".attn1.",  ".sa.").replace(".attn2.", ".ca.")
            n = n.replace("to_out.0", "o").replace("to_q", "q").replace("to_k", "k").replace("to_v", "v")
            n = n.replace(".ffn.net.0.proj", ".ffn.1").replace(".ffn.net.2", ".ffn.2")
            # The affine LayerNorm inside a block is norm3 in the reference but
            # is stored as norm2 on disk; norm1/norm2 are affine-free and absent.
            n = n.replace(".norm2.", ".norm3.")
            n = n.replace("scale_shift_table", "mod")
            yield "moss.dit." + n, arr

    # ── DAC VAE decoder ────────────────────────────────────────────────────
    # `continuous: True` means this is a KL-VAE, not a codec: there is no
    # quantizer and no codebooks. Only the decoder half is needed for
    # text-to-audio, which is 75.5 M of the checkpoint's 371 M parameters.
    try:
        import torch
    except ImportError:
        raise SystemExit("moss_soundeffect: the DAC VAE is a torch pickle; "
                         "pip install torch to convert it")
    sd = torch.load(moss_dir / "vae" / "vae_128d_48k.pth",
                    map_location="cpu", weights_only=True)["state_dict"]
    sd = {k: v.numpy() for k, v in sd.items()
          if k.startswith("decoder.") or k.startswith("post_quant_conv.")}

    def emit_conv(prefix, key):
        """A weight-normed conv: fold g/v, pass the bias through."""
        g, v = sd[key + ".weight_g"], sd[key + ".weight_v"]
        yield prefix + ".weight", _fold_weight_norm(g, v).astype(np.float16)
        yield prefix + ".bias",   sd[key + ".bias"].astype(np.float16)

    def emit_snake(prefix, key):
        """Snake: y = x + sin(a*x)^2 / (a + 1e-9).

        The reciprocal is precomputed in fp32 here so the graph is a plain
        multiply — and so it can match llama.cpp's fused Snake kernel, which
        pattern-matches MUL->SIN->SQR->MUL->ADD with a separate inv tensor.
        """
        a = sd[key + ".alpha"].astype(np.float32).reshape(-1)
        yield prefix + ".alpha", a.astype(np.float32)
        yield prefix + ".inv",   (1.0 / (a + 1e-9)).astype(np.float32)

    yield "moss.vae.post_quant.weight", sd["post_quant_conv.weight"].astype(np.float16)
    yield "moss.vae.post_quant.bias",   sd["post_quant_conv.bias"].astype(np.float16)

    for name, arr in emit_conv("moss.vae.dec.in", "decoder.model.0"):
        yield name, arr

    n_blocks = max(int(k.split(".")[2]) for k in sd
                   if k.startswith("decoder.model.") and k.split(".")[2].isdigit()) - 1
    for b in range(1, n_blocks):        # decoder.model.1 .. n_blocks-1
        base = f"decoder.model.{b}"
        pre  = f"moss.vae.dec.{b}"
        for name, arr in emit_snake(pre + ".snake", base + ".block.0"):
            yield name, arr
        for name, arr in emit_conv(pre + ".up", base + ".block.1"):
            yield name, arr
        for r in (2, 3, 4):             # three ResidualUnits, dilations 1/3/9
            rb = f"{base}.block.{r}.block"
            rp = f"{pre}.res.{r - 2}"
            for name, arr in emit_snake(rp + ".snake1", rb + ".0"):
                yield name, arr
            for name, arr in emit_conv(rp + ".conv1", rb + ".1"):
                yield name, arr
            for name, arr in emit_snake(rp + ".snake2", rb + ".2"):
                yield name, arr
            for name, arr in emit_conv(rp + ".conv2", rb + ".3"):
                yield name, arr

    for name, arr in emit_snake("moss.vae.dec.out.snake", f"decoder.model.{n_blocks}"):
        yield name, arr
    for name, arr in emit_conv("moss.vae.dec.out", f"decoder.model.{n_blocks + 1}"):
        yield name, arr


def write_soundeffect_kv(writer, moss_config: dict, moss_dir: Path):
    with (moss_dir / "transformer" / "config.json").open() as f:
        dit = json.load(f)
    with (moss_dir / "vae" / "config.json").open() as f:
        vae = json.load(f)
    with (moss_dir / "scheduler" / "scheduler_config.json").open() as f:
        sch = json.load(f)

    writer.add_uint32("moss.dit.dim",        int(dit["dim"]))
    writer.add_uint32("moss.dit.n_layers",   int(dit["num_layers"]))
    writer.add_uint32("moss.dit.n_heads",    int(dit["num_heads"]))
    writer.add_uint32("moss.dit.ffn_dim",    int(dit["ffn_dim"]))
    writer.add_uint32("moss.dit.in_dim",     int(dit["in_dim"]))
    writer.add_uint32("moss.dit.out_dim",    int(dit["out_dim"]))
    writer.add_uint32("moss.dit.text_dim",   int(dit["text_dim"]))
    writer.add_uint32("moss.dit.freq_dim",   int(dit["freq_dim"]))
    writer.add_float32("moss.dit.eps",       float(dit.get("eps", 1e-6)))

    writer.add_float32("moss.sched.shift",   float(sch.get("shift", 5.0)))
    writer.add_uint32("moss.sched.train_steps",
                      int(sch.get("num_train_timesteps", 1000)))

    writer.add_uint32("moss.vae.latent_dim", int(vae.get("latent_dim", 128)))
    # decoder_rates from the checkpoint metadata; product is the total upsample.
    rates = [8, 5, 4, 3, 2]
    writer.add_array("moss.vae.decoder_rates", rates)
    hop = 1
    for r in rates:
        hop *= r
    writer.add_uint32("moss.vae.hop", hop)          # 960 -> 50 latent frames/s

    # Text conditioning is a fixed 512-slot window, right-padded and hard-zeroed
    # past the real length.
    writer.add_uint32("moss.text.max_len", 512)
    # Duration is purely textual: the DiT always produces max_seconds worth of
    # latents and the waveform is cropped afterwards.
    writer.add_uint32("moss.max_seconds",
                      int(moss_config.get("max_inference_seconds", 30)))


class Family:
    """Everything that differs between MOSS model families."""

    def __init__(self, arch, backbone_config_keys, remap_backbone, collect_extras,
                 codec_version, downsample_rate, n_channels, sampling_rate,
                 audio_start, audio_end, n_vq, backbone_subdir=None,
                 write_extra_kv=None):
        self.arch                 = arch
        self.backbone_config_keys = backbone_config_keys
        self.remap_backbone       = remap_backbone
        self.collect_extras       = collect_extras
        self.codec_version        = codec_version
        self.downsample_rate      = downsample_rate
        self.n_channels           = n_channels
        self.sampling_rate        = sampling_rate
        self.audio_start          = audio_start
        self.audio_end            = audio_end
        self.n_vq                 = n_vq
        # When set, the backbone is already a standalone checkpoint in this
        # subdirectory and just needs the tokenizer alongside it — no tensor
        # extraction or renaming.
        self.backbone_subdir      = backbone_subdir
        self.write_extra_kv       = write_extra_kv


FAMILIES = {
    "moss_tts_delay": Family(
        arch="moss_tts_delay",
        backbone_config_keys=("language_config",),
        remap_backbone=remap_backbone_delay,
        collect_extras=collect_audio_extras_delay,
        codec_version=1, downsample_rate=1920, n_channels=1, sampling_rate=24000,
        audio_start=151652, audio_end=151653, n_vq=32,
    ),
    "moss_soundeffect": Family(
        arch="moss_soundeffect",
        # The text encoder is already a standalone Qwen3ForCausalLM checkpoint
        # in text_encoder/; it only needs the tokenizer copied alongside.
        backbone_config_keys=(),
        backbone_subdir="text_encoder",
        remap_backbone=None,
        collect_extras=collect_soundeffect_extras,
        write_extra_kv=write_soundeffect_kv,
        # Not an RVQ model at all: DiT + flow matching + a continuous KL-VAE.
        codec_version=0, downsample_rate=960, n_channels=1, sampling_rate=48000,
        audio_start=0, audio_end=0, n_vq=0,
    ),
    "moss_tts_local": Family(
        arch="moss_tts_local",
        # v1.5 ships qwen3_config and language_config with identical contents;
        # prefer the explicitly-named one.
        backbone_config_keys=("qwen3_config", "language_config"),
        remap_backbone=remap_backbone_local,
        collect_extras=collect_audio_extras_local,
        codec_version=2, downsample_rate=3840, n_channels=2, sampling_rate=48000,
        audio_start=151669, audio_end=151670, n_vq=12,
    ),
}


def detect_family(moss_config: dict, override: str | None = None) -> Family:
    if override:
        if override not in FAMILIES:
            raise SystemExit(f"--arch {override!r} is not one of {sorted(FAMILIES)}")
        return FAMILIES[override]
    mt = str(moss_config.get("model_type", "")).strip()
    if mt in FAMILIES:
        return FAMILIES[mt]
    if moss_config.get("_class_name") == "MossSoundEffectPipeline":
        return FAMILIES["moss_soundeffect"]
    for a in moss_config.get("architectures", []):
        if a == "MossTTSLocalModel":
            return FAMILIES["moss_tts_local"]
        if a == "MossTTSDelayModel":
            return FAMILIES["moss_tts_delay"]
    raise SystemExit(
        f"cannot determine model family (model_type={mt!r}, "
        f"architectures={moss_config.get('architectures')}); pass --arch explicitly")


def write_moss_sidecar(out_gguf: Path,
                       moss_config: dict, moss_dir: Path,
                       codec_dir: Path | None,
                       fam: Family,
                       llama_cpp_dir: Path | None = None) -> None:
    """Write the MOSS extras (audio embeddings, family-specific heads, optional
    codec, plus the moss.* KV namespace) into a sidecar GGUF.

    The C++ loader opens this file alongside the backbone GGUF.
    """
    if llama_cpp_dir is not None:
        gguf_py = str((llama_cpp_dir / "gguf-py").resolve())
        if gguf_py not in sys.path:
            sys.path.insert(0, gguf_py)
    import gguf

    writer = gguf.GGUFWriter(str(out_gguf), fam.arch)

    # ── 1. emit MOSS KV ─────────────────────────────────────────────────────
    n_vq             = int(moss_config.get("n_vq", fam.n_vq))
    audio_vocab_size = int(moss_config.get("audio_vocab_size", 1024))
    audio_pad_code   = int(moss_config.get("audio_pad_code", audio_vocab_size))
    sampling_rate    = int(moss_config.get("sampling_rate", fam.sampling_rate))
    downsample_rate  = fam.downsample_rate
    n_channels       = fam.n_channels

    # The codec is the authority on hop size / channel count when we have it.
    if codec_dir is not None and (codec_dir / "config.json").exists():
        with (codec_dir / "config.json").open() as f:
            codec_cfg = json.load(f)
        downsample_rate = int(codec_cfg.get("downsample_rate", downsample_rate))
        n_channels      = int(codec_cfg.get("number_channels", n_channels))
        sampling_rate   = int(codec_cfg.get("sampling_rate",
                              codec_cfg.get("sample_rate", sampling_rate)))

    # Read back by the C++ loader; `general.architecture` (which GGUFWriter also
    # writes) is the fallback for sidecars produced before this key existed.
    writer.add_string("moss.architecture", fam.arch)

    writer.add_uint32("moss.n_vq", n_vq)
    writer.add_uint32("moss.audio_vocab_size", audio_vocab_size)
    writer.add_uint32("moss.audio_pad_code", audio_pad_code)
    writer.add_uint32("moss.sampling_rate", sampling_rate)
    writer.add_uint32("moss.downsample_rate", downsample_rate)
    writer.add_float32("moss.frame_rate", float(sampling_rate) / float(downsample_rate))
    writer.add_uint32("moss.codec.version", fam.codec_version)
    writer.add_uint32("moss.codec.number_channels", n_channels)

    writer.add_uint32("moss.token.audio_start",
                      int(moss_config.get("audio_start_token_id", fam.audio_start)))
    writer.add_uint32("moss.token.audio_end",
                      int(moss_config.get("audio_end_token_id", fam.audio_end)))
    writer.add_uint32("moss.token.audio_user_slot",
                      int(moss_config.get("audio_user_slot_token_id", 151654)))
    writer.add_uint32("moss.token.audio_gen_slot",
                      int(moss_config.get("audio_assistant_gen_slot_token_id", 151656)))
    writer.add_uint32("moss.token.im_start",
                      int(moss_config.get("im_start_token_id", 151644)))
    writer.add_uint32("moss.token.im_end",
                      int(moss_config.get("im_end_token_id", 151645)))
    writer.add_uint32("moss.token.pad",
                      int(moss_config.get("pad_token_id", 151643)))
    if fam.arch == "moss_tts_delay":
        writer.add_uint32("moss.token.audio_delay_slot",
                          int(moss_config.get("audio_assistant_delay_slot_token_id", 151662)))

    if fam.arch == "moss_tts_local":
        gpt2 = moss_config.get("gpt2_config", {})
        n_layer = int(moss_config.get("local_transformer_layers", gpt2.get("n_layer", 1)))
        writer.add_uint32("moss.local.n_layer",  n_layer)
        writer.add_uint32("moss.local.n_embd",   int(gpt2.get("n_embd", 2560)))
        writer.add_uint32("moss.local.n_head",   int(gpt2.get("n_head", 32)))
        writer.add_uint32("moss.local.n_inner",  int(gpt2.get("n_inner", 9728)))
        writer.add_float32("moss.local.rope_base",
                           float(gpt2.get("rope_base", 1000000.0)))
        writer.add_float32("moss.local.ln_eps",
                           float(gpt2.get("layer_norm_epsilon", 1e-6)))
        mode = str(moss_config.get("local_text_head_mode", "binary"))
        if mode != "binary":
            raise RuntimeError(
                f"moss_tts_local: only local_text_head_mode='binary' is supported, "
                f"got {mode!r}")

    if fam.write_extra_kv is not None:
        fam.write_extra_kv(writer, moss_config, moss_dir)

    writer.add_bool("moss.codec.present", codec_dir is not None)

    # ── 2. add MOSS audio tensors ──────────────────────────────────────────
    #
    # f16 is the default, but a collector that deliberately produced f32 keeps
    # it. That matters: MOSS-SoundEffect's Snake activation ships a precomputed
    # 1/(alpha + 1e-9), and three of those reciprocals exceed f16's 65504 ceiling
    # (alpha gets as small as 8.3e-6, giving 1.2e5). Rounding them to inf would
    # feed inf into the decoder and corrupt the waveform silently.
    audio_count = 0
    for name, arr in fam.collect_extras(moss_dir, moss_config):
        writer.add_tensor(name, arr if arr.dtype == np.float32
                                else arr.astype(np.float16))
        audio_count += 1
    log.info("added %d MOSS audio/head tensors", audio_count)

    # ── 3. add codec tensors (optional) ────────────────────────────────────
    if codec_dir is not None:
        codec_count = 0
        for name, arr in collect_codec_tensors(codec_dir):
            writer.add_tensor(name, arr.astype(np.float16))
            codec_count += 1
        log.info("added %d codec tensors", codec_count)

    # ── 4. flush ───────────────────────────────────────────────────────────
    log.info("writing sidecar %s", out_gguf)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def resolve_model_dir(spec: str, cache_dir: str | None) -> Path:
    p = Path(spec)
    # diffusers pipelines (MOSS-SoundEffect) describe themselves in
    # model_index.json and have no top-level config.json.
    if p.is_dir() and ((p / "config.json").exists()
                       or (p / "model_index.json").exists()):
        return p.resolve()
    log.info("downloading %s from HuggingFace", spec)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(spec, cache_dir=cache_dir,
                                   ignore_patterns=["*.md", "__pycache__"]))


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    ap = argparse.ArgumentParser(
        description="Convert MOSS-TTS-Delay (+ codec) to a single GGUF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--moss-tts", required=True,
                    help="HF id or local dir of MOSS-TTS-Delay (e.g. OpenMOSS-Team/MOSS-TTS)")
    ap.add_argument("--codec", default=None,
                    help="HF id or local dir of MOSS-Audio-Tokenizer; omit to skip codec for now")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    ap.add_argument("--llama-cpp-dir",
                    default=str(Path(__file__).resolve().parent.parent / "third_party" / "llama.cpp"),
                    help="Path to a llama.cpp source tree — we shell out to its "
                         "convert_hf_to_gguf.py and import its gguf-py (no build needed). "
                         "Default: the bundled third_party/llama.cpp submodule.")
    ap.add_argument("--cache-dir", default=None,
                    help="HF cache dir (defaults to ~/.cache/huggingface)")
    ap.add_argument("--scratch-dir", default=None,
                    help="Temp dir for the extracted Qwen3 backbone (default: a tempdir)")
    ap.add_argument("--backbone-dtype", default="f16", choices=["f16", "f32", "bf16"],
                    help="Output dtype for the backbone (default: f16)")
    ap.add_argument("--keep-scratch", action="store_true",
                    help="Don't delete the scratch dir after conversion")
    ap.add_argument("--skip-extract", action="store_true",
                    help="If scratch/qwen3_backbone.gguf already exists, skip stages 1 and 2")
    ap.add_argument("--arch", default=None, choices=sorted(FAMILIES),
                    help="Override model-family detection (normally taken from "
                         "config.json's model_type)")
    ap.add_argument("--sidecar-only", action="store_true",
                    help="Only (re)write the .extras.gguf sidecar, leaving an "
                         "existing backbone GGUF at --output alone. Useful for "
                         "iterating on the codec/extras without redoing the "
                         "multi-minute backbone conversion.")
    args = ap.parse_args()

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    moss_dir = resolve_model_dir(args.moss_tts, args.cache_dir)
    codec_dir = resolve_model_dir(args.codec, args.cache_dir) if args.codec else None

    cleanup_scratch = False
    if args.scratch_dir:
        scratch = Path(args.scratch_dir).resolve()
        scratch.mkdir(parents=True, exist_ok=True)
    else:
        scratch = Path(tempfile.mkdtemp(prefix="moss-convert-"))
        cleanup_scratch = not args.keep_scratch

    cfg_path = moss_dir / "config.json"
    if not cfg_path.exists():
        # diffusers pipelines describe themselves in model_index.json instead
        cfg_path = moss_dir / "model_index.json"
    with cfg_path.open() as f:
        moss_config = json.load(f)
    fam = detect_family(moss_config, args.arch)
    log.info("model family: %s", fam.arch)

    try:
        qwen3_dir = scratch / "qwen3_backbone"
        backbone_gguf = scratch / "qwen3_backbone.gguf"

        if args.sidecar_only:
            if not out_path.exists():
                raise SystemExit(f"--sidecar-only: {out_path} does not exist")
            log.info("=== stages 1-3a skipped (--sidecar-only) ===")
        else:
            if args.skip_extract and backbone_gguf.exists():
                log.info("=== stages 1+2 skipped (using cached %s) ===", backbone_gguf)
            else:
                log.info("=== stage 1: extract Qwen3 backbone ===")
                if fam.backbone_subdir:
                    assemble_backbone_dir(moss_dir, qwen3_dir, fam)
                else:
                    extract_qwen3_backbone(moss_dir, qwen3_dir, fam, moss_config)

                log.info("=== stage 2: convert backbone to GGUF (via llama.cpp) ===")
                run_llama_cpp_converter(qwen3_dir, backbone_gguf,
                                        Path(args.llama_cpp_dir), args.backbone_dtype)
                # The extracted safetensors are dead weight once the GGUF exists,
                # and they are the same size as it (8+ GB for a 4B backbone).
                if not args.keep_scratch:
                    shutil.rmtree(qwen3_dir, ignore_errors=True)

            # Stage 3a: place the backbone GGUF at the user's output path
            # (libllama validates that every tensor in a GGUF is "claimed" by the
            # model loader, so we can't put unknown moss.* tensors in there — they
            # live in a sidecar file alongside the backbone).
            log.info("=== stage 3a: place backbone GGUF at %s ===", out_path)
            if out_path.resolve() != backbone_gguf.resolve():
                if cleanup_scratch:
                    # Scratch is a tempdir we own and will delete anyway; moving
                    # avoids a second full-size copy sitting on disk.
                    shutil.move(str(backbone_gguf), str(out_path))
                else:
                    shutil.copy2(backbone_gguf, out_path)

        sidecar_path = out_path.with_suffix(".extras.gguf")
        log.info("=== stage 3b: write MOSS sidecar → %s ===", sidecar_path)
        write_moss_sidecar(sidecar_path, moss_config, moss_dir, codec_dir,
                           fam, Path(args.llama_cpp_dir))
    finally:
        if cleanup_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
        else:
            log.info("scratch retained at %s", scratch)

    log.info("done — wrote %s (%.2f GB)", out_path,
             out_path.stat().st_size / 1024 ** 3)


if __name__ == "__main__":
    main()
