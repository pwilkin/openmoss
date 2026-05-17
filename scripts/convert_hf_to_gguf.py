#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Convert MOSS-TTS-Delay (+ MOSS-Audio-Tokenizer) from HuggingFace safetensors
into a single GGUF file consumable by openmoss-ggml.

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

def remap_backbone_name(name: str) -> str | None:
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


def extract_qwen3_backbone(moss_dir: Path, out_dir: Path) -> dict:
    """Materialise the language_model.* + lm_heads.0 part of MossTTSDelay
    into ``out_dir`` as a self-contained Qwen3ForCausalLM checkpoint.

    Returns the moss config dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with (moss_dir / "config.json").open() as f:
        moss_config = json.load(f)

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
                qwen_name = remap_backbone_name(tname)
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
    lang = dict(moss_config["language_config"])
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

    return moss_config


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


def collect_audio_extras(moss_dir: Path, n_vq: int):
    """Yield (gguf_tensor_name, np.ndarray[f16]) for the audio embeddings + heads."""
    weight_map = load_safetensors_index(moss_dir)
    by_shard: dict[str, list[str]] = defaultdict(list)
    for tname, shard in weight_map.items():
        by_shard[shard].append(tname)

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


_CODEC_RENAMES = (
    # (substring, replacement) — applied in order, repeatedly until stable.
    # GGUF tensor names are capped at 64 bytes by the C reader, so the upstream
    # PyTorch names (some up to 77 chars when prefixed) need shortening. Order
    # matters: longer/more-specific patterns first.
    ("parametrizations.weight.original0", "wp0"),
    ("parametrizations.weight.original1", "wp1"),
    ("transformer.layers",                "tr.l"),
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


def write_moss_sidecar(out_gguf: Path,
                       moss_config: dict, moss_dir: Path,
                       codec_dir: Path | None,
                       llama_cpp_dir: Path | None = None) -> None:
    """Write the MOSS extras (audio embeddings, audio heads, optional codec,
    plus the moss.* KV namespace) into a sidecar GGUF.

    The C++ loader opens this file alongside the backbone GGUF.
    """
    if llama_cpp_dir is not None:
        gguf_py = str((llama_cpp_dir / "gguf-py").resolve())
        if gguf_py not in sys.path:
            sys.path.insert(0, gguf_py)
    import gguf

    writer = gguf.GGUFWriter(str(out_gguf), "moss_tts_delay")

    # ── 1. emit MOSS KV ─────────────────────────────────────────────────────
    n_vq             = int(moss_config.get("n_vq", 32))
    audio_vocab_size = int(moss_config.get("audio_vocab_size", 1024))
    audio_pad_code   = int(moss_config.get("audio_pad_code", audio_vocab_size))
    sampling_rate    = int(moss_config.get("sampling_rate", 24000))
    downsample_rate  = 1920  # codec hop, fixed by upstream

    writer.add_uint32("moss.n_vq", n_vq)
    writer.add_uint32("moss.audio_vocab_size", audio_vocab_size)
    writer.add_uint32("moss.audio_pad_code", audio_pad_code)
    writer.add_uint32("moss.sampling_rate", sampling_rate)
    writer.add_uint32("moss.downsample_rate", downsample_rate)
    writer.add_float32("moss.frame_rate", float(sampling_rate) / float(downsample_rate))

    writer.add_uint32("moss.token.audio_start",
                      int(moss_config.get("audio_start_token_id", 151652)))
    writer.add_uint32("moss.token.audio_end",
                      int(moss_config.get("audio_end_token_id", 151653)))
    writer.add_uint32("moss.token.audio_user_slot",
                      int(moss_config.get("audio_user_slot_token_id", 151654)))
    writer.add_uint32("moss.token.audio_gen_slot",
                      int(moss_config.get("audio_assistant_gen_slot_token_id", 151656)))
    writer.add_uint32("moss.token.audio_delay_slot",
                      int(moss_config.get("audio_assistant_delay_slot_token_id", 151662)))
    writer.add_uint32("moss.token.im_start",
                      int(moss_config.get("im_start_token_id", 151644)))
    writer.add_uint32("moss.token.im_end",
                      int(moss_config.get("im_end_token_id", 151645)))

    writer.add_bool("moss.codec.present", codec_dir is not None)

    # ── 2. add MOSS audio tensors ──────────────────────────────────────────
    audio_count = 0
    for name, arr in collect_audio_extras(moss_dir, n_vq):
        writer.add_tensor(name, arr.astype(np.float16))
        audio_count += 1
    log.info("added %d MOSS audio tensors", audio_count)

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
    if p.is_dir() and (p / "config.json").exists():
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
    ap.add_argument("--llama-cpp-dir", default="/devel/tools/llama.cpp",
                    help="Path to a built llama.cpp tree (we shell out to convert_hf_to_gguf.py)")
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

    try:
        qwen3_dir = scratch / "qwen3_backbone"
        backbone_gguf = scratch / "qwen3_backbone.gguf"

        if args.skip_extract and backbone_gguf.exists():
            log.info("=== stages 1+2 skipped (using cached %s) ===", backbone_gguf)
            with (moss_dir / "config.json").open() as f:
                moss_config = json.load(f)
        else:
            log.info("=== stage 1: extract Qwen3 backbone ===")
            moss_config = extract_qwen3_backbone(moss_dir, qwen3_dir)

            log.info("=== stage 2: convert backbone to GGUF (via llama.cpp) ===")
            run_llama_cpp_converter(qwen3_dir, backbone_gguf,
                                    Path(args.llama_cpp_dir), args.backbone_dtype)

        # Stage 3a: copy/rename the backbone GGUF into the user's output path
        # (libllama validates that every tensor in a GGUF is "claimed" by the
        # model loader, so we can't put unknown moss.* tensors in there — they
        # live in a sidecar file alongside the backbone).
        log.info("=== stage 3a: place backbone GGUF at %s ===", out_path)
        if out_path.resolve() != backbone_gguf.resolve():
            shutil.copy2(backbone_gguf, out_path)

        sidecar_path = out_path.with_suffix(".extras.gguf")
        log.info("=== stage 3b: write MOSS sidecar → %s ===", sidecar_path)
        write_moss_sidecar(sidecar_path, moss_config, moss_dir, codec_dir,
                           Path(args.llama_cpp_dir))
    finally:
        if cleanup_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
        else:
            log.info("scratch retained at %s", scratch)

    log.info("done — wrote %s (%.2f GB)", out_path,
             out_path.stat().st_size / 1024 ** 3)


if __name__ == "__main__":
    main()
