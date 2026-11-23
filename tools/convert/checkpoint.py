"""What the checkpoint converters share (checkpoint_inkling2.py, checkpoint_qwen35.py, checkpoint_deepseek41.py,
hy4flash_bf16.py): the safetensors files of a checkpoint directory, NVIDIA Model Optimizer's NVFP4 rows moved into
Basalt's blocks, the routed-expert blob job, the GDN value-head reorder, and the model metadata and tokenizer as GGUF
keys. Needs numpy.
"""
from __future__ import annotations

import json
import pathlib
import re

import numpy as np

import model_sections as S

UNUSED, NORMAL, CONTROL, USER_DEFINED = 5, 1, 3, 4
# The pre-tokenizer split of the qwen35 vocabulary as Hugging Face tokenizers writes it.
QWEN35_HF_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
                   r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
# The tensors a file beside the model-NNNNN-of-NNNNN shards may hold: the drafter, vision, the experts, the n-gram table.
SKIP = re.compile(r"^(mtp\.|model\.visual\.|model\.language_model\.layers\.\d+\.(mlp\.experts\.|ple\.ple_embedding\.))")


class Checkpoint:
    """The safetensors files of a checkpoint directory: name -> (path, absolute offset, dtype, shape, bytes)."""

    def __init__(self, d, side: dict[str, str] | None = None):
        """`side`: files beside the shards whose tensor names carry no prefix of their own, each mapped to the prefix
        its names get here (Inkling's `mtp.safetensors` -> `mtp.`)."""
        self.dir = pathlib.Path(d)
        cfg = self.dir / "config.json"
        if not cfg.is_file():
            raise S.ConvertError(f"{cfg} not found")
        self.config = json.loads(cfg.read_text(encoding="utf-8"))
        self.text = self.config.get("text_config", self.config)
        self.files = sorted(self.dir.glob("*.safetensors")) + sorted(self.dir.glob("tensors/*.safetensors"))
        self.shards = [p for p in self.files if re.fullmatch(r"model-\d+-of-\d+\.safetensors", p.name)]
        if not self.shards:
            raise S.ConvertError(f"{self.dir}: no model-NNNNN-of-NNNNN.safetensors shards")
        side = side or {}
        self.t, self.order = {}, []
        for p in self.files:
            head, base = S.read_safetensors_header(p)
            items = sorted(((side.get(p.name, "") + k, v) for k, v in head.items() if k != "__metadata__"),
                           key=lambda kv: kv[1]["data_offsets"][0])
            for k, v in items:
                if k in self.t:
                    raise S.ConvertError(f"{p.name}: duplicate tensor {k}")
                a, b = v["data_offsets"]
                self.t[k] = (p, base + a, v["dtype"], list(v["shape"]), b - a)
                if p in self.shards:
                    self.order.append(k)
                elif not SKIP.match(k):
                    raise S.ConvertError(f"{p.name} is not a model-NNNNN-of-NNNNN shard and holds {k}, which would be "
                                         "dropped; only the n-gram table, drafter and vision tensors may live there")

    def raw(self, name) -> np.ndarray:
        p, off, _, _, n = self.t[name]
        return np.fromfile(p, dtype=np.uint8, count=n, offset=off)

    def values(self, name) -> np.ndarray:
        """A BF16, F32 or I64 tensor as float32 (int64 for I64), shaped."""
        p, off, dt, shape, n = self.t[name]
        raw = self.raw(name)
        if dt == "BF16":
            return S._bf16_to_f32(raw.view("<u2")).reshape(shape)
        if dt == "F32":
            return raw.view("<f4").reshape(shape)
        if dt == "I64":
            return raw.view("<i8").reshape(shape)
        raise S.ConvertError(f"{name} is {dt}")

    def need(self, name, dtype, shape=None):
        if name not in self.t:
            raise S.ConvertError(f"the checkpoint has no {name}")
        _, _, dt, sh, _ = self.t[name]
        if dt != dtype or (shape is not None and sh != list(shape)):
            raise S.ConvertError(f"{name} is {dt} {sh}, expected {dtype} {list(shape) if shape else ''}")


# ------------------------------------------------------------------ routed experts
def nvfp4_rows(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Model Optimizer's NVFP4 rows -> Basalt's blocks. `codes` [rows, cols/2]: value 2i in the low nibble of byte i,
    2i+1 in the high one; `scales` [rows, cols/16] UE4M3. A Basalt block of 64 values is 4 scale bytes then 32 code
    bytes where sub-block s keeps value j in the low nibble of byte 8 s + j and value j + 8 in its high nibble."""
    rows, cols = codes.shape[0], codes.shape[1] * 2
    v = np.empty((rows, cols), np.uint8)
    v[:, 0::2] = codes & 15
    v[:, 1::2] = codes >> 4
    v = v.reshape(rows, cols // 64, 4, 16)
    out = np.empty((rows, cols // 64, 36), np.uint8)
    out[..., :4] = scales.reshape(rows, cols // 64, 4)
    out[..., 4:] = (v[..., :8] | (v[..., 8:] << 4)).reshape(rows, cols // 64, 32)
    return out.reshape(-1)


def csf_decode(fixed: np.ndarray, exceptions: np.ndarray) -> np.ndarray:
    """A CSF stream pair (Model Optimizer's compressed block scales) -> the UE4M3 scale bytes [rows, cols] it holds.

    fixed       uint8 [rows / 16, 16 + 8 cols]: per slab of 16 rows, the rows' base bytes, then each row's cols 4-bit
                codes, two to a byte, low nibble first; a scale byte is base + code
    exceptions  uint32: low 24 bits the flat position row * cols + col, high 8 bits the scale byte there"""
    if fixed.ndim != 2 or fixed.shape[1] < 32 or fixed.shape[1] % 16:
        raise S.ConvertError(f"a CSF fixed stream of shape {list(fixed.shape)}")
    rows, cols = fixed.shape[0] * 16, (fixed.shape[1] // 16 - 1) * 2
    base = fixed[:, :16].reshape(rows, 1).astype(np.int16)
    if (base > 240).any():
        raise S.ConvertError("a CSF base byte above 240")
    packed = fixed[:, 16:].reshape(rows, cols // 2)
    out = (base + np.stack((packed & 15, packed >> 4), axis=-1).reshape(rows, cols)).reshape(-1)
    pos = exceptions & 0xFFFFFF
    if pos.size and int(pos.max()) >= rows * cols:
        raise S.ConvertError("a CSF exception past the end of its matrix")
    out[pos] = exceptions >> 24
    return out.astype(np.uint8).reshape(rows, cols)


def _scales(ent) -> np.ndarray:
    """An expert matrix's UE4M3 scale bytes: from the plain tensor, ent = (path, offset, shape), or from its CSF stream
    pair, ent = ((path, offset, shape), (path, offset, count))."""
    if len(ent) == 3:
        p, off, shape = ent
        return np.fromfile(p, dtype=np.uint8, count=shape[0] * shape[1], offset=off).reshape(shape)
    (fp, fo, fshape), (ep, eo, n) = ent
    fixed = np.fromfile(fp, dtype=np.uint8, count=fshape[0] * fshape[1], offset=fo).reshape(fshape)
    return csf_decode(fixed, np.fromfile(ep, dtype="<u4", count=n, offset=eo))


def _expert_job(args):
    """Blobs of experts [s, s + n) of one layer: [gate | up | down] blocks, and for NVFP4 the FP32 scale tail."""
    entries, n = args[0], args[1]
    out = []
    for e in range(n):
        parts, tail = [], []
        for (wp, wo, wshape), sent, (tp, to) in entries[e]:
            w = np.fromfile(wp, dtype=np.uint8, count=wshape[0] * wshape[1], offset=wo).reshape(wshape)
            try:
                sc = _scales(sent)
            except S.ConvertError as err:
                return f"{err} (the matrix at {wp.name}+{wo})"
            if (sc & 0x80).any() or (sc == 0x7F).any():
                return f"a UE4M3 scale with the sign bit set or NaN at {wp.name}+{wo}"
            tail.append(np.fromfile(tp, dtype="<f4", count=1, offset=to)[0])
            parts.append(nvfp4_rows(w, sc))
        t = np.array(tail + [0.0], dtype="<f4")
        if not (np.isfinite(t).all() and (t[:3] > 0).all()):
            return f"a weight_scale_2 that is not finite and positive: {t[:3].tolist()}"
        out.append(np.concatenate(parts).tobytes())
        out.append(t.tobytes())
    return b"".join(out)


bounded_imap = S.bounded_imap


# How a qwen35 checkpoint tensor becomes its GGUF tensor (checkpoint_qwen35.py).
METHODS = {
    "id": "as stored",
    "norm": "bf16 + 1 in f32 (exact)",
    "heads": "value-head rows in the GGUF order",
    "negexp": "-exp(A_log), computed in f64 and rounded once to f32, value heads in the GGUF order",
    "conv": "[C, 1, K] -> [C, K]",
    "gconv": "[C, 1, K] -> [C, K], value channels in the GGUF head order",
    "vrows": "rows after q and k in the GGUF value-head order",
    "zrows": "rows in the GGUF value-head order",
    "vcols": "columns in the GGUF value-head order",
    "split": "the fused indexer projection split into q and k rows",
}


def _index(how: str, n: int, first: int) -> np.ndarray | None:
    if how in ("heads", "negexp"):
        return S.gdn_block_source()
    if how in ("vrows", "gconv"):
        return S.gdn_index(n, first)
    if how in ("zrows", "vcols"):
        return S.gdn_index(n, 0)
    return None


# ------------------------------------------------------------------ metadata and tokenizer
def front_matter(readme: pathlib.Path) -> dict:
    """The simple `key: value` / `key:\\n- item` YAML block at the top of a model card."""
    out, key = {}, None
    if not readme.is_file():
        return out
    lines = readme.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return out
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line.startswith("- ") and key:
            out.setdefault(key, []).append(line[2:].strip())
        elif ":" in line:
            key, v = (s.strip() for s in line.split(":", 1))
            if v:
                out[key] = v
    return out


def f32(v) -> float:
    return float(np.float32(v))


def tokenizer_kv(ck: Checkpoint, pre_name: str = "qwen35", split: str = QWEN35_HF_SPLIT,
                 n_vocab: int | None = None) -> dict:
    """The tokenizer as GGUF metadata (llama.cpp's gpt2 conversion): ids without a token become `[PADn]` (unused),
    added tokens are control tokens when marked special or written `<|...|>`, other added tokens user-defined.
    `pre_name` is the pre-tokenizer the engine runs, `split` the one regex the checkpoint's tokenizer.json must carry
    for it; `n_vocab` the ids kept (default the config's vocab_size)."""
    d = ck.dir
    tj = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
    tcfg = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
    model = tj.get("model", {})
    if model.get("type") != "BPE":
        raise S.ConvertError(f"tokenizer.json: a {model.get('type')} model; only byte-level BPE is converted")
    pre = tj.get("pre_tokenizer") or {}
    splits = [p.get("pattern", {}).get("Regex") for p in pre.get("pretokenizers", []) if p.get("type") == "Split"]
    if splits != [split]:
        raise S.ConvertError(f"tokenizer.json: the pre-tokenizer is not the {pre_name} split")
    n_vocab = int(ck.text["vocab_size"]) if n_vocab is None else int(n_vocab)
    rev = {int(i): t for t, i in model["vocab"].items()}
    added = {int(a["id"]): a for a in tj.get("added_tokens", [])}
    for i, a in added.items():
        rev[i] = a["content"]
    if max(rev) >= n_vocab:
        raise S.ConvertError(f"tokenizer.json: id {max(rev)} is beyond the vocabulary of {n_vocab}")
    tokens, types = [], []
    for i in range(n_vocab):
        if i not in rev:
            tokens.append(f"[PAD{i}]")
            types.append(UNUSED)
            continue
        t = rev[i]
        tokens.append(t)
        if i in added:
            special = added[i].get("special") or (t.startswith("<|") and t.endswith("|>"))
            types.append(CONTROL if special else USER_DEFINED)
        else:
            types.append(NORMAL)
    merges = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]
    ids = {t: i for i, t in enumerate(tokens)}

    def token_id(key):
        v = tcfg.get(key)
        v = v.get("content") if isinstance(v, dict) else v
        return ids.get(v) if v else None

    gen = json.loads((d / "generation_config.json").read_text(encoding="utf-8")) if (d / "generation_config.json").is_file() else {}
    kv = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": pre_name, "tokenizer.ggml.tokens": tokens,
          "tokenizer.ggml.token_type": types, "tokenizer.ggml.merges": merges}
    eos, pad = token_id("eos_token"), token_id("pad_token")
    bos = token_id("bos_token")
    if bos is None:
        bos = gen.get("bos_token_id", ck.text.get("bos_token_id"))
    for k, v in (("eos", eos), ("padding", pad), ("bos", bos)):
        if v is not None:
            kv[f"tokenizer.ggml.{k}_token_id"] = int(v)
    kv["tokenizer.ggml.add_bos_token"] = bool(tcfg.get("add_bos_token", False))
    tpl = d / "chat_template.jinja"
    template = tpl.read_text(encoding="utf-8") if tpl.is_file() else tcfg.get("chat_template")
    if template:
        kv["tokenizer.chat_template"] = template
    return kv


