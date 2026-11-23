"""Sections of a .basalt file built from the source weights, for tools/convert/convert.py.

  dense / dense.index       float tensors in the form the engine reads, and the table of every non-expert tensor
  experts / experts.layout  per layer, n_expert blobs of [gate | up | down] rows, and where each layer starts

Sections are produced as byte-chunk iterators for ModelWriter.add_chunks. Needs numpy.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
import struct
import sys
from collections import deque

import numpy as np
import os

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import ggufq  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402


class ConvertError(Exception):
    """An input that cannot be converted; the message says why."""


FLOAT = {"BF16", "F32", "F16"}
ROLES = ("gate", "up", "down")
ALIGN = 64                      # between the tensors of `dense`
EXPERT_RE = re.compile(r"^blk\.\d+\.ffn_(gate|up|down|gate_up)_exps\.(weight|scale|input_scale)$")
KIND = {"BF16": "4", "F16": "5", "F32": "2"}


# ------------------------------------------------------------------ the source model
def expand_shards(first: pathlib.Path) -> list[pathlib.Path]:
    """`<name>-00001-of-0000N.gguf` -> all N shards (a missing one is an error); any other name -> itself."""
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", first.name)
    if not m:
        return [first]
    total = int(m.group(2))
    paths = [first.with_name(first.name[:m.start()] + "-%05d-of-%05d.gguf" % (i, total)) for i in range(1, total + 1)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise ConvertError("missing model shards: " + ", ".join(missing))
    return paths


def check_split(files) -> None:
    """The split keys as the engine checks them: shard 1 carries the metadata and every shard declares split.count / split.no consistently."""
    n = len(files)
    meta0 = files[0].metadata
    if n == 1:
        if int(meta0.get("split.count", 1)) > 1:
            raise ConvertError(f"{files[0].path.name} is shard 1 of {meta0['split.count']}, but its name has no "
                               "-00001-of-0000N.gguf to find the others by")
        return
    if "general.architecture" not in meta0:
        raise ConvertError(f"{files[0].path.name} has no general.architecture; the first shard of a split model "
                           "carries the metadata")
    total = meta0.get("split.tensors.count")
    for i, g in enumerate(files):
        md = g.metadata
        if md.get("split.count") != n or md.get("split.no") != i or \
                (total is not None and md.get("split.tensors.count") != total):
            raise ConvertError(f"{g.path.name} does not declare itself shard {i + 1} of {n} of this model "
                               "(split.count / split.no / split.tensors.count)")
    if total is not None and sum(len(g.tensors) for g in files) != total:
        raise ConvertError(f"the {n} shards hold {sum(len(g.tensors) for g in files)} tensors, but "
                           f"split.tensors.count is {total}")


class SourceModel:
    """All shards of one model: name -> (GGUFFile, TensorInfo, shard path), in file order."""

    def __init__(self, paths, described=frozenset()):
        """`described`: tensors whose shard may stop after its header; their names land in `short`."""
        self.paths = [pathlib.Path(p) for p in paths]
        self.files = [GGUFFile(p) for p in self.paths]
        self.sizes = [p.stat().st_size for p in self.paths]
        check_split(self.files)
        self._maps: dict[pathlib.Path, np.ndarray] = {}
        self.where: dict[str, tuple] = {}
        self.short: set[str] = set()
        for p, g, size in zip(self.paths, self.files, self.sizes):
            for t in g.tensors:
                n = t.expected_bytes()
                if n is not None and g.data_start + t.offset + n > size and t.name in described:
                    self.short.add(t.name)
                elif n is None or g.data_start + t.offset + n > size:
                    raise ConvertError(f"{p.name}: unsupported or truncated tensor {t.name}")
                if t.name in self.where:
                    raise ConvertError(f"{p.name}: duplicate tensor {t.name}")
                self.where[t.name] = (g, t, p)

    def bytes(self, name: str) -> np.ndarray:
        g, t, p = self.where[name]
        if p not in self._maps:
            self._maps[p] = np.memmap(p, dtype=np.uint8, mode="r")
        a = g.data_start + t.offset
        return self._maps[p][a:a + t.expected_bytes()]

    def abs_range(self, name: str):
        """(shard path, absolute offset, size) of a tensor's bytes."""
        g, t, p = self.where[name]
        return p, g.data_start + t.offset, t.expected_bytes()


def is_expert(name: str) -> bool:
    return bool(EXPERT_RE.match(name))


# ------------------------------------------------------------------ dense
def _dequantize(raw: np.ndarray, type_name: str) -> np.ndarray:
    """The tensor's values as float32 (floats exactly, quantized types through tools/ggufq.py)."""
    if type_name == "F32":
        return raw.view(np.float32)
    if type_name == "F16":
        return raw.view(np.float16).astype(np.float32)
    if type_name == "BF16":
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return ggufq.dequantize(raw, ggufq.GGMLQuantizationType[type_name]).astype(np.float32).reshape(-1)


def bf16_bytes(raw: np.ndarray, type_name: str) -> bytes:
    """Round-to-nearest-even BF16 of the tensor's values (ggml's conversion, halfway cases included)."""
    values = ggufq.dequantize(raw, ggufq.GGMLQuantizationType[type_name])
    if not np.isfinite(values).all():
        raise ValueError("cannot convert non-finite weights to BF16")
    return ggufq.quantize(values, ggufq.GGMLQuantizationType.BF16).tobytes()


def f16_values(values: np.ndarray, name: str) -> np.ndarray:
    """values -> F16, nearest-even; refuses a value that is not finite or overflows F16."""
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: cannot convert non-finite weights to F16")
    with np.errstate(over="ignore"):
        h = values.astype(np.float16)
    if not np.isfinite(h).all():
        raise ValueError(f"{name}: a value is beyond F16's range, which the engine reads it as")
    return h


def convert_tensor(name: str, type_name: str, raw: np.ndarray, compat_bf16: bool, arch):
    """The dense form of one tensor: (index kind, stored bytes, destination bytes, record or None), or an error string.
        `record` describes a conversion for meta.conversions."""
    form = arch.form_of(name)
    if form is None or form == type_name:
        return KIND[type_name], raw.tobytes(), raw.nbytes, None
    values = _dequantize(raw, type_name)
    rec = {"src_type": type_name, "dst_type": form}
    if form == "BF16":
        u = values.view(np.uint32) if type_name == "F32" else None
        if u is not None and not np.count_nonzero(u & 0xFFFF):
            if not np.isfinite(values).all():
                return f"{name} is F32 with non-finite values; refused rather than converted"
            data = (u >> 16).astype("<u2").tobytes()       # exact BF16 values
            rec.update(method="f32->bf16: the values are BF16's (low 16 bits zero)", exact=True, max_abs_err=0.0)
            return "4", data, len(data), rec
        if not compat_bf16:
            what = "F32 with values that are not BF16" if type_name == "F32" else type_name
            return f"{name} is {what}, but the engine requires BF16; use --compat-bf16 (rounds to nearest-even)"
        data = bf16_bytes(raw, type_name)
        got = (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        rec.update(method="dequantize->bf16 round-to-nearest-even (--compat-bf16)" if type_name not in FLOAT else
                   "%s->bf16 round-to-nearest-even (--compat-bf16)" % type_name.lower())
        rec.update(exact=bool(np.array_equal(got, values)),
                   max_abs_err=float(np.max(np.abs(got.astype(np.float64) - values))) if values.size else 0.0)
        return "4", data, len(data), rec
    if form == "F16":
        if type_name != "F32" and not compat_bf16:
            return f"{name} is {type_name}, but the engine reads F16; use --compat-bf16 (rounds to nearest-even)"
        h = f16_values(values, name)
        err = float(np.max(np.abs(h.astype(np.float64) - values))) if values.size else 0.0
        rec.update(exact=bool(np.array_equal(h.astype(np.float32), values)), max_abs_err=err)
        if type_name == "F32":
            # stored as F32 (index kind 3); the loader narrows to F16
            rec.update(method="f32->f16 by the loader (index kind 3, round-to-nearest-even)")
            return "3", raw.tobytes(), raw.nbytes // 2, rec
        rec.update(method="%s->f16 round-to-nearest-even (--compat-bf16)" % type_name.lower())
        data = h.astype("<f2").tobytes()
        return "5", data, len(data), rec
    # form == "F32"
    if type_name not in ("F16", "BF16") and not compat_bf16:
        return f"{name} is {type_name}, but the engine reads F32; use --compat-bf16"
    data = values.astype("<f4").tobytes()
    rec.update(method="%s->f32 widening" % type_name.lower(), exact=True, max_abs_err=0.0)
    return "2", data, len(data), rec


class Dense:
    """The `dense` section: `plan` validates every tensor first, `chunks` streams the pool and fills `rows` and `records`,
        `index_text` is the `dense.index`."""

    def __init__(self, model: SourceModel, compat_bf16: bool, arch, only=None):
        """`only`: a predicate on tensor names; the tensors it rejects are not in this pool."""
        self.model = model
        self.arch = arch
        self.compat_bf16 = compat_bf16
        self.todo = []
        self.rows: list[list[str]] = []
        self.records: list[dict] = []
        self.pool_bytes = 0
        problems = []
        for name, (g, t, p) in model.where.items():
            if is_expert(name) or name in arch.not_in_dense or (only is not None and not only(name)):
                continue
            form = arch.form_of(name)
            # quantized tensors are served from the source unless the pool needs them dequantized (--compat-bf16)
            native = t.type_name not in FLOAT and arch.native(name, t.type_name)
            # a row gives the first dimension and the rest flattened; meta.tensors has a served tensor's own shape
            if t.type_name not in FLOAT and not native and not compat_bf16:
                problems.append(f"{name} is {t.type_name}, but the engine requires {form}; use --compat-bf16")
            self.todo.append((name, t, p, native))
        if problems:
            more = f"\n... and {len(problems) - 8} more" if len(problems) > 8 else ""
            raise ConvertError("\n".join(problems[:8]) + more)

    def chunks(self):
        at = 0
        for name, t, p, native in self.todo:
            ne0 = int(t.shape[0])
            ne1 = math.prod(int(d) for d in t.shape[1:]) if len(t.shape) > 1 else 0
            if native:
                self.rows.append([name, "0", "0", "0", "0", "0", "0", str(ne0), str(ne1), "8", "0", "32"] + ["0"] * 7)
                continue
            raw = self.model.bytes(name)
            try:
                got = convert_tensor(name, t.type_name, raw, self.compat_bf16, self.arch)
            except ValueError as e:
                got = str(e)
            if isinstance(got, str):
                raise ConvertError(got)
            kind, data, dst_bytes, rec = got
            if rec is not None:
                self.records.append({"name": name, "shard": p.name, **rec,
                                     "source_sha256": hashlib.sha256(raw.tobytes()).hexdigest()})
            self.rows.append([name, "0", kind, str(at), str(len(data)), "0", str(dst_bytes), str(ne0), str(ne1),
                              "0", "0", "1"] + ["0"] * 7)
            yield data
            pad = (-len(data)) % ALIGN
            if pad:
                yield b"\0" * pad
            at += len(data) + pad

    def index_text(self) -> bytes:
        at = 0
        for r in self.rows:
            r[5] = str(at)
            at += (int(r[6]) + ALIGN - 1) // ALIGN * ALIGN
        self.pool_bytes = at
        out = ["# dense index v3\n", "# align %d pool %d tensors %d\n" % (ALIGN, at, len(self.rows))]
        out += [" ".join(r) + "\n" for r in self.rows]
        return "".join(out).encode("utf-8")


# ------------------------------------------------------------------ experts
class Experts:
    """The `experts` section and its `experts.layout`; each role is resolved by name in the shard that holds it."""

    def __init__(self, model: SourceModel, layers=None):
        """`layers`: the layer ids that hold routed experts (default 0 .. the last layer with expert tensors); the
            layout lists exactly these, by layer id."""
        T = {n: w[1] for n, w in model.where.items()}
        exps = [n for n in T if n.startswith("blk.") and n.endswith("_exps.weight")]
        if not exps:
            raise ConvertError("the model has no expert tensors (blk.N.ffn_{gate,up,down}_exps.weight)")
        self.model = model
        self.layers = list(layers) if layers is not None else list(range(1 + max(int(n.split(".")[1]) for n in exps)))
        if not self.layers:
            raise ConvertError("no layer holds routed experts")
        self.n_layers = len(self.layers)
        # router rows = experts kept (a pruned model has fewer)
        self.n_expert = int(T["blk.%d.ffn_gate_inp.weight" % self.layers[0]].shape[1])
        if any(int(T["blk.%d.ffn_gate_inp.weight" % l].shape[1]) != self.n_expert for l in self.layers):
            raise ConvertError("the routers disagree on the expert count; a per-layer pruned model cannot be converted")
        self.layout = []
        offset = 0
        for l in self.layers:
            names = ["blk.%d.ffn_%s_exps.weight" % (l, r) for r in ROLES]
            if any(n not in T for n in names):
                raise ConvertError("layer %d: missing %s" % (l, ", ".join(n for n in names if n not in T)))
            ts = [T[n] for n in names]
            if any(t.expected_bytes() is None or len(t.shape) != 3 or int(t.shape[2]) != self.n_expert for t in ts):
                raise ConvertError("layer %d: an expert tensor is not [*, *, %d] of whole blocks" % (l, self.n_expert))
            per = [t.expected_bytes() // self.n_expert for t in ts]
            blob = sum(per)
            self.layout.append((l, ts[0].type_id, ts[2].type_id, offset, blob, names, ts[1].type_id))
            offset += blob * self.n_expert
        self.total = offset

    def layout_text(self) -> bytes:
        """v5; v7 (a sixth column, the up rows' type) only when a layer's gate and up differ in type, so every other
            model stays readable by engines from before v7."""
        mixed = any(gt != ut for _, gt, _, _, _, _, ut in self.layout)
        if not mixed:
            head = ("# experts v5: layer gu_type d_type offset blob_bytes (n_expert %d, total %d)\n"
                    % (self.n_expert, self.total))
            return (head + "".join("%d %d %d %d %d\n" % (l, gt, dt, off, blob)
                                   for l, gt, dt, off, blob, _, _ in self.layout)).encode("utf-8")
        head = ("# experts v7: layer gu_type d_type offset blob_bytes up_type (n_expert %d, total %d)\n"
                % (self.n_expert, self.total))
        return (head + "".join("%d %d %d %d %d %d\n" % (l, gt, dt, off, blob, ut)
                               for l, gt, dt, off, blob, _, ut in self.layout)).encode("utf-8")

    def chunks(self, group: int = 32):
        """The blobs layer by layer, `group` experts at a time: each expert is [gate | up | down] of its slices."""
        for l, gt, dt, off, blob, names, _ in self.layout:
            arrays = [self.model.bytes(n).reshape(self.n_expert, -1) for n in names]
            for s in range(0, self.n_expert, group):
                yield np.concatenate([a[s:s + group] for a in arrays], axis=1).tobytes()


NVFP4 = 40
NVFP4_TAIL = 16          # per expert, after down: FP32 {s_gate, s_up, s_down, 0}


def nvfp4_row_bytes(n: int) -> int:
    """Bytes of one NVFP4 row of n values (blocks of 64 in 36 bytes)."""
    if n % 64:
        raise ConvertError(f"an NVFP4 row of {n} values is not whole blocks")
    return n // 64 * 36


def nvfp4_layout_text(n_expert: int, total: int, layout) -> bytes:
    """The v6 `experts.layout`: one `layer gu_type d_type offset blob_bytes` line per layer."""
    head = ("# experts v6: layer gu_type d_type offset blob_bytes (n_expert %d, total %d); "
            "an NVFP4 blob ends in FP32 {s_gate, s_up, s_down, 0}\n" % (n_expert, total))
    return (head + "".join("%d %d %d %d %d\n" % x for x in layout)).encode("utf-8")


# ------------------------------------------------------------------ quantization and checkpoint files
def q8_0(x: np.ndarray) -> bytes:
    """ggml's reference rounding over rows of a float32 array: blocks of 32, fp16 d = amax / 127."""
    b = x.reshape(-1, 32).astype(np.float32)
    d = np.abs(b).max(axis=1) / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0).astype(np.float32)
    v = b * inv[:, None]
    q = (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int8)
    out = np.empty((b.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def _bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def fp8_e4m3_table() -> np.ndarray:
    """All 256 E4M3 (finite-only, bias 7) codes as float32; the two NaN codes are NaN."""
    b = np.arange(256)
    sign = np.where(b >> 7, -1.0, 1.0)
    e, m = (b >> 3) & 15, b & 7
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e.astype(np.float64) - 7))
    v = np.where((e == 15) & (m == 7), np.nan, v)
    return (sign * v).astype(np.float32)


def read_safetensors_header(path):
    """(header, offset of the data) of a safetensors file; refuses a file shorter than its header says."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ConvertError(f"{path}: too short for a safetensors header")
        n = struct.unpack("<Q", raw)[0]
        if 8 + n > size:
            raise ConvertError(f"{path}: truncated, the header alone ({n} bytes) is past the end of the file")
        head = json.loads(f.read(n))
    end = 0
    for k, v in head.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        if not 0 <= a <= b:
            raise ConvertError(f"{path}: {k} has bad data offsets {a}..{b}")
        end = max(end, b)
    if 8 + n + end > size:
        raise ConvertError(f"{path}: truncated, its header covers {8 + n + end} bytes but the file has {size}")
    return head, 8 + n


def bounded_imap(pool, fn, jobs, depth: int):
    """pool.imap with at most `depth` results waiting, so a slow writer never piles them up in memory."""
    q = deque()
    for j in jobs:
        q.append(pool.apply_async(fn, (j,)))
        if len(q) >= depth:
            yield q.popleft().get()
    while q:
        yield q.popleft().get()


# ------------------------------------------------------------------ Q6_K
Q6K_BLOCK, Q6K_BYTES = 256, 210


def q6_k(x: np.ndarray) -> bytes:
    """Q6_K blocks of 256 over a float32 array (length a multiple of 256): per 16-value group the least-squares
    scale from the signed-max starting grid, 8-bit group scales against one fp16 super-scale, values re-rounded
    against the scales as stored. The layout is ggml block_q6_K.

    Every per-group term is built by 16 explicit elementwise steps instead of a reduction over the last axis: a bulk
    reduction can return wrong sums for these shapes, which shows up as whole 16-value groups turning to zero. The
    elementwise form is exact and costs nothing at this width."""
    nb = x.size // Q6K_BLOCK
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(nb, 16, 16)
    with np.errstate(divide="ignore", invalid="ignore"):
        # the signed value of the largest magnitude in each group (the first of equals, as argmax takes)
        mx = x[:, :, 0].copy()
        for k in range(1, 16):
            v = x[:, :, k]
            mx = np.where(np.abs(v) > np.abs(mx), v, mx)
        start = np.where(mx != 0, -32.0 / np.where(mx == 0, 1, mx), 0.0)
        l = np.clip(np.rint(x * start[..., None]), -32, 31)
        x64 = x.astype(np.float64)
        l64 = l.astype(np.float64)
        den = np.zeros((nb, 16))
        num = np.zeros((nb, 16))
        for k in range(16):
            c, v = l64[:, :, k], x64[:, :, k]
            den += c * c
            num += v * c
        scale = np.where(den > 0, num / np.where(den == 0, 1, den), 0.0)
        smax = scale[:, 0].copy()
        for g in range(1, 16):
            smax = np.where(np.abs(scale[:, g]) > np.abs(smax), scale[:, g], smax)
        d = smax / 127.0
        d16 = d.astype(np.float16)
        sc = np.clip(np.rint(scale / np.where(d == 0, 1, d)[..., None]), -128, 127)
        eff = d16.astype(np.float32)[..., None] * sc
        q = np.where(eff[..., None] != 0, np.rint(x / np.where(eff == 0, 1, eff)[..., None]), 0)
    q = np.clip(q, -32, 31).astype(np.int32) + 32
    q = q.reshape(nb, 2, 4, 32)
    ql = np.empty((nb, 2, 64), np.uint8)
    ql[..., :32] = (q[:, :, 0] & 15) | ((q[:, :, 2] & 15) << 4)
    ql[..., 32:] = (q[:, :, 1] & 15) | ((q[:, :, 3] & 15) << 4)
    qh = (q[:, :, 0] >> 4) | ((q[:, :, 1] >> 4) << 2) | ((q[:, :, 2] >> 4) << 4) | ((q[:, :, 3] >> 4) << 6)
    out = np.empty((nb, Q6K_BYTES), np.uint8)
    out[:, 0:128] = ql.reshape(nb, 128)
    out[:, 128:192] = qh.astype(np.uint8).reshape(nb, 64)
    out[:, 192:208] = sc.astype(np.int8).view(np.uint8).reshape(nb, 16)
    out[:, 208:210] = d16.view(np.uint8).reshape(nb, 2)
    return out.tobytes()


