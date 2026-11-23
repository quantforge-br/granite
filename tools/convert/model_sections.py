"""What the checkpoint converters build sections with (tools/convert/checkpoint.py, checkpoint_inkling2.py): the NVFP4 row
size, ggml's Q8_0 rounding, the E4M3 codes, safetensors headers and a pool map that bounds what waits. Needs numpy.
"""
from __future__ import annotations

import json
import os
import struct
from collections import deque

import numpy as np


class ConvertError(Exception):
    """An input that cannot be converted; the message says why."""


NVFP4 = 40
NVFP4_TAIL = 16          # per expert, after down: FP32 {s_gate, s_up, s_down, 0}


def nvfp4_row_bytes(n: int) -> int:
    """Bytes of one NVFP4 row of n values (blocks of 64 in 36 bytes)."""
    if n % 64:
        raise ConvertError(f"an NVFP4 row of {n} values is not whole blocks")
    return n // 64 * 36


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

