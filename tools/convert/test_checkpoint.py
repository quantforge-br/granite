"""Tests for checkpoint.py on small synthetic safetensors files.

    python3 -m unittest discover -s tools/convert -p 'test_checkpoint.py'"""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import checkpoint as C  # noqa: E402
import model_sections as S  # noqa: E402

E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6]


def bf16(x: np.ndarray) -> np.ndarray:
    """float32 values that are exactly BF16 -> their u16 codes."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    assert not np.count_nonzero(u & 0xFFFF)
    return (u >> 16).astype("<u2")


def write_safetensors(path: Path, tensors: dict) -> None:
    """name -> (dtype, shape, raw bytes), in the given order."""
    head, blobs, at = {}, [], 0
    for k, (dt, shape, raw) in tensors.items():
        head[k] = {"dtype": dt, "shape": list(shape), "data_offsets": [at, at + len(raw)]}
        blobs.append(raw)
        at += len(raw)
    hb = json.dumps(head).encode()
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + b"".join(blobs))


def ue4m3(x) -> float:
    x = int(x)
    e, m = (x >> 3) & 15, x & 7
    return m / 8 * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)


def modelopt_values(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """What Model Optimizer's NVFP4 rows mean: value 2i is the low nibble of byte i, 2i+1 the high one."""
    rows = codes.shape[0]
    nib = np.empty((rows, codes.shape[1] * 2), np.uint8)
    nib[:, 0::2], nib[:, 1::2] = codes & 15, codes >> 4
    mag = np.array(E2M1, np.float64)[nib & 7] * np.where(nib & 8, -1, 1)
    return mag * np.vectorize(ue4m3)(scales).repeat(16, axis=1)


def basalt_values(blocks: np.ndarray, cols: int) -> np.ndarray:
    """Basalt's NVFP4 block decode (dequantize_nvfp4): sub-block s reads byte 8 s + j, low nibble value j,
    high nibble value j + 8."""
    b = blocks.reshape(-1, cols // 64, 36)
    out = np.empty((b.shape[0], cols // 64, 64))
    for s in range(4):
        sc = np.vectorize(ue4m3)(b[:, :, s])
        q = b[:, :, 4 + 8 * s:12 + 8 * s]
        for half, nib in ((0, q & 15), (8, q >> 4)):
            out[:, :, 16 * s + half:16 * s + half + 8] = (np.array(E2M1)[nib & 7] * np.where(nib & 8, -1, 1)) * sc[..., None]
    return out.reshape(b.shape[0], cols)


class Nvfp4RepackTest(unittest.TestCase):
    def test_values_survive_the_repack(self):
        rng = np.random.default_rng(0)
        codes = rng.integers(0, 256, size=(5, 64), dtype=np.uint8)            # 5 rows of 128 values
        scales = rng.integers(1, 0x7F, size=(5, 8), dtype=np.uint8)
        got = C.nvfp4_rows(codes, scales)
        self.assertEqual(got.size, 5 * S.nvfp4_row_bytes(128))
        np.testing.assert_array_equal(basalt_values(got, 128), modelopt_values(codes, scales))

    def test_expert_blob_and_tail(self):
        rng = np.random.default_rng(1)
        h, ff = 128, 64
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.safetensors"
            tens, want = {}, []
            shapes = {"gate_proj": (ff, h), "up_proj": (ff, h), "down_proj": (h, ff)}
            for i, (r, (rows, cols)) in enumerate(shapes.items()):
                w = rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
                s = rng.integers(1, 0x7F, size=(rows, cols // 16), dtype=np.uint8)
                tens[f"{r}.weight"] = ("U8", w.shape, w.tobytes())
                tens[f"{r}.weight_scale"] = ("F8_E4M3", s.shape, s.tobytes())
                tens[f"{r}.weight_scale_2"] = ("F32", (), np.float32(0.5 + i).tobytes())
                want.append(C.nvfp4_rows(w, s).tobytes())
            write_safetensors(p, tens)
            head, base = S.read_safetensors_header(p)
            ent = []
            for r, (rows, cols) in shapes.items():
                ent.append(((p, base + head[f"{r}.weight"]["data_offsets"][0], [rows, cols // 2]),
                            (p, base + head[f"{r}.weight_scale"]["data_offsets"][0], [rows, cols // 16]),
                            (p, base + head[f"{r}.weight_scale_2"]["data_offsets"][0])))
            blob = C._expert_job(([ent, ent], 2))
            one = b"".join(want) + np.array([0.5, 1.5, 2.5, 0], "<f4").tobytes()
            self.assertEqual(blob, one + one)
            # a scale byte with the sign bit set is refused
            raw = bytearray(p.read_bytes())
            raw[base + head["up_proj.weight_scale"]["data_offsets"][0]] |= 0x80
            p.write_bytes(bytes(raw))
            self.assertIsInstance(C._expert_job(([ent], 1)), str)


class TruncationTest(unittest.TestCase):
    def test_short_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.safetensors"
            write_safetensors(p, {"a": ("BF16", (4,), b"\1" * 8), "b": ("F32", (2,), b"\2" * 8)})
            self.assertEqual(S.read_safetensors_header(p)[0]["b"]["data_offsets"], [8, 16])
            full = p.read_bytes()
            for cut in (1, 9):
                p.write_bytes(full[:-cut])
                with self.assertRaisesRegex(S.ConvertError, "truncated"):
                    S.read_safetensors_header(p)
            p.write_bytes(full[:20])                       # inside the header
            with self.assertRaisesRegex(S.ConvertError, "truncated"):
                S.read_safetensors_header(p)
            p.write_bytes(full[:5])
            with self.assertRaises(S.ConvertError):
                S.read_safetensors_header(p)


class TokenizerKvTest(unittest.TestCase):
    def test_types_gaps_and_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            vocab = {"a": 0, "b": 1, "ab": 2}
            tj = {"model": {"type": "BPE", "vocab": vocab, "merges": [["a", "b"]]},
                  "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                      {"type": "Split", "pattern": {"Regex": C.QWEN35_HF_SPLIT}}, {"type": "ByteLevel"}]},
                  "added_tokens": [{"id": 4, "content": "<|end|>", "special": True},
                                   {"id": 5, "content": "<|fim|>", "special": False},
                                   {"id": 6, "content": "<think>", "special": False}]}
            (d / "tokenizer.json").write_text(json.dumps(tj))
            (d / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|end|>", "pad_token": "<|end|>",
                                                                 "bos_token": None, "add_bos_token": False}))
            (d / "generation_config.json").write_text(json.dumps({"bos_token_id": 4}))
            (d / "chat_template.jinja").write_text("{{ x }}")
            (d / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 8}}))
            (d / "model-00001-of-00001.safetensors").write_bytes(struct.pack("<Q", 2) + b"{}")
            kv = C.tokenizer_kv(C.Checkpoint(d))
            self.assertEqual(kv["tokenizer.ggml.tokens"], ["a", "b", "ab", "[PAD3]", "<|end|>", "<|fim|>", "<think>", "[PAD7]"])
            self.assertEqual(kv["tokenizer.ggml.token_type"], [1, 1, 1, 5, 3, 3, 4, 5])
            self.assertEqual(kv["tokenizer.ggml.merges"], ["a b"])
            self.assertEqual([kv[f"tokenizer.ggml.{k}_token_id"] for k in ("eos", "padding", "bos")], [4, 4, 4])
            self.assertEqual(kv["tokenizer.chat_template"], "{{ x }}")
            tj["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = r"\s+"
            (d / "tokenizer.json").write_text(json.dumps(tj))
            with self.assertRaises(S.ConvertError):
                C.tokenizer_kv(C.Checkpoint(d))


def csf_encode(scales: np.ndarray):
    """The CSF stream pair of scale bytes [rows, cols]: a row's base is the start of its fullest 16-wide window (at
    most 240); a byte outside base..base+15 becomes an exception word."""
    rows, cols = scales.shape
    base = np.zeros(rows, np.int16)
    for r in range(rows):
        v = scales[r].astype(np.int16)
        counts = [int(((v >= b) & (v <= b + 15)).sum()) for b in range(int(v.min()), int(v.max()) + 1)]
        base[r] = min(int(v.min()) + int(np.argmax(counts)), 240)
    code = scales.astype(np.int16) - base[:, None]
    exc = (code < 0) | (code > 15)
    code = np.where(exc, 0, code).astype(np.uint8)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    fixed = np.concatenate([base.reshape(-1, 16).astype(np.uint8), packed.reshape(rows // 16, -1)], axis=1)
    pos = np.flatnonzero(exc)
    return fixed, (pos | scales.reshape(-1)[pos].astype(np.int64) << 24).astype(np.uint32)


class CsfTest(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(3)
        scales = (rng.integers(0, 16, size=(48, 40)) + rng.integers(20, 90, size=(48, 1))).astype(np.uint8)
        for r, c, b in ((5, 7, 0), (30, 0, 126), (47, 39, 1)):         # outside their rows' windows
            scales[r, c] = b
        fixed, exc = csf_encode(scales)
        self.assertEqual((fixed.shape, exc.size), ((3, 16 + 8 * 40), 3))
        np.testing.assert_array_equal(C.csf_decode(fixed, exc), scales)

    def test_by_hand(self):
        fixed = np.zeros((1, 32), np.uint8)                            # 16 rows of 2 scales
        fixed[0, :16] = 40                                             # every row's base
        fixed[0, 16] = 0x21                                            # row 0: codes 1, 2
        fixed[0, 17] = 0xF0                                            # row 1: codes 0, 15
        got = C.csf_decode(fixed, np.array([3 | 7 << 24], np.uint32))  # row 1, column 1 holds 7
        self.assertEqual(got[:2].tolist(), [[41, 42], [40, 7]])
        self.assertTrue((got[2:] == 40).all())

    def test_bad_streams_are_refused(self):
        fixed, exc = csf_encode(np.full((16, 4), 50, np.uint8))
        high = fixed.copy()
        high[0, 3] = 241
        for f, x, want in ((high, exc, "above 240"), (fixed, np.array([16 * 4], np.uint32), "past the end"),
                           (fixed[:, :-8], exc, "shape"), (fixed.reshape(-1), exc, "shape")):
            with self.subTest(want=want), self.assertRaisesRegex(S.ConvertError, want):
                C.csf_decode(f, x)


if __name__ == "__main__":
    unittest.main()

