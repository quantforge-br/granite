"""Tests for the model_sections.py helpers the checkpoint converters use.

    python3 -m unittest discover -s tools/convert -p 'test_convert.py'"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_sections as S  # noqa: E402


class Fp8E4M3TableTest(unittest.TestCase):
    """Every E4M3 code decodes as the plain formula says."""

    @staticmethod
    def decode(code: int) -> float:
        sign, e, m = -1.0 if code >> 7 else 1.0, (code >> 3) & 15, code & 7
        if e == 15 and m == 7:
            return float("nan")
        return sign * (m / 8 * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))

    def test_table_matches_formula(self):
        want = np.array([self.decode(c) for c in range(256)], dtype=np.float32)
        np.testing.assert_array_equal(S.fp8_e4m3_table(), want)


class BoundedImapTest(unittest.TestCase):
    def test_order_and_depth(self):
        class Pool:
            live = peak = 0

            def apply_async(self, fn, args):
                Pool.live += 1
                Pool.peak = max(Pool.peak, Pool.live)
                return self

            def get(self):
                Pool.live -= 1
                return 1

        self.assertEqual(sum(S.bounded_imap(Pool(), None, range(20), 3)), 20)
        self.assertEqual(Pool.peak, 3)


def _dequant_q8_0(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    d = arr[:, :2].copy().view(np.float16).astype(np.float32)
    q = arr[:, 2:].view(np.int8).astype(np.float32)
    return (d * q).reshape(-1)


class Q80Pow2Test(unittest.TestCase):
    """P2.4: power-of-2 Q8_0 scale d = 2^(e_max - 6) maps top 4 MXFP8/E4M3 binades exactly."""

    def test_mxfp8_top4_binades_zero_error(self):
        e4m3 = S.fp8_e4m3_table()
        # Codes in the top 4 positive/negative binades (e in 12..15, excluding NaN 0x7F/0xFF)
        top_codes = [c for c in range(256) if ((c >> 3) & 15) >= 12 and (c & 0x7F) != 0x7F]
        rng = np.random.default_rng(42)
        n_blocks = 256
        codes = rng.choice(top_codes, size=(n_blocks, 32))
        # Ensure at least one element in binade e=15 per block so e_max aligns with the top binade
        e15_codes = [c for c in top_codes if ((c >> 3) & 15) == 15]
        codes[:, 0] = rng.choice(e15_codes, size=n_blocks)
        ue8m0 = rng.integers(115, 138, size=(n_blocks, 1), dtype=np.int32)
        vals = (e4m3[codes] * np.ldexp(np.float32(1.0), ue8m0 - 127)).astype(np.float32)
        got = _dequant_q8_0(S.q8_0_pow2(vals)).reshape(n_blocks, 32)
        max_abs = float(np.max(np.abs(got - vals)))
        self.assertEqual(max_abs, 0.0)

    def test_mxfp8_gaussian_beats_q8_0_rms_and_exact_fraction(self):
        e4m3 = S.fp8_e4m3_table()
        valid_vals = np.sort(e4m3[np.array([c for c in range(256) if (c & 0x7F) != 0x7F])])
        rng = np.random.default_rng(42)
        w = rng.normal(0.0, 0.02, size=(4096, 32)).astype(np.float32)
        amax = np.abs(w).max(axis=1, keepdims=True)
        _, exp = np.frexp(amax)
        scale = np.ldexp(np.float32(1.0), (exp - 1) - 8)
        scaled = np.clip(w / scale, -448.0, 448.0)
        idx = np.searchsorted(valid_vals, scaled)
        idx0 = np.clip(idx - 1, 0, len(valid_vals) - 1)
        idx1 = np.clip(idx, 0, len(valid_vals) - 1)
        pick = np.where(np.abs(scaled - valid_vals[idx0]) <= np.abs(scaled - valid_vals[idx1]),
                        valid_vals[idx0], valid_vals[idx1])
        mxfp8 = (pick * scale).astype(np.float32)

        v = mxfp8.reshape(-1)
        rms_v = float(np.sqrt(np.mean(v ** 2)))
        e_ref = _dequant_q8_0(S.q8_0(mxfp8)) - v
        e_pow2 = _dequant_q8_0(S.q8_0_pow2(mxfp8)) - v
        rel_ref = float(np.sqrt(np.mean(e_ref ** 2))) / rms_v
        rel_pow2 = float(np.sqrt(np.mean(e_pow2 ** 2))) / rms_v
        exact_frac = float(np.mean(e_pow2 == 0.0))
        self.assertLess(rel_pow2 * 1.3, rel_ref)
        self.assertGreater(exact_frac, 0.80)

    def test_edge_cases_zero_non_e4m3_and_subnormal(self):
        blocks = np.zeros((4, 32), dtype=np.float32)
        # Block 0: all zeros
        # Block 1: non-E4M3 mantissa > 127/64 (1.999) — must NOT overflow int8
        blocks[1, :] = np.linspace(-1.999, 1.999, 32, dtype=np.float32)
        # Block 2: tiny subnormal below fp16 power-of-2 floor (falls back to amax/127)
        blocks[2, :] = np.linspace(-1e-6, 1e-6, 32, dtype=np.float32)
        # Block 3: large values near 448.0
        blocks[3, :] = np.linspace(-448.0, 448.0, 32, dtype=np.float32)
        got = _dequant_q8_0(S.q8_0_pow2(blocks)).reshape(4, 32)
        self.assertTrue(np.all(np.isfinite(got)))
        np.testing.assert_array_equal(got[0], np.zeros(32, dtype=np.float32))
        self.assertLess(float(np.max(np.abs(got[1] - blocks[1]))), 0.02)
        self.assertLess(float(np.max(np.abs(got[3] - blocks[3]))), 4.0)


if __name__ == "__main__":
    unittest.main()
