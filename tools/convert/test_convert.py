"""Tests for convert.py's shared parts and the model_sections.py helpers the checkpoint converters use.

    python3 -m unittest discover -s tools/convert -p 'test_convert.py'"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import convert  # noqa: E402
import ggufq  # noqa: E402
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


class Q6KTest(unittest.TestCase):
    """Q6_K blocks decode (through the reference dequantizer) to within the 6-bit error of their source."""

    def dequant(self, blob: bytes) -> np.ndarray:
        return ggufq.dequantize(np.frombuffer(blob, dtype=np.uint8), ggufq.GGMLQuantizationType.Q6_K).reshape(-1)

    def test_error_bound(self):
        rng = np.random.default_rng(3)
        x = (rng.standard_normal((64, 512)) * 0.02).astype(np.float32)
        x[3, :256] = 0.0
        x[4, 7] = 0.5
        got = S.q6_k(x)
        self.assertEqual(len(got), 64 * 2 * S.Q6K_BYTES)
        deq = self.dequant(got).reshape(x.shape)
        err = deq - x
        self.assertLess(np.sqrt((err ** 2).mean()), 0.025 * x.std())
        blk = np.abs(x).reshape(-1, 256).max(axis=1)
        self.assertTrue((np.abs(err).reshape(-1, 256).max(axis=1) <= blk * 0.035 + 1e-7).all())
        np.testing.assert_array_equal(deq[3, :256], 0.0)

    def test_large_input_quantizes_each_block_once(self):
        """A matrix of many blocks must quantize each block the same as it does alone: a per-group term that goes
        wrong only at scale zeroes whole 16-value groups, which both checks below catch."""
        rng = np.random.default_rng(6)
        x = (rng.standard_normal((512, 512)) * 0.03).astype(np.float32)       # 1024 blocks
        blob = S.q6_k(x)
        self.assertEqual(blob[:len(S.q6_k(x[:8]))], S.q6_k(x[:8]))
        deq = self.dequant(blob).reshape(x.shape)
        self.assertLess(np.sqrt(((deq - x) ** 2).mean()), 0.02 * x.std())
        self.assertLess((deq == 0).mean(), 0.06)


class GgufStemTest(unittest.TestCase):
    """arch.gguf_stem: the file name alone, a trailing `.gguf` (any case) and a shard suffix (any case) of any digit
    count stripped."""

    def test_stems(self):
        self.assertEqual(convert.A.gguf_stem("/models/unsloth/Some-Model-UD-Q4_K_XL-00001-of-00004.GGUF"),
                         "Some-Model-UD-Q4_K_XL")
        self.assertEqual(convert.A.gguf_stem("Some-Model-UD-Q4_K_XL-1-of-4.gguf"), "Some-Model-UD-Q4_K_XL")
        self.assertEqual(convert.A.gguf_stem("Some-Model-UD-Q4_K_XL-00001-OF-00004.gguf"),
                         "Some-Model-UD-Q4_K_XL")
        self.assertEqual(convert.A.gguf_stem("model.gguf"), "model")
        self.assertEqual(convert.A.gguf_stem("model"), "model")


if __name__ == "__main__":
    unittest.main()
