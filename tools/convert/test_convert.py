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


if __name__ == "__main__":
    unittest.main()
