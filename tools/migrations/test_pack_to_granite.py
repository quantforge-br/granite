"""Tests for pack_to_granite.py on tiny Basalt packs.

    python3 -m unittest discover -s tools/migrations -p 'test_*.py'"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "convert"))
import hc8 as H  # noqa: E402
import model_file  # noqa: E402
import pack_to_granite as M  # noqa: E402
from model_file import SECTION_ALIGN, ModelFile, ModelWriter  # noqa: E402

SCRIPT = HERE / "pack_to_granite.py"


def basalt_pack(path: Path, sections: list[tuple[str, bytes]], version: int = 2) -> None:
    """A pack as Basalt writes it: its header magic and the given version."""
    saved = model_file.MAGIC, model_file.VERSION
    model_file.MAGIC, model_file.VERSION = M.BASALT_MAGIC, version
    try:
        w = ModelWriter(str(path))
        for name, data in sections:
            w.add_bytes(name, data)
        w.close()
    finally:
        model_file.MAGIC, model_file.VERSION = saved


def meta_bytes(meta: dict) -> bytes:
    return json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class Convert(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        rng = np.random.default_rng(3)
        w = (rng.standard_normal((2, 64)).astype(np.float32).view(np.uint32) >> 16).astype("<u2")
        self.copy = H.quantize(w)
        table = H.table([("blk.0.hc_attn_down.weight", 2, 64, self.copy)])
        self.hc8 = M.BASALT_HC8_MAGIC + table[len(H.MAGIC):]
        self.meta = {"format": "basalt", "version": 2, "name": "Tiny ünï", "arch": "inkling2",
                     "tensors": {"token_embd.weight": {"offset": 4096, "format": "bf16"}}, "scale": 1e-06}
        self.sections = [("tokenizer/vocab.json", b'{"a":0}'), ("gguf", bytes(range(256)) * 40), ("hc8", self.hc8),
                         ("vision/mmproj", b"M" * 100), ("meta", meta_bytes(self.meta))]
        self.old = self.t / "old.basalt"
        basalt_pack(self.old, self.sections)

    def tearDown(self):
        self.tmp.cleanup()

    def test_convert(self):
        out = self.t / "new.granite"
        M.convert(self.old, out)
        m = ModelFile(str(out))
        self.assertEqual(out.read_bytes()[:8], b"GRANITMF")
        self.assertEqual(m.version, 2)
        self.assertEqual(m.names(), [n for n, _ in self.sections])
        for n in m.names():
            self.assertTrue(m.verify(n), n)
        for n, b in self.sections:
            if n not in ("meta", "hc8"):
                self.assertEqual(m.read(n), b, n)
        # meta: the format alone changes, the rest of the text stays as the converter wrote it
        self.assertEqual(m.read("meta"), meta_bytes(self.meta).replace(b'"format":"basalt"', b'"format":"granite"'))
        self.assertEqual(json.loads(m.read("meta")), dict(self.meta, format="granite"))
        # hc8: the magic alone changes
        self.assertEqual(m.read("hc8"), b"GRANITH8" + self.hc8[8:])
        self.assertEqual(M.HC8_MAGIC, H.MAGIC)
        self.assertEqual(H.read_table(m.read("hc8")), {"blk.0.hc_attn_down.weight": (2, 64, H.srow_of(64), self.copy)})

    def test_without_hc8(self):
        # Inkling 2 Mini's pack has no hc8 section
        old = self.t / "nohc8.basalt"
        sections = [s for s in self.sections if s[0] != "hc8"]
        basalt_pack(old, sections)
        out = self.t / "nohc8.granite"
        M.convert(old, out)
        m = ModelFile(str(out))
        self.assertEqual(m.names(), [n for n, _ in sections])
        self.assertEqual(json.loads(m.read("meta"))["format"], "granite")

    def test_a_longer_meta_moves_the_next_section(self):
        # meta grows by one byte; at a whole number of alignment units the next section starts one unit later
        meta = dict(self.meta, pad="")
        meta["pad"] = "x" * (SECTION_ALIGN - len(meta_bytes(meta)))
        self.assertEqual(len(meta_bytes(meta)), SECTION_ALIGN)
        old = self.t / "long.basalt"
        sections = [("meta", meta_bytes(meta)), ("gguf", b"G" * 5000)]
        basalt_pack(old, sections)
        out = self.t / "long.granite"
        M.convert(old, out)
        before, after = M.old_sections(old), ModelFile(str(out))
        self.assertEqual(after.section("gguf")[0], before["gguf"][0] + SECTION_ALIGN)
        self.assertEqual(after.read("gguf"), b"G" * 5000)
        self.assertTrue(after.verify("gguf") and after.verify("meta"))
        self.assertEqual(json.loads(after.read("meta")), dict(meta, format="granite"))

    def test_the_reader_names_the_script(self):
        with self.assertRaisesRegex(ValueError, "a Basalt pack; convert it with tools/migrations/pack_to_granite.py"):
            ModelFile(str(self.old))

    def run_script(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)

    def test_refusals(self):
        out = self.t / "new.granite"
        M.convert(self.old, out)
        r = self.run_script(out, self.t / "again.granite")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already a .granite pack", r.stderr)
        r = self.run_script(self.old, out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("exists (use --force", r.stderr)
        r = self.run_script(self.old, self.old, "--force")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("another file than OLD", r.stderr)
        junk = self.t / "junk.bin"
        junk.write_bytes(b"\1" * (model_file.HEADER_BYTES + 10))
        r = self.run_script(junk, self.t / "junk.granite")
        self.assertIn("not a Basalt pack (bad magic)", r.stderr)
        v1 = self.t / "v1.basalt"
        basalt_pack(v1, self.sections, version=1)
        r = self.run_script(v1, self.t / "v1.granite")
        self.assertIn("Basalt pack version 1; this script reads version 2", r.stderr)
        cut = self.t / "cut.basalt"
        cut.write_bytes(self.old.read_bytes()[:-10])
        r = self.run_script(cut, self.t / "cut.granite")
        self.assertIn("truncated", r.stderr)
        for name in ("junk", "v1", "cut"):
            self.assertFalse((self.t / f"{name}.granite").exists())
            self.assertFalse((self.t / f"{name}.granite.partial").exists())

    def test_a_damaged_section_is_refused(self):
        raw = bytearray(self.old.read_bytes())
        off, size, _ = M.old_sections(self.old)["gguf"]
        raw[off + size // 2] ^= 1
        bad = self.t / "bad.basalt"
        bad.write_bytes(bytes(raw))
        out = self.t / "bad.granite"
        with self.assertRaisesRegex(SystemExit, "section gguf: its bytes do not match"):
            M.convert(bad, out)
        self.assertFalse(out.exists())
        self.assertFalse(Path(str(out) + ".partial").exists())

    def test_the_cli_writes_the_pack(self):
        out = self.t / "cli.granite"
        r = self.run_script(self.old, out)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("done: 5 sections", r.stdout)
        self.assertEqual(ModelFile(str(out)).names(), [n for n, _ in self.sections])


if __name__ == "__main__":
    unittest.main()
