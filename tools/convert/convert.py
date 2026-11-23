"""Build the single .granite model file from a safetensors checkpoint.

    python3 tools/convert/convert.py --checkpoint INKLING2_DIR --out MODEL.granite [--name NAME] [--workers N] [--force]

From Thinking Machines' Inkling 2 Mini release (tools/convert/checkpoint_inkling2.py)
  --checkpoint DIR   a checkpoint of model type inkling2_loop: the release's safetensors (the model and its DSpark
                     drafter), config.json, hf_quant_config.json, the tokenizer files and the chat template. Everything
                     comes from DIR

Options
  --name NAME        the model's name, reported by the server
  --workers N        converter processes
  --force            overwrite --out

It writes a version 2 file. Sections are listed in include/granite/core/model_format.hpp. Everything is streamed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))   # for model_file
from model_file import ModelWriter  # noqa: E402

FINGERPRINT_BYTES = 64 << 20


def die(msg: str):
    raise SystemExit(f"convert: {msg}")


# ------------------------------------------------------------------ helpers
def fingerprint(path: Path) -> str:
    """sha256 over the first and last 64 MiB and the size: fast, and enough to tell two files apart."""
    size = path.stat().st_size
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(FINGERPRINT_BYTES))
        f.seek(max(0, size - FINGERPRINT_BYTES))
        h.update(f.read(FINGERPRINT_BYTES))
    h.update(struct.pack("<Q", size))
    return h.hexdigest()


# ------------------------------------------------------------------ the conversion
class Job:
    def __init__(self, writer: ModelWriter):
        self.w = writer
        self.rows: list[tuple[str, int]] = []
        self.t0 = time.time()

    def _report(self, name, nbytes, secs):
        mbs = nbytes / (1 << 20) / secs if secs > 0 else float("inf")
        print(f"  {name:<44} {nbytes / (1 << 30):9.3f} GiB {mbs:9.0f} MB/s", flush=True)
        self.rows.append((name, nbytes))

    def bytes(self, name, data: bytes):
        t = time.time()
        self.w.add_bytes(name, data)
        self._report(name, len(data), time.time() - t)

    def file(self, name, src, offset=0, size=None):
        t = time.time()
        self.w.add_file(name, str(src), offset, size)
        self._report(name, self.w.entries[-1][2], time.time() - t)

    def chunks(self, name, chunks):
        t = time.time()
        n = self.w.add_chunks(name, chunks)
        self._report(name, n, time.time() - t)


def checkpoint_model_type(d: Path) -> str | None:
    """The `model_type` of a checkpoint directory's config.json (None when it has none)."""
    cfg = d / "config.json"
    if not cfg.is_file():
        return None
    return json.loads(cfg.read_text(encoding="utf-8")).get("model_type")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="the .granite file to write")
    ap.add_argument("--name")
    ap.add_argument("--checkpoint", metavar="DIR", help="an Inkling 2 safetensors checkpoint (inkling2_loop): a whole inkling2 pack")
    ap.add_argument("--workers", type=int, default=16, help="converter processes")
    ap.add_argument("--force", action="store_true", help="overwrite --out")
    a = ap.parse_args()
    if not a.out:
        ap.error("--out is required")
    if not a.checkpoint:
        ap.error("--checkpoint is required")
    if not Path(a.checkpoint).is_dir():
        ap.error(f"missing input: {a.checkpoint}")
    model_type = checkpoint_model_type(Path(a.checkpoint))
    if model_type != "inkling2_loop":
        ap.error(f"{a.checkpoint}: model type {model_type!r}; --checkpoint takes an Inkling 2 checkpoint (inkling2_loop)")
    import checkpoint_inkling2 as I
    I.build(a, Job, fingerprint, die)


if __name__ == "__main__":
    main()
