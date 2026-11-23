"""Write a .granite pack from an upstream Basalt pack (.basalt), without re-quantizing anything.

    python3 tools/migrations/pack_to_granite.py OLD.basalt NEW.granite [--force]

Both are version 2 of one container; granite's carries its own names. NEW gets the header magic GRANITMF, `"format":
"granite"` in `meta` and, when OLD has an `hc8` section, that table's magic GRANITH8. Every other section is copied byte
for byte and checked against the SHA-256 in OLD's table; `meta` and `hc8` are checked the same way before they are
rewritten and get new hashes. A tensor's offset in `meta` is relative to its section, so a section may start at another
file offset in NEW. OLD is only read; NEW is written beside it as NEW.partial and renamed when complete. Stdlib only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
from model_file import BASALT_MAGIC, ENTRY, HEADER, HEADER_BYTES, MAGIC, MAX_SECTIONS, VERSION, ModelWriter  # noqa: E402

# the hc8 table's header (tools/convert/hc8.py): magic, version, entries
HC8_HEADER = struct.Struct("<8sII")
HC8_MAGIC, BASALT_HC8_MAGIC, HC8_VERSION = b"GRANITH8", b"BASALTH8", 1


def die(msg: str):
    raise SystemExit(f"pack_to_granite: {msg}")


def new_meta(raw: bytes) -> bytes:
    """`meta` with "format": "granite", written as the converter writes it (key order kept)."""
    meta = json.loads(raw)
    if not isinstance(meta, dict) or meta.get("format") != "basalt":
        die(f'meta: "format" is {meta.get("format") if isinstance(meta, dict) else None!r}, not "basalt"')
    meta["format"] = "granite"
    return json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def new_hc8(raw: bytes) -> bytes:
    """The `hc8` table with granite's magic; its entries and data unchanged."""
    if len(raw) < HC8_HEADER.size:
        die("hc8: shorter than its header")
    magic, version, _ = HC8_HEADER.unpack_from(raw)
    if magic != BASALT_HC8_MAGIC or version != HC8_VERSION:
        die("hc8: not a Basalt hyper-connection table (bad magic or version)")
    return HC8_MAGIC + raw[len(HC8_MAGIC):]


def old_sections(src: Path) -> dict[str, tuple[int, int, bytes]]:
    """OLD's section table, name -> (offset, bytes, SHA-256), once its header says it is a Basalt pack this script reads.
    The container's layout is granite's own (model_file.py); only the magic differs."""
    with open(src, "rb") as f:
        head = f.read(HEADER_BYTES)
    if len(head) < HEADER_BYTES:
        die(f"{src}: shorter than a pack's header")
    magic, version, n, file_bytes, _ = HEADER.unpack_from(head)
    if magic == MAGIC:
        die(f"{src} is already a .granite pack")
    if magic != BASALT_MAGIC:
        die(f"{src}: not a Basalt pack (bad magic)")
    if version != VERSION:
        die(f"{src}: Basalt pack version {version}; this script reads version {VERSION}")
    if n > MAX_SECTIONS:
        die(f"{src}: the section table is too long")
    if src.stat().st_size < file_bytes:
        die(f"{src}: truncated ({src.stat().st_size} of {file_bytes} bytes)")
    sections = {}
    for i in range(n):
        name, off, size, digest, _ = ENTRY.unpack_from(head, HEADER.size + i * ENTRY.size)
        sections[name.rstrip(b"\0").decode()] = (off, size, digest)
    return sections


def convert(src: Path, out: Path) -> None:
    sections = old_sections(src)
    partial = Path(str(out) + ".partial")
    partial.unlink(missing_ok=True)
    w = ModelWriter(str(partial))
    t0 = time.time()
    ok = False
    try:
        with open(src, "rb") as f:
            for name, (off, size, digest) in sections.items():
                if name in ("meta", "hc8"):
                    f.seek(off)
                    raw = f.read(size)
                    if hashlib.sha256(raw).digest() != digest:
                        die(f"section {name}: its bytes do not match the SHA-256 in {src}'s table")
                    w.add_bytes(name, new_meta(raw) if name == "meta" else new_hc8(raw))
                else:
                    w.add_file(name, str(src), off, size)
                    if w.entries[-1][3] != digest:
                        die(f"section {name}: its bytes do not match the SHA-256 in {src}'s table")
                print(f"  {name:<44} {size / (1 << 30):9.3f} GiB", flush=True)
        w.close()
        ok = True
    finally:
        if not ok:
            w.f.close()
            partial.unlink(missing_ok=True)
    os.replace(partial, out)
    print(f"done: {len(w.entries)} sections, {out.stat().st_size / (1 << 30):.3f} GiB, {time.time() - t0:.1f} s -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old", help="the Basalt pack (read only)")
    ap.add_argument("new", help="the .granite pack to write")
    ap.add_argument("--force", action="store_true", help="overwrite NEW")
    a = ap.parse_args()
    src, out = Path(a.old), Path(a.new)
    if not src.is_file():
        die(f"missing input: {src}")
    if out.exists() and out.resolve() == src.resolve():
        die("NEW must be another file than OLD")
    if out.exists() and not a.force:
        die(f"{out} exists (use --force to overwrite)")
    convert(src, out)


if __name__ == "__main__":
    main()
