"""Engine settings, tokenizer and vision inputs for a .granite model file or a GGUF. Nothing is unpacked: the tokenizer is
read from the pack's sections (or built from the GGUF's metadata) into memory, and the vision programs open the projector
inside the file themselves."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import tokenizer  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402
from model_file import ModelFile  # noqa: E402
from serve.models import dialect_for  # noqa: E402
from serve.engine.flags import flag_value  # noqa: E402

VISION_EXE = "granite-vision"
# llama.cpp's sampling keys in the GGUF metadata -> the request field each one defaults
GGUF_SAMPLING = {"general.sampling.temp": "temperature", "general.sampling.top_p": "top_p",
                 "general.sampling.top_k": "top_k", "general.sampling.min_p": "min_p",
                 "general.sampling.penalty_repeat": "repetition_penalty",
                 "general.sampling.penalty_last_n": "penalty_last_n"}


def cache_root() -> Path:
    return Path.home() / ".cache" / "granite"


def cache_dir(key: str) -> Path:
    """One cache directory per model, named by `key` (ModelInputs.key)."""
    return cache_root() / key


def with_conversation_dir(words: list[str], cdir: Path) -> list[str]:
    """`words` plus the default conversation directory (inside the model cache folder `cdir`) when the disk budget
    is above zero and no directory is named."""
    budget = flag_value(words, "--conversation-cache-disk-mib")
    if budget and budget.isdigit() and int(budget) > 0 and not flag_value(words, "--conversation-cache-dir"):
        return words + ["--conversation-cache-dir", str(cdir / "conversations")]
    return words


def _short(v):
    """A GGUF float32 as the shortest decimal that is the same float32 (0.949999988079071 -> 0.95)."""
    if not isinstance(v, float):
        return v
    bits = struct.pack("<f", v)
    for digits in range(1, 10):
        s = float(f"{v:.{digits}g}")
        if struct.pack("<f", s) == bits:
            return s
    return v


def sampling_defaults(meta: dict) -> dict:
    """The model's sampling defaults: the general.sampling.* keys of meta.gguf_kv, by request field."""
    kv = meta.get("gguf_kv") or {}
    return {field: _short(kv[key]) for key, field in GGUF_SAMPLING.items() if kv.get(key) is not None}


TOKENIZER_SECTIONS = ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json", "chat_template.jinja")


def tokenizer_files(mf: ModelFile) -> dict[str, bytes]:
    """The `tokenizer/*` sections the server reads, by file name (those a pack lacks are absent)."""
    if "tokenizer/vocab.json" not in mf.sections:
        raise ValueError(f"{mf.path}: no tokenizer sections")
    return {n: mf.read("tokenizer/" + n) for n in TOKENIZER_SECTIONS if "tokenizer/" + n in mf.sections}


@dataclass(frozen=True)
class ModelInputs:
    """What the server reads from a model file: its meta (name and gguf_kv, as a pack's meta section carries them), the
    name of its cache folder (a hash of what identifies the model), its tokenizer files and whether it has the vision
    projector."""
    meta: dict
    key: str
    tokenizer: dict[str, bytes]
    projector: bool


def is_gguf(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(4) == b"GGUF"


def read_inputs(path: str) -> ModelInputs:
    """A .granite's inputs from its sections; a GGUF's from its metadata, keyed by the hash of its header (the metadata and
    the tensor directory). ValueError when the file is neither, or carries no tokenizer."""
    if not is_gguf(path):
        mf = ModelFile(path)
        return ModelInputs(json.loads(mf.read("meta")), mf.sections["meta"][2].hex()[:16], tokenizer_files(mf),
                           "vision/mmproj" in mf.sections)
    try:
        g = GGUFFile(Path(path))
        with open(path, "rb") as f:
            key = hashlib.sha256(f.read(g.data_start)).hexdigest()[:16]
        md = g.metadata
        files = tokenizer.metadata_files(md)
    except (ValueError, KeyError, struct.error, UnicodeDecodeError) as e:
        raise ValueError(f"{path}: {e}") from e
    return ModelInputs({"name": md.get("general.name"), "gguf_kv": md}, key, files, False)


def vision_config(path: str, projector: bool, engine_bin: str, sidecar: bool = True) -> dict:
    """The vision sidecar's config, which names the model file (the sidecar opens `vision/mmproj` inside it);
    ValueError names what is missing."""
    engine = Path(engine_bin).resolve()
    side = engine.parent / (VISION_EXE + engine.suffix)   # with the engine's suffix: granite-vision.exe on Windows
    if not projector:
        raise ValueError("this model file has no vision projector (vision/mmproj); drop --vision")
    if sidecar and not side.is_file():
        raise ValueError(f"{side} not found (build the vision sidecar), or drop --vision")
    return {"vision_bin": str(side), "model": os.path.abspath(path)}


@dataclass(frozen=True)
class ModelSpec:
    """One model the server runs. `engine_words` are its engine flags as given. `vision`: a GPU number as a string, "cpu"
    or "dynamic". `gpu`: as --gpu."""
    path: str
    alias: str | None = None
    gpu: object = None
    vision: bool | str = False
    engine_words: tuple[str, ...] = ()


def model_config(spec: ModelSpec, engine_bin: str) -> dict:
    """The engine settings a model implies; its `engine_words` follow its arguments as given (the engine's parser lets
    a later flag win), its `vision` adds the sidecar or, "dynamic", the in-engine encoder (ValueError if it cannot be
    set up)."""
    path, vision = spec.path, spec.vision
    inputs = read_inputs(path)
    meta = inputs.meta
    path = os.path.abspath(path)
    cdir = cache_dir(inputs.key)
    words = with_conversation_dir(list(spec.engine_words), cdir)
    args = ["--model", path]   # settings come from the engine defaults and the server
    args += words
    cdir.mkdir(parents=True, exist_ok=True)
    cfg = {"engine_bin": os.path.abspath(engine_bin), "args": args, "model_name": spec.alias or meta.get("name") or "granite",
           "sampling": sampling_defaults(meta),
           "arch": (meta.get("gguf_kv") or {}).get("general.architecture"), "log": str(cdir / "engine.log"),
           "tokenizer": inputs.tokenizer}
    if vision and dialect_for(cfg["arch"]).in_engine_vision:
        if vision != "dynamic":
            raise ValueError(f"this model's images are encoded in the engine: --vision dynamic, not {vision}")
        if not inputs.projector:
            raise ValueError("this model file has no vision projector (vision/mmproj); drop --vision")
        args.append("--vision")
        cfg["vision"] = {"mode": "dynamic", "gpu": True}
        return cfg
    v = vision_config(path, inputs.projector, engine_bin, sidecar=vision != "dynamic") if vision else None
    if v:
        v["mode"] = vision
        v["gpu"] = vision != "cpu"
        cfg["vision"] = v
        # dynamic: the engine encodes, opening the projector inside the model file itself; else the sidecar's
        # embeddings come in with each image
        if vision == "dynamic" and "--vision" not in args:
            args.append("--vision")
        elif vision != "dynamic" and "--vision-external" not in args:
            args.append("--vision-external")
    return cfg
