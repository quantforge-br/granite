# `.basalt` model file

One file holds everything the engine and the server load: weights, tokenizer and chat template, DSpark drafter,
vision projector, and the GGUF metadata with default sampling. Flags and the expert profile are
not in it.

- **Layout:** a 64 KiB header (magic `BASALTMF`, version 2, up to 511 named sections with offset, size and SHA-256), then sections aligned
  to 2 MiB. Authoritative: [model_format.hpp](../include/basalt/core/model_format.hpp).
- **Versions:** the engine reads version 2 only. Version 1 files convert with
  [tools/migrations/basalt_v1_to_v2.py](../tools/migrations/README.md), without a download. What changed:
  [file format changes](basalt-file-changelog.md).

## `NVFP4_PLANES` (tensor type 45)

An NVFP4 matrix of `rows` x `K` (rows a multiple of 128, K of 64) as two planes in one tensor, 36 bytes per 64 values like
ggml's NVFP4 blocks (NInfer's and Infernix's `block_scale_k16_m128x4` layout without its FP32 word). The Inkling 2 Mini pack
stores no such tensor; the engine rounds the DSpark drafter's Q8_0 block products to this layout in memory at load:

- **Codes**, `rows * K / 2` bytes from the tensor's start: row-major, byte `b` of a row holds value `2b` in its low nibble and
  `2b + 1` in its high nibble (E2M1, as NVIDIA's checkpoints store them).
- **Scales**, `rows * K / 16` UE4M3 bytes right after the codes, one per 16 values: 512-byte tiles of 128 rows x 4 groups,
  row-block major, so the scale of row `r` and group `g` (`= k / 16`) sits at
  `((r / 128) * (K / 64) + g / 4) * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + g % 4`.
- The matrix's FP32 scale is a separate F32 [1] tensor `<name>.scale`; a value is `e2m1 * ue4m3 * scale`.

[nvfp4_planes.hpp](../include/basalt/kernels/nvfp4_planes.hpp) is the engine's copy of this layout.

## Inkling 2 Mini pack

The pack (`Inkling-2-Mini-NVFP4-DQ8.basalt`, 19.7 GB) comes from Thinking Machines Lab's official release alone
([thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Apache-2.0): both safetensors files
(`model` and the DSpark drafter `mtp`), `config.json`, `hf_quant_config.json`, the tokenizer files and the chat template. No other
input. [checkpoint_inkling2.py](../tools/convert/checkpoint_inkling2.py) is what `convert.py --checkpoint` runs on a checkpoint of
model type `inkling2_loop`; the drafter is in the release.

```sh
hf download thinkingmachines/Inkling-2-Mini --local-dir Inkling-2-Mini
python3 tools/convert/convert.py --checkpoint Inkling-2-Mini --out Inkling-2-Mini-NVFP4-DQ8.basalt --name "Inkling 2 Mini"
```

- `experts` and `experts.layout`: the routed experts of each of the 24 layers, 128 to a layer, one blob per expert in the order
  `[gate | up | down]` (the checkpoint's `w13` rows alternate gate and up; they are separated). `experts.layout` is a text file with
  one `layer gu_type d_type offset blob_bytes` line per layer after a header line.
  - Layers 0 to 19 (the release's NVFP4): NVFP4 (type 40) blobs of 4,478,992 bytes. The E2M1 codes and the E4M3 scale of every 16
    values move bit for bit, the nibbles regrouped into 64-value blocks; the blob ends in an FP32 tail `{s_gate, s_up, s_down, 0}`
    with `s = amax / outer_scale_denominator` of the expert's matrices (gate and up share one `s`, from `w13`).
  - Layers 20 to 23 (the release's MXFP8): Q8_0 blobs of 8,460,288 bytes, no tail. Each value is its E4M3 code times
    2^(UE8M0 - 127) in ggml's reference Q8_0 rounding; one Q8_0 block is one MXFP8 block, so the 36 blocks of a down row and the 72
    of a gate or up row are the release's own. Rounding error 0.54% RMS of the release's values
    ([precision](precision/inkling2.md)).
  - The engine reads a row of an expert 256 values at a time, so it allocates each layer's experts with `mmq::read_past_bytes`
    extra zeroed bytes after the last blob; the file holds none.
- `gguf`: every other text tensor in llama.cpp-style names. The attention projections (q, k, v, relative-bias, gate, sink,
  output) and the k and v short convolutions' weight projections Q8_0 (ggml's reference rounding of the BF16 values); the token embedding and the output head BF16 as stored, their
  padded rows past the real vocabulary (200,058 of 201,024) dropped, the engine rounding the head to Q8_0 as it loads; the loop
  controllers BF16 as stored; norms, the block-residual and message queries, the router and its bias, the short-convolution taps,
  the relative-bias tables and the clip scales F32 (exact). The drafter's tensors are in it as `dspark.*`, BF16 and F32 as stored.
- `vision/mmproj`: the vision tower as llama.cpp's clip GGUF (projector `inkling2v`), BF16 and F32 as stored.
- `tokenizer/*` and `meta`: the release's tokenizer files and chat template, its geometry (24 layers, the prelude, core and coda
  ranges, the loop's passes, the experts) and its license; the pretokenizer is the checkpoint's o200k split.
- 18.38 GiB: experts 14.71, `gguf` 3.03 (drafter included), vision 0.60. Every rounding is deterministic and runs once, in pieces by
  worker processes. How each tensor rounds and what it costs in quality: [precision](precision/inkling2.md).
