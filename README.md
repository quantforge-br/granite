# granite

granite is a local inference engine for **Inkling 2 Mini** (Thinking Machines Lab), tuned for one machine: two NVIDIA
RTX 3090 (Ampere, sm_86) with peer-to-peer access over PCIe, on Linux. It is a fork of
[Basalt](https://github.com/jesdga95/basalt) (upstream main `34f4c3bb`, v0.5.6), cut down to one model and one GPU
generation. The server speaks the OpenAI and Anthropic APIs.

Status:

- The build targets sm_86 only, and Inkling 2 Mini runs on one RTX 3090, its least routed experts in RAM when the context
  leaves no room for them.
- The other upstream models are removed.
- Two alike cards run each window tensor parallel over P2P: each holds every dense weight and half of every expert, and
  each writes its partial sums into the other's memory ([GPUs](docs/configuration/gpus.md#two-cards)). The attention is not
  split.

## Hardware target

- 2x RTX 3090 (24 GiB, ~936 GB/s, 82 SMs, 6 MB L2), PCIe Gen4 x8/x8, P2P through a BAR1 P2P driver.
- FP16/BF16 tensor-core math at full rate with FP32 accumulation (gpu-unlock) helps the prompt path; decode is bound by
  memory bandwidth and uses int8 dot products.
- CUDA 13.0 or newer, CMake 3.24+, GCC 13+, Python 3.10+.

## Model and quantization

The default pack is `Inkling-2-Mini-NVFP4-DQ8.granite` (19.7 GB), converted once from upstream's `.basalt` in
[jesdga/Inkling-2-Mini-Basalt](https://huggingface.co/jesdga/Inkling-2-Mini-Basalt) or built from the official release
([guide](docs/models/inkling2.md#pack)):

- Routed experts of layers 0 to 19 in NVFP4, bit for bit the release's own: the model ships its experts in NVFP4 and
  MXFP8, and no BF16 checkpoint exists.
- Experts of layers 20 to 23, the attention projections and the head in Q8_0.
- Mean KLD 0.00289 and top-1 97.76% on one RTX 3090 against the release with its experts dequantized to BF16
  ([precision](docs/precision/inkling2.md)).

On Ampere, NVFP4 decodes like any 4-bit GGUF type (a 16-entry table to int8, then `dp4a` against 8-bit activations), so
it costs the same per byte; it wins on quality because nothing is rounded twice. Requantized GGUFs of the same model
lose more: ggml-org's IQ4_NL (the NVFP4 experts requantized to IQ4_NL) scores 0.0095 KLD and 95.5% top-1 against the
BF16. Smaller GGUF mixes (GSQ-RCO IQ3_XS / IQ4_XS) read fewer bytes per token and are to be measured against the same
gate.

## Quick start

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j            # bin/granite, bin/granite-bench, bin/granite-probe, bin/granite-vision
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
hf download jesdga/Inkling-2-Mini-Basalt Inkling-2-Mini-NVFP4-DQ8.basalt --local-dir models
python3 tools/migrations/pack_to_granite.py models/Inkling-2-Mini-NVFP4-DQ8.basalt models/Inkling-2-Mini-NVFP4-DQ8.granite
bin/granite serve models/Inkling-2-Mini-NVFP4-DQ8.granite --gpu 0
```

The server listens on 127.0.0.1. The model guide: [Inkling 2 Mini](docs/models/inkling2.md). Every setting:
[configuration](docs/configuration/index.md).

A `.granite` pack is upstream Basalt's container under granite's names. The engine refuses a `.basalt` pack;
`tools/migrations/pack_to_granite.py` converts one once, without a download ([model file](docs/model-file.md)).

## Credits

- **Code:** [Basalt](https://github.com/jesdga95/basalt) by Jesús Gulfo (MIT), which granite forks;
  [Strata](https://github.com/Niko1221/Strata), the project Basalt forks (MIT);
  [Infernix](https://github.com/Wallawalla47/Infernix) (Ian Ranson and contributors, Apache-2.0), whose kernels are
  adapted; [ggml and llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT); and ideas from
  [NInfer](https://github.com/Neroued/ninfer). Licences: [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
- **Model:** Inkling 2 Mini by Thinking Machines Lab, under the licence on its card.

## License

MIT, see [LICENSE](LICENSE).
