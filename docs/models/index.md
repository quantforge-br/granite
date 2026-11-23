# Models

A model is one `.basalt` file, mixture-of-experts or dense. The server reads the architecture from the
file and picks the chat format, tool syntax and thinking switch. Several models can share one address ([router](../configuration/models.md)). What each model supports:

| | DeepSeek-V4.1-Flash | Qwen3.8-27B | Hy4-flash | Inkling 2 Mini |
| --- | --- | --- | --- | --- |
| Architecture id | `deepseek41` | `qwen35` | `hy4flash` | `inkling2` |
| Packs | default `DeepSeek-V4.1-Flash-1017-UD-IQ1_M`, [Hugging Face](https://huggingface.co/jesdga/DeepSeek-V4.1-Flash-1017-Basalt) | `Qwen3.8-27B-QUASAR-NVFP4`, [Hugging Face](https://huggingface.co/jesdga/Qwen3.8-27B-Basalt), built from QUASAR's checkpoint ([model file](../model-file.md#qwen38-27b-pack)) | `Hy4-flash-NVFP4`, [Hugging Face](https://huggingface.co/jesdga/Hy4-flash-Basalt), built from Tencent's BF16 checkpoint ([model file](../model-file.md#hy4-flash-nvfp4-pack)) | `Inkling-2-Mini-NVFP4-DQ8`, [Hugging Face](https://huggingface.co/jesdga/Inkling-2-Mini-Basalt), built from the official release ([guide](inkling2.md#pack)) |
| Layers, experts | 40; 384 per layer, top-6, plus one shared | 64 (48 GDN, 16 gated attention); dense | 42 (12 sparse MLA with an indexer, 30 sliding-window); layer 0 dense, then 512 per layer, top-16, plus one shared | 24 visited 48 times a token (a core of 12 run three times; 4 full attention, 20 sliding-window); 128 per layer, top-8 |
| Context | default 8,192; `--max-context` sets it, 1,024 to 1,048,576 | default 8,192; `--max-context` sets it, up to 262,144, past it with `--rope-scaling yarn` | default 8,192; `--max-context` sets it, up to 1,048,576 | default 8,192; `--max-context` sets it, up to 1,048,576 |
| Drafting | DSpark drafter in the pack, on; `--spec` default 6 | DFlash 2 block drafter in the pack (else the MTP layer), on; `--spec` default 16 | DFly2 block drafter in the pack, on; `--spec` default 6 | DSpark drafter in the pack, on; `--spec` default 8 |
| Chat, tools, thinking | yes; DSML tools | yes; `<tool_call>` tools | yes; `<tool_calls>` tools | yes; JSON tool calls |
| Reasoning effort | `none`, `low` (50), `medium` and `high` (75), `max` (100), or 1 to 100 | `none`, `low`, `medium`, `high`, `max` (`high` = `max`); a number is refused | `none`, `low`, `medium`, `high`, `max` (all but `none` and `max` are `high`); a number is refused | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, or 1 to 100 |
| Images | `--vision dynamic` only | `--vision dynamic` | `--vision dynamic` only | `--vision dynamic` only |
| Video | no | with images, and FFmpeg on the machine ([video](../running.md#video)) | no | no |
| `--kv` | fp8 (default: FP8 window, FP4 compressed rows), fp16 | fp16 (default), int8 | fp16 (default), fp8 | fp16 (default), fp8 |
| Cards | main card plus one expert-tier card | main card only | main card plus up to three expert-tier cards | main card, experts in RAM when the context needs their room; or two cards as a layer split |
| Layer split | no | no | no | yes, matched cards, priced at start |
| Parallel slots | yes; each slot drafts | yes; each slot drafts | not measured | not measured |
| Lossy flags | none | none | none | none |
| Prompt skip | yes (`--no-prompt-skip` turns it off) | no | no | no |
| Idle sleep, wake | yes (Linux); a second-card sleep may be refused | no | yes (Linux) | yes (Linux) |
| `basalt-bench --quality` | yes, one row per window | yes | yes | yes |
| Guide | [guide](deepseek41.md) | [guide](qwen38-27b.md) | [guide](hy4-flash.md) | [guide](inkling2.md) |
| Precision | [precision](../precision/deepseek41.md) | [precision](../precision/qwen38-27b.md) | [precision](../precision/hy4-flash.md) | [precision](../precision/inkling2.md) |
