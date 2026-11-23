# Models

A model is one `.basalt` file. The server reads the architecture from the file and picks the chat format, tool syntax and thinking switch. Several models can share one address ([router](../configuration/models.md)). What each model supports:

| | Hy4-flash | Inkling 2 Mini |
| --- | --- | --- |
| Architecture id | `hy4flash` | `inkling2` |
| Packs | `Hy4-flash-NVFP4`, [Hugging Face](https://huggingface.co/jesdga/Hy4-flash-Basalt), built from Tencent's BF16 checkpoint ([model file](../model-file.md#hy4-flash-nvfp4-pack)) | `Inkling-2-Mini-NVFP4-DQ8`, [Hugging Face](https://huggingface.co/jesdga/Inkling-2-Mini-Basalt), built from the official release ([guide](inkling2.md#pack)) |
| Layers, experts | 42 (12 sparse MLA with an indexer, 30 sliding-window); layer 0 dense, then 512 per layer, top-16, plus one shared | 24 visited 48 times a token (a core of 12 run three times; 4 full attention, 20 sliding-window); 128 per layer, top-8 |
| Context | default 8,192; `--max-context` sets it, up to 1,048,576 | default 8,192; `--max-context` sets it, up to 1,048,576 |
| Drafting | DFly2 block drafter in the pack, on; `--spec` default 6 | DSpark drafter in the pack, on; `--spec` default 8 |
| Chat, tools, thinking | yes; `<tool_calls>` tools | yes; JSON tool calls |
| Reasoning effort | `none`, `low`, `medium`, `high`, `max` (all but `none` and `max` are `high`); a number is refused | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, or 1 to 100 |
| Images | `--vision dynamic` only | `--vision dynamic` only |
| Video | no | no |
| `--kv` | fp16 (default), fp8 | fp16 (default), fp8 |
| Cards | main card plus up to three expert-tier cards | main card, experts in RAM when the context needs their room; or two cards as a layer split |
| Layer split | no | yes, matched cards, priced at start |
| Parallel slots | not measured | not measured |
| Lossy flags | none | none |
| Prompt skip | no | no |
| Idle sleep, wake | yes (Linux) | yes (Linux) |
| `basalt-bench --quality` | yes | yes |
| Guide | [guide](hy4-flash.md) | [guide](inkling2.md) |
| Precision | [precision](../precision/hy4-flash.md) | [precision](../precision/inkling2.md) |
