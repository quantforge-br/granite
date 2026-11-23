# Models

A model is one `.basalt` file. The server reads the architecture from the file and picks the chat format, tool syntax and thinking switch. granite builds one model, Inkling 2 Mini; several of its packs can share one address ([router](../configuration/models.md)). What it supports:

| | Inkling 2 Mini |
| --- | --- |
| Architecture id | `inkling2` |
| Packs | `Inkling-2-Mini-NVFP4-DQ8`, [Hugging Face](https://huggingface.co/jesdga/Inkling-2-Mini-Basalt), built from the official release ([guide](inkling2.md#pack)) |
| Layers, experts | 24 visited 48 times a token (a core of 12 run three times; 4 full attention, 20 sliding-window); 128 per layer, top-8 |
| Context | default 8,192; `--max-context` sets it, up to 1,048,576 |
| Drafting | DSpark drafter in the pack, on; `--spec` default 8 |
| Chat, tools, thinking | yes; JSON tool calls |
| Reasoning effort | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, or 1 to 100 |
| Images | `--vision dynamic` only |
| Video | no |
| `--kv` | fp16 (default), fp8 |
| Cards | one sm_86 card, experts in RAM when the context needs their room; or two alike cards as one tensor-parallel pair ([GPUs](../configuration/gpus.md#two-cards)) |
| Parallel slots | not measured |
| Lossy flags | none |
| Prompt skip | no |
| Idle sleep, wake | yes |
| `granite-bench --quality` | yes |
| Guide | [guide](inkling2.md) |
| Precision | [precision](../precision/inkling2.md) |
