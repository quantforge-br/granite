# Example configurations

Start with `bin/granite serve --config config.json`. Model paths are relative to the file. Settings: [configuration](index.md).

## One card

Every expert sits on the card while the context leaves room; past that the least routed go to page-locked RAM and the CPU
computes them ([guide](../models/inkling2.md#memory)).

```json
{
  "version": 2,
  "models": [{ "model": "Inkling-2-Mini-NVFP4-DQ8.basalt", "gpu": "0", "max-context": 65536, "vision": "dynamic" }],
  "server": { "host": "127.0.0.1", "port": 8080 }
}
```

## Long context on one card

Like the first, with `"kv": "fp8"` (half the full-attention caches) and `"max-context": 1048576`.

## Two cards

Like the first, with `"gpu": "0,1"` and `"max-context": 262144`: both cards run every token, each with half of every expert
([GPUs](gpus.md#two-cards)). The context must fit on both cards; on two 24 GiB cards with images on, 262,144 positions fit.

Several models: [models and the router](models.md).
