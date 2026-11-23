# Configuration reference

Every setting: server options, engine flags, and the sampling fields. Effects on speed and precision: [running](../running.md),
[precision](../precision/index.md).

- [How settings are given](#how-settings-are-given) · [Automatic settings](#automatic-settings) · [Server](#server) · [Model and GPUs](#model-and-gpus)
- [Context and KV](#context-and-kv-cache) · [Experts and memory](#experts-and-memory) · [Drafting](#drafting)
- [Slots](#parallel-slots) · [Conversation cache](#conversation-cache) · [Vision](#vision)
- Pages: [models and the router](models.md), [sampling](sampling.md), [examples](examples.md), [diagnostics](diagnostics.md)

## How settings are given

```sh
bin/granite serve --config config.json [options]
bin/granite serve MODEL.granite [--python PY] [options]
```

- **Server options** (the Python server): model, address, port, slots, vision, logging. `--name value` on the command line.
- **Engine flags** (`bin/granite --help`): context, KV, experts, speculation. Given like any flag; the server passes them
  on in order and asks the engine which it accepts.

A config file is one JSON object. `models` lists the models; `engine` applies to every model that takes it; `server` is shared:

```json
{ "version": 2, "models": [{ "model": "model.granite", "kv": "fp8" }],
  "engine": { "max-context": 65536 }, "server": { "port": 8080 } }
```

- Where a setting goes in a config file: `server` is the `server` block, `entry` a model entry or the top-level `engine`,
  `engine` (an engine flag) the same places, `model entry` a model entry only.
- Keys are flag names without dashes. Switches are `true`. A repeatable option (`allowed-host`, `cors-origin`, `image-url-host`,
  `image-dir`) takes a list. `model` paths are relative to the file.
- Precedence: command line, then config file, then defaults; `--model` and `--alias` do not go with `--config` (its entries name the models).
  A wrong block or an unknown name stops the start. `api-key` cannot be in a file.
- The defaults are the fastest settings that need no knowledge of the machine. The keys that change output or memory (8,192 context,
  fp16 KV) are kept for you to set.

## Automatic settings

When left out, the engine sizes the CPU pool that computes the experts in RAM. A given value always wins. The pool starts only
when some experts are in RAM; the engine log then says `experts in RAM run on the CPU: N pool workers`.

| Setting | Rule |
| --- | --- |
| `pool-affinity` | `auto` |
| `pool-workers` | physical cores minus the host thread; on hybrid CPUs, P-cores minus one plus half the E-cores; lowered to fit a cgroup CPU quota |

## Server

Command line `--name`; the Where column says which config block takes it.

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `model` | model entry | path | required | The `.granite` file: weights, tokenizer, chat template, drafter, vision projector and default sampling. Or a GGUF ([GGUF](../models/inkling2.md#gguf)). |
| `alias` | model entry | string | name in the file | The model name the API reports. |
| `host` | server | address | `127.0.0.1` | Listen address. `0.0.0.0` serves the network; never without `api-key`. |
| `port` | server | integer | `8080` | TCP port. |
| `api-key` | command line | string | none | Require this key on `/v1/*` (`Authorization: Bearer` or `x-api-key`). Command line only. |
| `api-key-file` | command line | path | none | Read that key from the first line of this file, which keeps it out of the process list. Not with `api-key`. Command line only. |
| `allowed-host` | server | string, repeatable | none | Also answer to this `Host` name; `"*"` turns the check off. |
| `cors-origin` | server | string, repeatable | none | Let web pages from this origin call `/v1/*`. |
| `fit-max-tokens` | entry | switch | off | Clamp `max_tokens` to the remaining context instead of a 400. |
| `open` | server | switch | off | Open the web page when ready. |
| `lazy` | server | switch | off | Start unloaded; load on the first request. |
| `idle-unload` | entry | seconds | off | Unload after this long idle. Frees VRAM for other programs. |
| `idle-sleep` | entry | seconds | off | Sleep after this long idle: VRAM freed, RAM kept, wake in seconds. [running](../running.md#sleep---idle-sleep) |
| `min-free-vram-mib` | entry | integer | off | Load an unloaded model only when this much VRAM is free; else 503. |
| `grammar` | entry | switch (`--no-grammar`) | on | Hold tool calls and JSON formats (`json_object`, `json_schema`) to their schema inside the engine. Without it, other JSON formats are refused with 400. |
| `max-payload-mib` | server | MiB | `256` | Largest request body the server reads; a bigger one gets a 413. |
| `debug` | server | switch | off | One log line per request, and the raw model text. |
| `request-lines` | server | switch | off | Echo the per-request summary on stdout. |
| `temperature`, `top-p`, `top-k`, `min-p`, `presence-penalty`, `frequency-penalty`, `repetition-penalty`, `penalty-last-n`, `seed` | entry | number (`top-k`, `penalty-last-n`, `seed`: integer) | the model file's `sampling`, else the engine's | Default for a request that leaves the field out. See [sampling](sampling.md). |
| `reasoning-effort` | entry | `none`, `low`, `medium`, `high`, `xhigh` or `max` | unset (the template's default) | Reasoning effort for requests that name no effort, thinking switch or budget. See [sampling](sampling.md). |
| `reasoning-budget-tokens` | entry | integer | `0` | Thinking budget for requests that name none. A request's own value wins. |
| `thinking-loop-tokens` | entry | integer | `256` | End thinking when one token repeats this many times in a row; `0` turns it off. |
| `answer-room` | entry | `auto` or integer | `auto` | Output tokens kept for closing the thinking and the answer: `auto`, or a fixed number (`0`: none). See [sampling](sampling.md#thinking). |
| `config` | command line | path | none | The config file. Command line only. |

## Model and GPUs

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `gpu` | entry, `granite`, `granite-bench` | `"0"`, `"0,1"`, `"auto"` | GPU 0 alone | Cards, numbered as `nvidia-smi`. The first is the main card. Every card must be sm_86. `auto` takes every sm_86 card and makes the one with the most memory bandwidth main. Two cards run tensor parallel. [GPUs](gpus.md) |

## Context and KV cache

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `max-context` | engine | tokens | 8192 | Longest context, from 1,024 up to the pack's context length. Shared by `parallel` slots. The KV cache grows with it. |
| `kv` | engine | `fp16`, `fp8` | `fp16` | Storage of the full-attention caches. `fp8` halves them (E4M3, unit scale; [guide](../models/inkling2.md#memory)). |

## Experts and memory

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `expert-cache` | engine | `auto` or a count | `auto` | `auto`: every expert on the card when it fits beside the context, else the least routed go to page-locked RAM, just enough, and the CPU computes them ([guide](../models/inkling2.md#memory)). A count keeps at most N on the card: a diagnostic for one card (a second card then idles). |
| `expert-profile` | engine | file | `data/profiles/inkling2-24x128.prof` | The routing counts that pick which experts go to RAM first. |
| `vram-reserve-mib` | engine | MiB | 0 | The free VRAM that `POST /v1/vram` with `"reserve_mib": null` goes back to ([lending VRAM](../running.md#lending-vram-post-v1vram)). The start's sizing keeps its own margin and does not read it. |
| `ram-reserve-mib` | engine | MiB | 2048 | RAM left free for the rest of the system: while serving, the RAM guard holds it. `0` turns the guard off. [memory floors](../running.md#memory-floors) |

## Drafting

Output is exactly plain decoding's.

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `spec` | engine | integer | 8 with the pack's DSpark drafter, else 0 | Verify windows of at most N tokens: the last accepted token plus up to N-1 drafts. `0` or `1` turns it off. [guide](../models/inkling2.md#drafting) |

## Parallel slots

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `parallel` | entry | 1..8 | 1 | Serve N requests at once. Sets the engine's `batch`. |
| `slot-full-context` | entry | switch | off | Give every slot the whole `max-context`, at N times the KV. |

## Conversation cache

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `prompt-cache` | engine | integer | 6 | Conversation checkpoints kept in RAM. `0` reads every prompt from the start. |
| `prompt-cache-root` | engine | tokens | 2048 (`0` = never) | Checkpoint the end of the system prompt when it is at least this long. |
| `turn-token` | engine | token id | the pack's | The token that opens a chat turn; checkpoints go at the last one. |
| `tail-role-token` | engine | token id | -1 (off) | A short trailing turn that opens with this token keeps the checkpoint in front of it. |
| `conversation-cache-mib` | engine | MiB | 0 (off) | RAM budget for parked conversations. |
| `conversation-cache-slots` | engine | integer | 4 | Most parked conversations. |
| `conversation-cache-min-free-mib` | engine | MiB | 2560 | Physical RAM kept free when parking. |
| `conversation-cache-min-tokens` | engine | tokens | 0 | Park only conversations of at least this many tokens. |
| `conversation-cache-disk-mib` | engine | MiB | 0 (off) | Disk budget for parked conversations. [running](../running.md#conversations-on-disk---conversation-cache-disk-mib) |
| `conversation-cache-dir` | engine | directory | model cache folder | Another folder for the disk tier. |

## Vision

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `vision` | entry | `off`, `dynamic` | `off` | Image input, encoded inside the engine on the main card. Needs a model file with a vision projector. 32 MiB and 64 megapixels per image. [running](../running.md#vision---vision) |
| `image-url-host` | server | `HOST` or `HOST:PORT`, repeatable | none | Allow `http(s)` image URLs on this host from other machines. [image sources](image-sources.md) |
| `image-dir` | server | directory, repeatable | none | Allow image files under this directory from other machines. [image sources](image-sources.md) |
| `image-remote-only` | server | switch | off | Treat this machine's requests as remote too. [image sources](image-sources.md) |

Diagnostic flags: [diagnostics](diagnostics.md).
