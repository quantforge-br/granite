# Diagnostics and measurement flags

Flags for measuring, debugging and testing the engine. All are off by default and are not settings.
Binary: **CLI** = `bin/granite` one-shot (`--tokens IDS`), **serve** = `--serve` engine, **bench** = `granite-bench`.

## One-shot generation

The one-shot run decodes greedily and prints one `T <id>` line per token.

| Flag | Binary | Default | What it does |
|---|---|---|---|
| `--tokens`, `--tokens-file` | CLI | required | The prompt as token ids. |
| `--max-new` | CLI | 16 | Tokens to generate. |
| `--stop-eos` | CLI | off | Stop at an end-of-turn token. |
| `--eos-ids` | engine | the pack's | The end-of-turn tokens; implies `--stop-eos`. |
| `--image FILE`, `--image-at N` | CLI | | A picture encoded before prompt token N. Each `--image` needs an `--image-at`; the positions rise. |
| `--gpu-plan` | engine | | Print the `--gpu` choice as JSON and exit. |

## Logs and checks

| Flag | Binary | What it does |
|---|---|---|
| `--draft-log FILE` | CLI, serve | One line per decode window, for draft-policy study. |
| `--diag-dump DIR` | CLI, serve | The drafter writes its logits for its first 24 blocks to `DIR/dspark.bin`, and the log says how many screened picks differ from the exact head's. |
| `--diag-serial-prompt` | CLI | Read the prompt one token per step, through the decode path. |
| `--diag-logits FILE` | CLI | Write each decoded row's logits as f32. |
| `--diag-batch-check` | CLI | With `--batch N` (2 or more) and a prompt: every session's windows alone against the same windows batched, bitwise ([batched decoding](../design/batching.md)). |
| `--diag-tp-check` | CLI, serve | On two cards (tensor parallelism): after each prompt and decode call and each batch window, every cache and ring of the second card against the first's, bitwise; a difference fails the call. |
| `--expert-cache N` | engine | At most N experts on the card, the rest in RAM, computed by the CPU. One card only: with it a second card idles. |

## Quality gate and bench

| Flag or key | Binary | Default | What it does |
|---|---|---|---|
| `--quality` | bench | 20 chunks | Run the PPL/KLD/top-1 gate through decode. [method](../precision/quality.md) |
| `--kld-base FILE` | bench | | The llama.cpp KL-divergence base. |
| `--quality-t N` | bench | 3 | Tokens per verify window in the gate. |
| `--ppl-base`, `--ppl-out` | serve | | Base file for `ppl=` requests, and where they write rows. |
| `ppl=K`, `ppl_t=T` | request keys | | Score chunk K; window width T. |

| Bench flag | Default | What it does |
|---|---|---|
| `-t conv,struct,prose` | | Decode fixtures, run in that order. |
| `-p DEPTHS` | `-p 16384` with no `-t` or `-p` | Prefill depths, e.g. `32k`. |
| `-r N` | 1 | Repetitions; above 1 the table shows mean and deviation. |
| `--seed N` | | Seed every decode request. |
| `--set "FLAG VALUE"` | | Replace or add engine arguments (repeatable). |
| `--engine PATH`, `--fixtures DIR` | `granite` beside the bench, `data/bench` | Binary and fixtures. |
| `-o md` or `json` | md | Output format. |
| `--power-label W` | read from the GPU | The power limit to print. |
| `--logs DIR`, `--no-windows`, `--print-command` | | Logs, window stats, command printing. |
| `--hardware` | | Measure CPU, RAM and each GPU's link, then exit. |
| `--community`, `--submitter NAME`, `--community-file PATH` | | The community preset and its row, which names the model ([community runs](../community.md)). |

`granite-device` (in the build folder): `--list-devices`, `--link-rates`, `--selftest`.
