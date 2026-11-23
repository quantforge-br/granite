# How quality is measured

granite scores a model against a full-precision reference on the same text, through the decode path that generates tokens. The
[precision page](index.md) has the numbers.

## The gate

`granite-bench --quality` runs the gate. With `--logs DIR` it keeps the scored rows in `DIR/rows.tsv`;
`tools/ppl_gate.py --report` summarizes such a file and `tools/ppl_gate.py --diff` compares two row by row.

- **Text:** wikitext-2 in 512-token chunks. The first 256 tokens of each chunk are prefilled.
- **Scored rows:** 256 to 510, through the decode path in verify windows of `--quality-t` tokens (default 3), capped at the
  widest window the model scores: Inkling 2 Mini scores one token a window. The window tokens are the reference text, so the
  drafter's output is not used.
- **Reference:** a llama.cpp KL-divergence base file (`--kl-divergence-base`).
- **KLD** is KL(base || test) over the full vocabulary, from the fp32 log-softmax, summed in double. Base terms at log-probability -16 or
  below are skipped, a tiny underestimate.
- **Reported:** PPL, mean KLD (the tables also give median and p99) and top-1 agreement.
- **Length:** 20 chunks for a quick check. A change to numerics needs 100.
- **Coverage:** the 256-token prefill halves run the prompt path for that length (Inkling 2 Mini's experts through MMQ).
- **Determinism:** two runs of one build match row for row, with the placement pinned (see [greedy identity](#greedy-identity)).

## The BF16 reference

Every score is against the original model in BF16:

| Model | Base | Base PPL |
| --- | --- | --- |
| Inkling 2 Mini | the official release with its experts dequantized to BF16 (no BF16 checkpoint exists; the release ships NVFP4 and MXFP8 experts): a BF16 GGUF from `convert_hf_to_gguf --outtype bf16` (llama.cpp 29b52d398), through `llama-perplexity`, wikitext-2, 100 chunks of 512, `-b 8192 -ub 8192`; `inkling2-bf16-base.kld`. Not comparable with a true BF16 base | 7.0714 (upstream's) |

The upstream bases are published as the [basalt-kld-bases](https://huggingface.co/datasets/jesdga/basalt-kld-bases) dataset on
Hugging Face, with checksums: pass one to `granite-bench --quality --kld-base FILE` to score a build or a self-converted pack.

A quantized base is not a reference. llama.cpp's Q8_0 GGUF adds its own rounding, so every pack reads 0.003 to 0.010 KLD higher against a
Q8_0 base than against BF16.

Numerics changes: report PPL, mean KLD and top-1 next to the previous build. Bitwise-neutral changes: show greedy identity.

## Greedy identity

Pin the placement first, or experts moving between the CPU and the card will round differently: every expert on the card, or a
static spill (a fixed `--expert-cache N`, or a context of at most 32,768; [guide](../models/inkling2.md#memory)).

Drafting on and off, token by token: one engine session per configuration over `--serve`:

```sh
bin/granite --serve --model MODEL.basalt {--spec 0 | --spec 8} --gpu 0 --max-context 8192
```

After `READY`, `GEN 256 temperature=0 ID,ID,...` on stdin returns `T ID` lines and ends with `DONE`; `QUIT` ends the session.
Compare per position.

Gate rows of two builds: run `granite-bench --quality --logs DIR` with each, then
`python3 tools/ppl_gate.py --diff DIR1/rows.tsv DIR2/rows.tsv`.

## Reproduce the reference

A base is the model's BF16 GGUF through `llama-perplexity`, with the llama.cpp commit and flags its row in
[the table](#the-bf16-reference) names:

```sh
llama-perplexity -m MODEL-BF16-00001-of-0000N.gguf -f wikitext-2-raw/wiki.test.raw \
    --kl-divergence-base MODEL-bf16-base.kld -c 512 -b 8192 -ub 8192 --chunks 100 -ngl 99
```

Score a pack against it with the same command and `-m PACK.gguf --kl-divergence` added.
