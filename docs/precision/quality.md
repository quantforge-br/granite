# How quality is measured

Basalt scores a model against a full-precision reference on the same text, through the decode path that generates tokens. Every
model uses this method; each model's [precision page](index.md) has its numbers.

## The gate

`basalt-bench --quality` runs the gate. `tools/ppl_gate.py` runs it offline against a built engine ([commands](#reproduce-the-reference)).

- **Text:** wikitext-2 in 512-token chunks. The first 256 tokens of each chunk are prefilled.
- **Scored rows:** 256 to 510, through the decode path in verify windows of `--quality-t` tokens (default 3). The window tokens are the
  reference text, so the drafter's output is not used.
- **Reference:** a llama.cpp KL-divergence base file (`--kl-divergence-base`).
- **KLD** is KL(base || test) over the full vocabulary, from the fp32 log-softmax, summed in double. Base terms at log-probability -16 or
  below are skipped, a tiny underestimate.
- **Reported:** PPL, mean KLD (the tables also give median and p99) and top-1 agreement.
- **Length:** 20 chunks for a quick check. A change to numerics needs 100.
- **Coverage:** the 256-token prefill halves run each model's prompt path for that length. Inkling 2 Mini's halves run its FP4
  experts.
- **Determinism:** two runs of one build match row for row, with the tiers pinned (see [greedy identity](#greedy-identity)).

## The BF16 reference

Every score is against the original model in BF16:

| Model | Base | Base PPL |
| --- | --- | --- |
| Inkling 2 Mini | the official release with its experts dequantized to BF16 (no BF16 checkpoint exists; the release ships NVFP4 and MXFP8 experts): a BF16 GGUF from `convert_hf_to_gguf --outtype bf16` (llama.cpp 29b52d398), through `llama-perplexity`, wikitext-2, 100 chunks of 512, `-b 8192 -ub 8192`; `inkling2-bf16-base.kld`. Not comparable with the other models' true-BF16 bases | 7.0714 |

The bases are published as the [basalt-kld-bases](https://huggingface.co/datasets/jesdga/basalt-kld-bases) dataset on Hugging Face,
with checksums: pass one to `basalt-bench --quality --kld-base FILE` to score a build or a self-converted pack.

A quantized base is not a reference. llama.cpp's Q8_0 GGUF adds its own rounding, so every pack reads 0.003 to 0.010 KLD higher against a
Q8_0 base than against BF16.

Numerics changes: report PPL, mean KLD and top-1 next to the previous build. Bitwise-neutral changes: show greedy identity.

## Greedy identity

Pin the placement first, or experts moving between the CPU pool and the GPUs will round differently: `--adapt-swaps 0 --peer-adapt-swaps 0`,
a fixed `--expert-cache N`, and a fixed peer tier.

Forced-window identity (the same window width must give the same rows):

```sh
python3 tools/ppl_gate.py --model PACK.basalt --engine-bin bin/basalt --base MODEL-bf16-base.kld \
    --chunks 20 --t {1|3|7} --out DIR --expert-cache N --kv int8 --resident-experts --peer-exclusive --prefill auto
python3 tools/ppl_gate.py --diff DIR1/rows.tsv DIR7/rows.tsv
```

MTP on and off, token by token: one engine session per configuration over `--serve`:

```sh
bin/basalt --serve --model MODEL.basalt {--spec 0 | --spec 7} --expert-cache N --prefill auto --kv int8 \
    --vram-reserve-mib 700 --resident-experts --peer-exclusive --gpu 0,1 --max-context 8192 \
    --adapt-swaps 0 --peer-adapt-swaps 0 --exp concur9_logit_hash=1
```

After `READY`, `GEN 256 temperature=0 ID,ID,...` on stdin returns `T ID` lines and ends with `DONE`; `QUIT` ends the session. `LH` hash lines on stderr. Compare per position, matching rows by position.

## Reproduce the reference

A base is the model's BF16 GGUF through `llama-perplexity`, with the llama.cpp commit and flags its row in
[the table](#the-bf16-reference) names:

```sh
llama-perplexity -m MODEL-BF16-00001-of-0000N.gguf -f wikitext-2-raw/wiki.test.raw \
    --kl-divergence-base MODEL-bf16-base.kld -c 512 -b 8192 -ub 8192 --chunks 100 -ngl 99
```

Score a pack against it with the same command and `-m PACK.gguf --kl-divergence` added.
