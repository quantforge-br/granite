# GPUs

granite's kernels are built for sm_86 (RTX 30-series Ampere) only. Inkling 2 Mini runs on one card, or on two alike cards
with peer access as one tensor-parallel pair, half of every expert on each
([Inkling 2 Mini](../models/inkling2.md#two-cards)).

One setting picks the cards: `gpu`. It means the same in the server (`--gpu`, or `"gpu"` in a model entry or the top-level
`engine` of the config), in `granite` itself and in `granite-bench`.

| Value | Cards used | Main card |
| --- | --- | --- |
| not set | GPU 0 alone; one warning at start names the other visible sm_86 cards | GPU 0 |
| `"0,1"`, `"1,0"`, ... | exactly these, in this order | the first one listed |
| `"auto"` | every visible sm_86 card; any other card is skipped with one line naming it | the one with the most memory bandwidth, then the most VRAM |

Cards are numbered as `nvidia-smi` numbers them. When `CUDA_VISIBLE_DEVICES` is set, the numbers count within its list
(`CUDA_VISIBLE_DEVICES=1,0` makes `gpu 0` the card `nvidia-smi` calls 1), and its numbers are read as `nvidia-smi`'s
too. A number past the visible cards, a card listed twice, or a card that is not sm_86 stops the start with an error.
With one card (unset, or one number) no other card is visible to the process.

## Two cards

The start prints a `gpus:` line with the cards, then Inkling 2 Mini's own line says what the second card does.

- **Tensor parallel:** `tensor parallelism on CUDA0 and CUDA1: each holds every dense weight and half of every expert`. Both
  cards run every layer on the same rows. In a decode window each runs the attention of half of the heads with the output
  projection over their columns, then its half of every routed expert; after each, it writes its sum into the other's
  memory, and both add the two sums in the same order, so both hold the same bits. Prompt chunks run the whole attention
  on both cards. Each card runs half of the head's rows
  and the second copies its logits into the first card's, which alone runs the sampler, the drafter and the image encoder.
- **Runs alone:** `CUDA1 idles (...): CUDA0 runs alone`, with the reason: no peer access both ways, SM counts or shared
  memory per block that differ, or `--expert-cache` given (a one-card diagnostic).
- **Context:** it must fit beside the weights on both cards. Two cards keep no experts in RAM; the start says how much is
  missing when the context does not fit.
- **More cards:** Inkling 2 Mini uses at most two. A third card listed stays idle.
