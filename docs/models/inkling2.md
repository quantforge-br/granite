# Inkling 2 Mini

Thinking Machines Lab's Inkling 2 Mini (Apache-2.0), the model granite runs: `granite serve PACK`. What granite supports:
[models](index.md). Rounding and quality: [precision](../precision/inkling2.md).

Numbers marked upstream are Basalt's own, measured on Blackwell cards (RTX 5090, RTX 5060 Ti) before the fork, where the
prompt path ran the NVFP4 experts on FP4 tensor cores; granite runs them through MMQ ([long prompts](#long-prompts)).

## Pack

One file, `Inkling-2-Mini-NVFP4-DQ8.granite` (19.7 GB), built from the official release with its vision encoder and DSpark
drafter:

- Routed experts, layers 0 to 19: the release's NVFP4 codes and scales, bit for bit (14.7 GiB for all 24 layers).
- Routed experts, layers 20 to 23: the release's MXFP8 values as Q8_0.
- Attention, gate, short-convolution and output projections: Q8_0 of the BF16 values.
- Output head: BF16 in the file; the engine rounds it to Q8_0 as it loads. Token embedding BF16. Norms, router, biases and the
  loop's small tensors F32; loop controllers BF16.
- Vision encoder and projector (BF16, 0.6 GB) and the DSpark drafter (BF16, 0.6 GB), whole.

Build it from the release directory: `tools/convert/convert.py --checkpoint DIR` reads both safetensors files, the quant config,
the tokenizer and the template.

Upstream publishes the pack as `Inkling-2-Mini-NVFP4-DQ8.basalt` in
[jesdga/Inkling-2-Mini-Basalt](https://huggingface.co/jesdga/Inkling-2-Mini-Basalt) (19,734,300,551 bytes, SHA-256
`f7b4e5f8e17681c4735a7dfc51ffd8e805c44184fad4e1778b13187d79dc3787`). granite refuses a `.basalt` file: convert it once,
which copies every section, re-quantizes nothing and needs room for a second copy
([migration](../../tools/migrations/README.md)):

```sh
hf download jesdga/Inkling-2-Mini-Basalt Inkling-2-Mini-NVFP4-DQ8.basalt --local-dir models
python3 tools/migrations/pack_to_granite.py models/Inkling-2-Mini-NVFP4-DQ8.basalt models/Inkling-2-Mini-NVFP4-DQ8.granite
```

## GGUF

The engine and the server also open llama.cpp's `inkling2_loop` GGUF of one file as it is, with no conversion
(`granite serve --model FILE.gguf`); a split GGUF is joined first with `llama-gguf-split --merge`. One card, two cards and
experts in RAM run it as they run the pack. What loads:

- Routed experts: llama.cpp's fused gate and up tensor and its down tensor, in a type both the decode and the prompt kernels
  run; NVFP4 with its per-expert scales.
- Projections and the head: Q8_0 as stored, or BF16 rounded to Q8_0 as it loads (the pack's rounding). Token embedding
  Q8_0 or BF16. Loop controllers stored as F32 load as BF16, the values they hold.
- The tokenizer and chat template come from the file's metadata, the sampling defaults from its `general.sampling.*` keys.

Anything else is refused at load, naming the tensor or layer and the type it found. The file carries no vision encoder
and no DSpark drafter: `--vision` needs `--mmproj FILE`, and drafting is off unless `--drafter FILE` gives one
([drafting](#drafting)). Unsloth's `mmproj-BF16.gguf` holds the pack's projector tensors byte for byte, so it encodes a
picture to the pack's rows (`inkling2_vision_parity` on either).

Unsloth's `Inkling-2-Mini-UD-Q4_K_XL.gguf` (17.6 GB, SHA-256
`e64276656f072e9b2197a61723667cbdb71da66a716ffd966e572c21088f3cd3`) loads to the pack's bytes on the card except the token
embedding, Q8_0 of the pack's BF16 values (KLD 0.00309 against 0.00289, [precision](../precision/inkling2.md)).
`inkling2_gguf_parity` checks it tensor by tensor when both files are in `models/`.
From start to the first token on one RTX 3090 at 270 W it takes 9.4 s, the pack 9.3 s, or 8.7 s with the pack's converted
tensors still in the page cache.

## Architecture

The `inkling2` architecture (code in `src/models/inkling2`): 24 physical layers, 2,304 wide, no rotary positions anywhere in
the text model.

- **Loop:** a prelude (layers 0 to 5), a core (6 to 17) run three times, and a coda (18 to 23): 48 layer visits a token, each
  with its own cache and short-convolution state. Passes 2 and 3 start again from the prelude's output and mix in messages from
  the pass before through learned controllers.
- **Attention residuals:** a layer's input is a softmax mix over the finished blocks of 6 layers and the current one, not a
  plain residual stream.
- **Attention:** 24 query heads, 8 key/value heads of 128. Layers 5, 11, 17 and 23 attend every position (8 visits a token);
  the other 20 attend a window of 1,024. Keys and values pass a causal short convolution; a learned relative bias over the last
  1,024 positions, a sink per head and an output gate.
- **MoE:** 128 routed experts per layer, top 8 by sigmoid, 1,152 wide, with a softcapped SwiGLU; no shared expert.
- **Vision:** a 23-layer ViT reads a picture at its own size, padded to multiples of 40 pixels; one token per 40 x 40 cell.

## Memory

- **One card.** The main card holds the weights: 1.10 GiB dense, 14.71 GiB experts, the drafter 0.48 GiB. The token
  embedding (0.86 GiB) stays in page-locked RAM and the card gathers its rows. Two cards:
  [half of every expert on each](#two-cards).
- **Context:** `--max-context` from 1,024 to 1,048,576, default 8,192.
- **Per position:** the 8 full-attention visits keep every position, 32 KiB with `--kv fp16`, 16 KiB with `--kv fp8`, plus the
  drafter's full layer; the window layers' rings do not grow.
- **Experts in RAM when the context needs the room:** when the asked context does not fit beside every expert, the least
  routed experts (by the shipped profile `data/profiles/inkling2-24x128.prof`) are planned into page-locked RAM, just enough,
  and decode computes them on the CPU: after a layer's router its rows go to RAM, the CPU pool (`--pool-workers`) runs the
  pairs whose expert is in RAM while the card runs the rest, and the card takes their rows before it combines them. The
  full visits' caches then back their first 32,768 positions and grow as prompts
  need them, and the experts planned into RAM are copied back to the card while the caches leave room, the least routed
  giving way first when they grow. A short chat at `--max-context 1048576 --kv fp8` keeps every expert on an RTX 5090
  (upstream). With a context of at most 32,768, or with `--expert-cache`, the spill is static: what does not fit stays in
  RAM. Nothing changes when everything fits.
- **Sleep and lending:** `--idle-sleep` and `POST /v1/vram` ([running](../running.md#sleep---idle-sleep)) use the same copies.
  At a sleep the card's copies of the RAM experts go back to RAM (the CPU computes them) and the driver moves the rest
  of the GPU state to host memory; the wake takes the copies back while the card has room. `reserve_mib` resizes them the
  same way: copies go back when the card has less than that free, return when it has more. Only the copies are lendable:
  the experts a pack keeps on the card have no RAM copy, so a reserve above what the copies free is refused (`400`), and an
  engine with every expert on the card gives nothing back before the driver moves it. Upstream, at `--max-context 1048576
  --kv fp8` with images on, an RTX 5090 held 28.4 GB awake and 20 MiB asleep; the wake took about 2 s and every token after
  it equalled the one before.

## Two cards

Two alike cards (the same SM count and shared memory per block) with peer access both ways (`--gpu 0,1`) run every token
together (`tensor parallelism on CUDA0 and CUDA1`):

- **Weights:** each card holds every dense weight and half of every routed expert: gate and up rows 0 to 575 on the first
  card, 576 to 1,151 on the second, with the matching half of each down row (about 7.36 GiB of experts a card). Each also
  keeps a copy of the output projection's columns that read its heads (about 0.08 GiB).
- **A decode window's layer:** each card makes the queries, gates and sinks of half of the heads (key/value groups 0 to 3 on
  the first card, 4 to 7 on the second: 12 query heads each) and every group's keys and values, so both hold whole caches.
  It runs its heads' attention and the output projection over their 1,536 input columns, then the router and the mixes on
  the same rows as the other card, then its half of the routed experts. After each of the two products the card writes its
  sum into the other's memory, then its count of sums sent, which the other card's kernel waits for (a wait past two
  seconds fails the call); both add the two sums, the first card's first, so both cards hold the same bits
  (`--diag-tp-check` compares every cache after each call). Prompt chunks run the whole attention on both cards and split
  the experts the same way.
- **The head:** the first card holds every row (the drafter reads them all), the second a copy of the second half (about
  0.23 GiB). Each card runs its half of a window's head with the kernels one card runs, so every logit keeps its bits; the
  second runs its half as four products and copies each one's logits into the first card's while the next runs, so a
  window waits only for the last copy.
- **First card only:** the sampler, the drafter and the image encoder. The first card keeps the encoder's region
  (1.37 GiB) free: there is no expert memory to lend.
- **Context:** it must fit beside the weights on both cards: no experts in RAM, no `--expert-cache`. On two 24 GiB cards
  with images on, 262,144 positions fit; 524,288 do not, even with `--kv fp8` (the start says how much is missing).

A pair that is not alike, has no peer access, or asks for `--expert-cache` runs on the first card alone
(`CUDA1 idles (...): CUDA0 runs alone`).

## Long prompts

Prompts of more than 16 tokens read in chunks of up to 4,096 rows: every expert and projection product through MMQ (int8, q8_1
activations, [precision](../precision/inkling2.md#where-each-stage-rounds)), attention on the fp16 tensor cores. The drafter
sees every chunk. With experts in RAM a chunk stages each spilled layer's experts on the card, from their card copies when it
has them all.

Upstream, on an RTX 5090 with the FP4 prompt path: at `--max-context 1048576 --kv fp8` needles at 131K and 400K were found;
decode ran at 264 tok/s on a short chat, 173 at 131K and 108 at 400K (plain decoding), and a 400K prompt read at about 4,110
tok/s. Needles at 10, 50 and 90 % of 31,943 and 127,958-token prompts were found through the server (`--max-context 262144`).
At 400 W with every expert on the card, prompts of 2K, 16K, 32K and 128K tokens read at about 14,100, 13,300, 12,200 and
8,900 tok/s.

## Drafting

On when the model file has the DSpark drafter or `--drafter FILE` gives one. One block pass drafts up to seven tokens: three dense layers over the anchor and seven
mask rows, reading keys and values made from the target's visits 5, 23 and 41 and its final state, then a recurrent Markov
head over the target's head (screened through 2-bit copies of its rows). The draft policy picks each window by expected
tokens per millisecond (`include/granite/spec/draft_policy.hpp`). `--spec N` caps the window (default 8: the last accepted token
plus the block); `--spec 1` turns it off. The target checks every draft: output is plain decoding's.
A window also takes a draft copied from an earlier repeat of at least 3 tokens of the context when the policy expects more
from it than from the block: upstream, a 2,405-token copy prompt decoded 13 % faster.

`--drafter FILE` takes the drafter from another file: a `.granite` pack that carries one, or Unsloth's DSpark GGUF of the
model (architecture `dflash`: `dspark/dspark-Inkling-2-Mini-BF16.gguf`, or `dspark-Inkling-2-Mini-Q8_0_Old.gguf` with
Q8_0 projections). Their names map onto the pack's and the Markov head's 966 padding token rows are cut; both load to the
pack's drafter bytes (`inkling2_drafter_parity`). With the UD-Q4_K_XL GGUF on one RTX 3090 at 270 W, the BF16 file and the
pack's drafter both decode conv 167, struct 317 and prose 139 tok/s. llama.cpp's later DSpark layout (Unsloth's
`dspark-Inkling-2-Mini-Q8_0.gguf`: its own output rows, `markov_w2`, and per-layer context mixes) is refused.

## Parallel slots

`--parallel N` starts N sessions, each with its own caches, rings and drafter context; the forward's scratch, the prompt path
and the drafter's tap are one for all of them. The busy slots decode together, one pass through the weights of up to 8 rows
([batched decoding](../design/batching.md)), the rows going to the drafts likeliest kept. A request alone in its slot gives the
same greedy tokens as with `--parallel 1`, and requests decoding together each give the tokens they give alone.

## Images

`--vision dynamic` ([vision](../running.md#vision---vision)) keeps no card memory, so the context that keeps every expert on
the card is the same as without it. A request's pictures borrow the encoder's region (up to 1.37 GiB) from the experts: the
card copies of RAM experts go back first, then whole layers' experts are copied to RAM and come back byte for byte at the
same addresses after the pictures (one layer, about 0.3 s a request on an RTX 5090 with every expert on the card, upstream). The
engine's own encoder reads a picture at its size, padded to multiples of 40 pixels, one token per 40 x 40 cell: a 640 x 488
picture takes 208 tokens. A picture of more than 10,609 cells
(about 4,120 x 4,120) is scaled down to fit (bicubic). Transparent pixels are laid over the reference processor's grey
checkerboard (8 x 8 squares of 204 and 153), as the model saw in training; the model tends to call such a background black.

## Chat, thinking and tools

- **Template:** the release's Jinja template, with `<|message_*|>` roles and content blocks.
- **Thinking:** the template writes an effort level into a system message: `reasoning_effort` `none` (0), `minimal` (0.1),
  `low` (0.2), `medium` (0.7), `high` (0.9, the template's default), `xhigh` and `max` (0.99), or a number from 1 to 100 as
  hundredths (at most 0.99). There is no hard switch: at `none` the model still writes a short thinking block (a few hundred
  tokens), so leave room in `max_tokens`.
- **Structured output:** under a JSON schema or `json_object` the model thinks first, as its effort asks, and the grammar
  holds only the answer block; the answer cannot end before its value is whole. At `none` it answers at once.
- **Tools:** the model writes `<|content_invoke_tool_json|>{"name": ..., "args": {...}}`; several calls per turn are supported.
  The server also reads the forms the model drifts into: a call whose header has no name, or has no header at all (the name
  is in its JSON), a call written after a block it left open or cut off before its end, and a call written as XML
  (`<invoke name="..."><parameter name="...">...</parameter></invoke>`, bare or inside `<tool_call><function_calls>`, with
  or without `string="true"`) when it names one of the request's tools. Tags quoted in code, or naming no tool of the
  request, stay text. A call cut off by `max_tokens` is not a call (the finish is `length`), but the arguments it streamed
  are closed into valid JSON.

## Limits

- At most two cards, as one tensor-parallel pair (no expert tier, no third card). Experts in RAM run on the CPU,
  so a spilled card decodes slower than a full one: upstream, an RTX 5060 Ti alone at 180 W with a Ryzen 9 9950X and about
  1,100 experts in RAM decoded conversation at 59 tok/s, struct at 118 and prose at 53, about 2.6 times as fast as reading
  them over the bus; prompts read as before (2,430 tok/s at 16K).
- With experts in RAM, the server's hit rate (`hit=`, `decode expert cache hit rate`) and the engine's `experts: ... on the
  card` line give the share of decode lookups whose expert is on the card. The stats page's expert map shows every expert
  in VRAM.
- No video.

## Speed

`granite-bench` at `a4b75aab` on two RTX 3090s at 270 W (PCIe Gen4 x8/x8, driver 615.78.08, CUDA 13.5), the pack above,
drafting on unless the row says otherwise. One card is CUDA1 alone (`--gpu 1`), two are the tensor-parallel pair
(`--gpu 1,0`; CUDA0 also drives the display), run in A-B-B-A order:

| | One RTX 3090 | Two, tensor parallel | Two / one |
| --- | ---: | ---: | ---: |
| Decode: conversation | 172 tok/s | 233 tok/s | 1.35 |
| Decode: struct | 314 tok/s | 485 tok/s | 1.55 |
| Decode: prose | 136 tok/s | 193 tok/s | 1.42 |
| Plain decode (`--spec 0`) | 8.1 to 8.4 ms a token | 6.1 to 6.2 ms a token | 1.34 to 1.36 |
| Prompt read, 16K | 3,825 tok/s | 4,613 tok/s | 1.21 |
| Prompt read, 64K | 3,329 tok/s | 3,909 tok/s | 1.17 |

- Conversation and prose sample at temperature 0.6, and one card and two round differently, so their texts differ. Each
  decode row is the tokens over the decode time of five runs: seed 1 once in each leg, then seeds 2 to 4. Conversation's
  runs spread from 168 to 183 tok/s on one card and from 221 to 240 on two. Struct is greedy and runs to its 8,192-token cap.
- Drafts accepted: one card 62 % (conversation), 97 % (struct), 44 % (prose); two cards 57, 95 and 42 %. The draft policy
  times its windows: with two cards' faster verify it drafts slightly longer (struct 5.63 tokens against 5.56) for the same
  tokens a window.
- Plain decode takes 8.1 ms a token on conversation and prose and 8.4 on struct's longer context; two cards take 6.1 and 6.2.

Upstream Basalt's `basalt-bench` (seeded, `-r 2`), RTX 5090 alone at 400 W, the pack above, drafting on, the FP4 prompt
path:

| | Inkling 2 Mini (Basalt) | llama.cpp, IQ4_NL GGUF |
| --- | ---: | ---: |
| Decode: conversation | 347 tok/s | 94.5 |
| Decode: struct | 628 tok/s | 92.5 |
| Decode: prose | 296 tok/s | 94.5 |
| Prompt read | 12,310 tok/s (16K) | 3,093 (`llama-bench` pp512) |

Drafts accepted: conversation 68.0 %, struct 97.5 %, prose 50.7 %. llama.cpp (29b52d398, `-fa 1`, full offload, no drafting)
ran the same token ids through `llama-server` with the same lengths and temperatures; `llama-bench` tg128 reads 96.3 tok/s.
The struct fixture runs to its 8,192-token cap on both engines.
