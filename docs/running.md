# Running the server

Settings: [configuration](configuration/index.md). Diagnostic flags: `bin/granite --help`. Several models: [models and the router](configuration/models.md).

## Start-up

The page is up at once and the model loads behind it. The header reads `Loading` with the percent over a progress bar while weights load; choosing a model in the picker loads it, and the Unload button beside the model name unloads the loaded one until the next request; each reads `Loading…` or `Unloading…` until the server shows the change, and a refused one shows its error in the header. `/health` and `/metrics`
show the load: `load.percent` and `load.gib`, the GiB placed so far in one figure over every byte the model loads (a card, a second card, RAM), where it was one figure per target. A failed start-up load ends the server. Requests wait during a load and get a 503 if it fails.

## Concurrent requests: `--parallel N`

`--parallel N` (1 to 8) serves N requests at once in batch slots. Further requests wait in arrival order. `--max-context` is shared:
each slot holds `max-context / N` tokens (`--parallel 4 --max-context 262144` gives four slots of 65,536). A request longer than its slot
is refused with a 400 that names the length; the start log's `slots` line shows the context each slot gets. `--slot-full-context` gives every slot the whole `--max-context`, at N times the KV cache.

Each slot keeps two checkpoints (end of the system prompt, last turn), so requests that share a long system prompt read it once.

The engine tokenizes every prompt and matches it against what each slot holds. A reply the client sends back unchanged continues the
slot's own tokens even when the model wrote it in other pieces than the tokenizer cuts the same bytes into (not with images).

### Session affinity

With `--parallel`, a request that names its session goes to the slot that session's conversation sits in, so the next turn continues
the K/V already there. The session is the first of these headers that is set, then the request body's `session_id`: `x-llm-flow-id`,
`x-claude-code-session-id`, `x-session-id`, `x-session-affinity`, `x-grok-conv-id`, `x-grok-session-id`. A request with none picks a
slot by the longest shared prefix, then the least recently used. A session may hold several slots: a client may send one session id for a
session and its parallel subagents (Claude Code's `x-claude-code-session-id`, OpenCode's `X-Session-Id` and `x-session-affinity`, which
name the parent session), and each conversation keeps the slot its K/V is in. A request that continues none of the session's own slots
takes an unowned one first; with every slot owned, the session's own least recently used one comes before another session's.

### Prompts beside decoding slots

A prompt of up to 16,384 tokens is read at once, and the slots wait a few seconds. A longer one is read beside them, on its own thread,
while the slots keep decoding. A short request that arrives meanwhile goes first at the end of the current chunk.

## The console log

Every line starts with `[server]`. Values are `key=value`. Start-up prints a legend, a line per model, `ready`, `loaded`, `kv`, `slots`.
While a prompt is read, a slot prints `pp=<read>/<total>` with the rate. While requests generate, a `live` line appears every 2 s.
Each request ends with one `done` line:

```
[server] s1 done  pp=96/89c 0.1s  tg=1720 119.6t/s acc=65% dl=2.9  ttft=0.1s  end=stop  hit=89.7%  pcie=0.0%
```

- `pp=<prompt tokens>/<cached>c`: the rate counts newly read tokens only.
- `acc`: drafts accepted over drafts offered. `dl`: tokens per window, from the engine's own window count over the whole request (its solo and batch runs and every pass). With `--parallel` a window serves every busy slot in a fixed number of rows, so `dl` falls toward 1.0 as the slots fill. Both are absent when the request did not draft.
- `end`: `stop`, `length`, `cancelled`, `disconnect` or `error`.
- `hit`: the share of decode's expert lookups whose expert is on the card (below 100% when experts are in RAM). `pcie`: the share
  read over PCIe or from another card; Inkling 2 Mini reads none that way.

The engine's own `granite ...` lines go to `~/.cache/granite/<id>/engine.log`, one folder per model file (`<id>` is 16 hex digits
taken from the file); `[server] engine log: PATH` names it at start. A log of 10 MiB or more moves to `engine.log.1` at the
next start. An engine that dies of a fault or an unhandled exception writes a `granite crash: error: ...` line (the fault, the stage
it was in) and the server's warning quotes it with the exit status. The folder also holds the conversation
cache. Nothing is unpacked from the model file: the server reads the tokenizer from its sections, and the engine reads the vision
projector inside it.

## The stats page

The server's address serves a page, updated every second from `/metrics`, `/slots` and `/api/logs`. The top of the page always shows:

- Decode and prefill: the live rate with a 60 s chart; when idle, the last request's decode and the last prompt read of at least 512
  new tokens (a cached follow-up has no meaningful read rate). The decode number's tooltip has the best decode and the conversation
  cache; the prefill number's has the last prompt, its prefix cache hit and the best prefill (the fastest read listed when the
  server has none yet).
- Experts: where the last request's expert lookups were served (VRAM, CPU and RAM), as a legend, and a map of 640 experts sampled
  evenly from every layer (`experts_map`): its brightness is how often decode looked it up lately. Inkling 2 Mini marks every
  sampled expert as in VRAM, also when some are in RAM.
- With no model loaded, a slow field of basalt columns on its own (8 frames a second, paused while the tab is hidden, one still
  frame under reduced motion).

Under it, four tabs (keys `1` to `4`, `L` for the logs, or `#tab=req|slots|hw|logs` in the address):

- Requests: the queue, the recent requests, tokens per request, the last request, the best since the server started, the
  conversation cache (`conversation`: `parked`, `bytes`, `evictions`, `restored`, `restored_ms`) and the totals (`totals`).
- Slots: a card per slot with its current request, or for an idle slot its last one, its context and its drafting; past four
  slots, compact rows. With no model loaded the cards stay, empty.
- Hardware: each card's load, power, temperature, VRAM and PCIe traffic, and the CPU, RAM, disk reads, experts in VRAM, driver and
  uptime.
- Logs: the console lines, all or warnings only. The engine's long cache lines are shown short (`cache: restored 187k tokens in
  912 ms`); the log and `/api/logs` keep the raw line. The footer always shows the newest line and the request and token totals.

The engine's own figures, which the server asks for with `STATS` once the engine is ready:

- `/metrics` `experts_map`: `total` experts, and for each sampled one `tier` (0 VRAM, the only value Inkling 2 Mini gives),
  `routed` (decode lookups a second, 0 once nothing runs) and `recent` (its lookups, fading by 0.85 a second).
- `/slots` per slot: `started` (epoch seconds), `path` (the endpoint) and `prompt_ms` (the read time) of the running request;
  for its current or last request `accept_pos` (drafts accepted over offered by draft position) and `accept_hist` (tokens per
  decode window, a sample a second over 60 s); for a model with routed experts `hit`: its windows' lookups split `vram`, `peer`
  (always 0: Inkling 2 Mini copies no expert over the bus) and `cpu` (experts in RAM), with `lookups_s`, `peer_s` and
  `ram_s` a second. A batch window several slots share
  counts its lookups for each of them.
- `/metrics` `engine`: `commit` (the build's), and `expert_slots`, the experts every card holds now (fewer while asleep).
- `/metrics` `hardware`: each card's `sm_clock` and `mem_clock` (MHz), `throttle` (what holds the clocks down, empty for nothing),
  `expert_slots`; `cpu_ccd` (the load of each group of cores that share an L3 cache), `ram_parts` (`experts` the engine holds in
  RAM, `page_cache`, `other`; Linux only) and `disk_hist` (the read rate's last 60 s).

The header word reads `Thinking`, `Answering` or `Tool call` (`live.phase`) while exactly one request runs. The conversation cache
figures come from the engine's `conversation cache:` lines and clear when the engine unloads, sleeps or restarts; a model that prints
no line when it parks shows them after its first restore, without evictions.

## Conversations on disk: `--conversation-cache-disk-mib`

Off at 0. With a budget, conversations of at least 8,192 tokens are written to `conversations` in the model's cache folder when they leave
RAM, a session, or at exit. Only the system prompt's and the last three turns' states are stored. Files are bound to the model and settings
they were made with, and checked by hash; a bad file is deleted. The folder is capped by the budget (least recently used first), and a
write always leaves 4 GiB of the volume free. After a restart, a prompt that starts with a stored conversation reads it back.

## Sleep: `--idle-sleep`

After the given seconds without requests, the engine frees its VRAM: the card's copies of the experts planned into RAM go back to
RAM, and the driver moves the rest of the GPU state to host memory. The process stays. The next request wakes it in seconds. The
server stays awake when the GPU state would not fit in free RAM with 8 GiB to spare. `--idle-unload` ends the process instead.
Details: [Inkling 2 Mini](models/inkling2.md#memory).

## Lending VRAM: `POST /v1/vram`

`POST /v1/vram` with `{"reserve_mib": N}` moves the card's copies of the experts planned into RAM back to RAM, or returns them,
to keep N MiB free (`null`: the start's `--vram-reserve-mib`, default 0). Only those copies are lendable: the experts the card
keeps have no RAM copy, so a reserve above what the copies free is refused (`400`) ([Inkling 2 Mini](models/inkling2.md#memory)).
A resize needs a gap between requests. Idle, it applies at once and answers `200` with the new figures. While a request runs it is
queued and answered `202 {"status": "pending"}`: it applies when the running requests have finished, and requests that arrive meanwhile
wait for it, so steady traffic cannot hold it off. A later POST replaces a queued one (`"replaced": true`). A malformed body is a
`400`. `GET /v1/status` shows `vram.pending` (the queued reserve, or `null`), `vram.applied` and `vram.error` (the engine refused
the queued one).

## Memory floors

- `--ram-reserve-mib` (default 2048) is the RAM left free for the rest of the system. While serving, the RAM guard holds it: below
  it, it parks conversations to disk or drops them. A request waits up to 30 s for RAM to return. A running request stops with an
  error only when RAM is critical (under 1 GiB free, or below the reserve under memory pressure). `--ram-reserve-mib 0` turns the
  guard off. Programs left idle beside a small reserve may be swapped out.
- The disk tier keeps its floor a GiB above the reserve.

## Vision: `--vision`

`--vision off | dynamic` (config: `"vision"` in a model entry). Inkling 2 Mini's own encoder runs inside the engine on the main
card and reads the projector in place from the model file's `vision/mmproj` section; the server refuses the other modes. It keeps
no card memory at rest: a request's pictures borrow the encoder's region from the experts and give it back
([Inkling 2 Mini](models/inkling2.md#images)).

Which images a request may name: [image sources](configuration/image-sources.md). An image may not sit in a system prompt.
