# Models and the router

Several model entries share one server. A request's `model` field picks one, and one is loaded at a time. A swap ends one
engine process and starts the next.

granite builds one model, Inkling 2 Mini, so the entries are Inkling 2 Mini packs with their own settings. A request may name
a model by its file name, so each entry needs its own file.

```json
{
  "version": 2,
  "engine": { "idle-sleep": 300 },
  "models": [
    { "alias": "Inkling", "model": "inkling2.granite", "gpu": "0,1", "max-context": 262144, "vision": "dynamic" },
    { "alias": "Inkling-1M", "model": "inkling2-1m.granite", "gpu": "0", "kv": "fp8", "max-context": 1048576 }
  ],
  "server": { "host": "127.0.0.1", "port": 8080 }
}
```

- `models`: each entry has a `model` (relative to the file) and an optional `alias`, the name the API reports.
- `engine`: applies to every model that takes it. An entry's own value wins.
- `server`: shared by the whole server: `host`, `port`, `allowed-host`, `cors-origin`, `image-url-host`, `image-dir`,
  `image-remote-only`, `max-payload-mib`, `lazy`, `open`, `debug`, `request-lines`.
- An entry takes engine flags and these server options: `gpu`, `vision`, `parallel`, `slot-full-context`, `idle-sleep`,
  `idle-unload`, `min-free-vram-mib`, `fit-max-tokens`, `grammar`, the reasoning settings and the sampling defaults.
- `version` is `2`. A file in the old shape (no `models`) is refused.
- The command line wins for every model.

## Checks at start

Each entry is checked before anything loads, by the engine's own rules:

| | Model entry or command line | Top-level `engine` |
|---|---|---|
| Flag the model does not have | stops the start | left out for that model; a start line says so |
| Value the model refuses (`max-context` past its maximum) | stops the start | stops the start |
| Unknown name | stops the start | stops the start |

`kv` takes `fp16` or `fp8`; another value stops the start. Two entries that answer to the same name (an alias, or the same
file), or two entries sharing a conversation folder on disk, stop the start too. Copies of one pack share a cache folder, so with
`conversation-cache-disk-mib` each needs its own `conversation-cache-dir`.

## Requests and turns

A request's `model` is the alias or the file's name on every API. Any other name is a 404. `/v1/models` shows `loaded`, `loading`,
`unloaded` or `failed`.

Clients wait through a swap; they do not get a 503 for it. A streamed response sends keep-alive comments every 10 s while it waits.

- A request for the loaded model runs at once while nothing waits. Once a request for another model waits, later arrivals queue behind it.
- The head of the queue picks the next model; its swap starts when the running requests finish. The model takes every request queued
  for it at that moment, in one batch.
- A failed load answers its batch with a 503 naming the model, and the queue goes on. A load tries twice. With one model, a failed start
  load ends the server.

## What a swap keeps

| | Across a swap |
|---|---|
| Conversations on disk | kept in each model's folder |
| Conversations in RAM and slots | written to disk at exit when `conversation-cache-disk-mib` is on; otherwise lost |
| Sleeping engine | wakes on the next request for it |
| `POST /v1/vram` | the machine's; a model loaded later applies the last reserve too |
| Metrics | per model (`GET /metrics?model=NAME`) |

`POST /v1/load` with `{"model": NAME}` swaps ahead of a request. `/health` and `/metrics` show the swap's progress. The console prints
`[server] swap Inkling -> Inkling-1M (2 waiting)`, then `[server] swap: Inkling-1M ready in N s`.
