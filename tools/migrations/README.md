# Migration

`pack_to_granite.py` turns an upstream Basalt pack (`.basalt`) into a `.granite` pack, which is what the engine reads.
Both are version 2 of one container; granite's has its own names (the magics and `meta`'s `format`). Nothing is downloaded
or re-quantized: every other section is copied byte for byte, each checked against the SHA-256 in the old file's table.

```sh
python3 tools/migrations/pack_to_granite.py Inkling-2-Mini-NVFP4-DQ8.basalt Inkling-2-Mini-NVFP4-DQ8.granite
```

It needs Python 3.10+ (no packages) and free disk space for a second copy of the pack. The old file is only read; delete
it once the new one runs. What changed: [the file format changelog](../../docs/granite-file-changelog.md).

`config_v1_to_v2.py` turns a single-model server config (`"server"` naming the model, `"engine"` its flags) into the v2
shape with `"models"`, which is what the server reads ([configuration](../../docs/configuration/models.md)). The old file
is only read: it prints the new one, or writes it to a new file with `-o`.

```sh
python3 tools/migrations/config_v1_to_v2.py granite_config.json -o granite_config.v2.json
```
