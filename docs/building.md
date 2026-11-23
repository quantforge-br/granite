# Building and testing

The build commands are in the [README](../README.md). Requirements: [platforms](platforms.md).

## What is in the repository

- `src/`, `include/`: the engine (`granite`, `granite-bench`, `granite-probe`, `granite-vision`).
- `serve/`: the Python server with the OpenAI and Anthropic API and the stats page.
- `tools/`: the converter (`tools/convert/convert.py`), the `.basalt` reader (`model_file.py`), the gate-row report and diff
  (`ppl_gate.py --report`, `--diff`) and the tokenizer.
- `tests/`: parity and unit tests. `data/`: expert profiles (`data/profiles`) and benchmark fixtures (`data/bench`).

The build fetches nothing. The ggml and llama.cpp parts granite uses (quantization formats, CPU dot products, MMQ) are in `src/` and
`include/granite/ggml/`, under the MIT licence in `THIRD_PARTY_LICENSES`.

`-DWERROR=ON` makes project warnings errors (CI sets it on Linux).

## Models in the build

granite builds one model, Inkling 2 Mini (architecture id `inkling2`). `-DGRANITE_MODELS` is a semicolon list of architecture ids,
or `all`; the default is `inkling2`. An unknown id or an empty list stops the configure and names the valid ids. A pack of any
other architecture fails at load: no model adapter is registered for it.

## Tests

Configure with `-DBUILD_TESTING=ON`, build, and run `ctest --test-dir build`.

- Tests are labelled `cpu` or `gpu`: the kernel parity tests need the card, `ctest --test-dir build -L cpu` runs the rest.
- Server tests: `python -m unittest discover -s serve/tests -t .` after `pip install -r requirements-test.txt`.
- Tokenizer parity: register `tokenizer_parity` with `-DGRANITE_TOKENIZER_DIR=<a pack's tokenizer/ directory>`.

`granite-bench` reads its prompts from `data/bench` (run from the repository root, or pass `--fixtures DIR`).
