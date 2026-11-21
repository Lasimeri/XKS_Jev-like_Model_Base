# Changelog

## 0.1.0-rc.1 (2026-11-21)

The first release candidate.

### What it is

Jev-like typed judgments (noul, choice, score) read from a local LLM's
next-token distribution, on the CPU: one Rust crate and one binary
(`jevbase`), llama.cpp (pinned at f5b9bd39) linked statically and built for
AVX2 + FMA.

- **The method** of Intel-Phi-Jev's `xks`: each question rendered so its
  answer is the next token; the state prefilled once and forked per question;
  all suffixes in one batch; the label logits read at the last position
  (log-sum-exp over the vocabulary, exact or AVX2 + FMA); a choice averaged
  over 3 rotations of its options; the trie for labels of several tokens;
  both prompt layouts (lettered options, and the `jev` layout).
- **`jevbase ask`** (requests from files or standard input), **`jevbase
  serve`** (the System One API over HTTP), **`jevbase bench`** (decisions per
  second, cold and warm), **`jevbase info`**, and a terminal interface (std
  and termios only) that browses the example requests.
- **The System One API as TypeSafe documents it** (OpenAPI 0.2.0): request
  validation with the documented 422 and 400 bodies, `/v1/models` with the
  `jev-latest` and `jev-preview` aliases, bearer keys (403 without a key, 401
  with a wrong one, as the hosted Jev answers), the per-question and
  per-request context limits and the long-context header.
- **A client of any System One server** (`jevbase`, `xks`, the hosted Jev),
  configured as TypeSafe's SDKs are (`TYPESAFE_BASE_URL`,
  `TYPESAFE_API_KEY`, `TYPESAFE_DEFAULT_MODEL`), usable without the model
  (`default-features = false`: no llama.cpp to build).

### Verified (2026-11-21, Ryzen 7 5800X, Qwen3.8 2B Q8_0, 8 threads)

| check | result |
|---|---|
| against `xks` (the oracle), 32 requests, 113 decisions: `ask`, exact readout | byte-identical output (answers, usage, raw log-probabilities to 17 digits) |
| `serve` over HTTP, after the TypeSafe changes | byte-identical |
| the `jev` layout (every choice read by trie) | byte-identical |
| AVX2 + FMA readout | 113 of 113 answers identical, largest raw log-probability difference 4.7e-14; 270 to 285 us per 248,320-entry row against 675 to 714 exact |
| TypeSafe conformance (`tests/conformance/typesafe.sh`) | 17 of 17; 20 of 20 with `api_keys` set |
| unit tests | 18 of 18 client-only, 24 of 24 full; clippy clean in both shapes |
| decisions per second (12 developer-task requests, one round) | 1.25 warm (800 ms each), 1.11 cold, 0.58 for one question alone |

### Fixed during the release check

- A command whose output was closed by its reader (`jevbase info | head
  -1`) panicked ("failed printing to stdout: Broken pipe") in 50 of 50 runs;
  it now ends quietly as `cat` does (SIGPIPE, 50 of 50). `serve` keeps
  SIGPIPE ignored, so a vanished peer stays an error on the write.
- The binary carried 161 paths of the machine that built it (152 llama.cpp
  source paths from its asserts' `__FILE__`, 9 paths into cargo's registry).
  `make build` now maps them to relative ones (`-ffile-prefix-map` for
  llama.cpp, `--remap-path-prefix` for rustc): 0 remain, and two clones of
  the same commit built in folders of different path lengths give
  byte-identical binaries (one SHA-256). A change of the Makefile now
  rebuilds llama.cpp, so new flags always reach it. Oracle parity after the
  rebuild: largest reported and raw log-probability difference 0 (464
  compared).

### Known gaps

- The speed figures come from one round on twelve requests; the three-round
  bench on the full example files has not been run.
- Mechanical Jev's `mjev` has not been run against `jevbase serve` (its
  client contract was read and kept: `/health`, `data[].id`).
- The hosted Jev was reached only without a key: its refusals were measured,
  no answer was.
