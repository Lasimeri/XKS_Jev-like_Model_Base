# XKS_Jev-like_Model_Base

Typed judgments (**noul** yes/no, **choice** one of N, **score** a level of
2 to 10) read from a local LLM's next-token distribution, on the CPU. A
small base to start other projects from: one Rust crate, one binary
(`jevbase`), llama.cpp linked statically and built for AVX2 + FMA.

It is a simpler re-implementation of [Intel-Phi-Jev](https://github.com/Lasimeri/Intel-Phi-Jev)'s
`xks`, using **the same prediction method**, checked against `xks` as the
oracle: on the same 32 requests (113 decisions) both give byte-identical
output (see [Oracle](#oracle)).

## How a decision is read

The model never writes an answer. For each question:

1. The request's **state** (a document, a command, a log) goes into a
   prompt prefix that every question shares, inside `<document>` tags, as
   user text: its special tokens are never parsed, so it cannot act as
   instructions.
2. Each **question** becomes a suffix: the question, its options lettered
   `A)`, `B)`, ..., and the cue `Answer:`. Its answer is the next token:
   ` A`, ` B`, ... (one token each; 27 to 255 options use ` 001`..` 255`).
3. The state is **prefilled once** on sequence 0 and kept between requests.
   Every question gets a copy of it (`llama_memory_seq_cp`, recurrent state
   included on hybrid models), and all suffixes go through the model in one
   batch. Only the last position of each suffix produces logits.
4. The **readout**: log-probabilities over the whole vocabulary
   (log-sum-exp), kept at the label tokens, softmax over the labels.
5. A **choice** is read in 3 cyclic rotations of its options and averaged
   (position-bias control). Then a temperature (1.0 unless set), and the
   typed answer: P(yes); the option, its probabilities and a confidence; the
   expected level, its probabilities and a confidence.

The whole network runs for every question: what makes it a "decision" model
is what is read (a few rows of the output layer at one position) and what is
shared (the state, computed once).

## Speed on the CPU

Measured 2026-11-21 with `jevbase bench` on a Ryzen 7 5800X (8 cores, AVX2 +
FMA, no AVX-512), Qwen3.8 2B Q8_0 in RAM, 12 developer-task requests (36
decisions; a choice is 3 reads, so 60 reads), one round per setting, with a
desktop session running beside it. A decision is one answered question.

| setting | cold (state prefilled) | warm (state cached) | one question alone (warm) |
|---|---|---|---|
| 8 threads (default) | 1.11 /s (900 ms) | 1.25 /s (800 ms) | 0.58 /s (1,730 ms) |
| 6 threads | 0.94 /s | 1.06 /s | 0.50 /s |
| 4 threads | 0.67 /s | 0.76 /s | 0.35 /s |
| 8 threads pinned one per core | 1.13 /s | 1.26 /s | 0.59 /s |
| flash attention off / on | 1.11 / 1.12 /s | 1.25 / 1.27 /s | 0.58 / 0.59 /s |
| 8 threads again, last (drift control) | 1.01 /s | 1.15 /s | 0.54 /s |

- The prompt runs at 196 to 201 tokens a second at 8 threads and scales
  almost linearly with threads: the CPU's arithmetic is the limit. A
  question suffix is about 94 tokens, a decision about 157 on average.
- Questions asked together are cheaper than one at a time (800 vs 1,730 ms):
  one decode pushes all forks' tokens through the weights once.
- Pinning and flash attention moved less than the run-to-run drift (9%),
  so the defaults keep the oracle's settings. Repacking cannot help a Q8_0
  model here: this llama.cpp has AVX2 repack kernels for Q4_0, Q4_K,
  IQ4_NL and MXFP4 only.
- The readout (log-sum-exp over the 248,320-token vocabulary, once per read)
  takes 270 to 285 microseconds a row with AVX2 + FMA against 675 to 714 for
  the exact scalar sum (7 runs each), 2.4 to 2.5 times faster, about 0.7 ms
  saved per decision: under 0.1 percent of it, since the model's forward
  pass is the rest.

## Quick start

```sh
make            # fetch llama.cpp (pinned), build it for AVX2/FMA, build jevbase
make run        # the terminal interface (or ./target/release/jevbase)
```

The default model is `~/models/Qwen3.8-2B-Distill/Qwen3.8-2B-Q8_0.gguf`; any
GGUF llama.cpp loads works (`-m FILE`, or `model` in `jevbase.conf`).

```sh
jevbase ask examples/query.json              # one request, its answers as JSON
jevbase ask examples/dev_tasks.jsonl         # one request per line
echo '{"state":"...","questions":{...}}' | jevbase ask
jevbase serve                                # POST http://127.0.0.1:8097/v1/systemone
jevbase bench                                # decisions per second
jevbase info                                 # model, CPU features, every setting
make install                                 # ~/.local/bin/jevbase
```

### The terminal interface

`jevbase` in a terminal opens it on `examples/query.json` (or a file named
on the command line). The document pane is the state; the questions pane
lists the questions; **F5** (or Ctrl-R) asks them and the answers pane
shows each answer with a bar per option. The model loads in the background
while you edit.

| key | where | does |
|---|---|---|
| F5, Ctrl-R | anywhere | ask every question |
| Tab, Shift-Tab | anywhere | move between Document, Questions, Answers |
| a, Enter, d | Questions | add, edit, remove a question |
| Ctrl-O, Ctrl-S | anywhere | open, save a request file (the JSON `ask` takes) |
| F1 | anywhere | help |
| Ctrl-Q | anywhere | quit |

In the question editor, Tab moves between fields, Left/Right picks the type,
F2 or Ctrl-S keeps the question and Esc drops it. Options are one per line:
`key: description` for a choice, one level per line for a score, and an
optional `true: ...` and `false: ...` for a noul.

## Requests

The wire format is TypeSafe's System One (`POST /v1/systemone`), the one
`xks`, Mechanical Jev (`mjev`) and TypeSafe's SDKs speak. A request is a
`state` and named `questions`; see [`examples/query.json`](examples/query.json):

```json
{
  "state": "$ cargo clean && rm -rf ~/.cargo/registry && cargo build --release",
  "questions": {
    "class":   {"type": "choice", "instructions": "What kind of command is this?",
                "criteria": {"build": "compiles or bundles code", "test": "runs a test suite"}},
    "risk":    {"type": "score", "instructions": "How risky is this command?",
                "criteria": ["read-only", "changes files inside the project only",
                             "deletes or changes files outside the project"]},
    "outside": {"type": "noul", "instructions": "Does it touch files outside the project?"}
  }
}
```

`--debug` (or `debug = true`) adds each question's raw log-probabilities,
tokens evaluated and cached, label mass and latency.

## Settings

Every key is in [`jevbase.conf.example`](jevbase.conf.example) with its
default. They are read in this order, later ones winning: built-in defaults,
`~/.config/jevbase/jevbase.conf`, `./jevbase.conf`, `-c FILE`,
`JEVBASE_<KEY>` environment variables, command-line flags (`-m`, `-t`,
`--set key=value`). `jevbase info` prints the effective values.

## As a library

```rust
let cfg = jevbase::Config::load(None)?;            // or Config::defaults()
let judge = jevbase::open(&cfg)?;                  // loads the model
let req: jevbase::Request = serde_json::from_str(text)?;
let eval = judge.evaluate(&req)?;                  // typed answers
```

`judge.raw(&req)` gives the uncalibrated per-option log-probabilities, and
`judge.engine.score_many(prefix, items)` reads any prompts and labels you
render yourself.

## Oracle

The method is Intel-Phi-Jev's; its `xks` is the oracle. The check
(`make parity XKS=/path/to/xks`) runs both on the same 32 requests (113
decisions: 41 choice including one of 30 options read as a trie, 42 noul,
30 score; Intel-Phi-Jev's examples), in the same order from a fresh start,
with the oracle's x86 settings (8 threads, repacking off) and the exact
readout, built against the same llama.cpp commit and CPU flags. Measured
2026-11-21 on a Ryzen 7 5800X with Qwen3.8 2B Q8_0:

| run | result |
|---|---|
| `jevbase ask`, readout exact | byte-identical to `xks`: every answer, every raw log-probability (17 digits), tokens, label mass |
| `jevbase ask`, readout avx2 | 113 of 113 answers identical; largest raw log-probability difference 4.7e-14 |

To build the oracle: in an Intel-Phi-Jev checkout, `make llama-shared` here
first, then
`LLAMA_CPP_DIR=<this>/third_party/llama.cpp LLAMA_BUILD_DIR=<this>/third_party/llama.cpp/build-shared/bin cargo build --release`.

## What is in the repository

| path | what |
|---|---|
| `src/protocol.rs` | the System One request and response |
| `src/prompt.rs` | questions rendered so their answer is the next token |
| `src/engine.rs` | llama.cpp in process: prefill once, fork, read, trie |
| `src/readout.rs` | log-sum-exp over the vocabulary: exact, or AVX2 + FMA |
| `src/score.rs` | softmax, temperature, confidence, expected level |
| `src/judge.rs` | a request in, typed answers out |
| `src/config.rs` | every setting |
| `src/server.rs` | `jevbase serve` (std only) |
| `src/tui/` | the terminal interface (std and termios only, no TUI crate) |
| `csrc/shim.c` | the flat C surface of llama.cpp the Rust side calls |
| `tests/parity/` | the oracle requests, runner and comparison |
| `tests/bench/` | the tuning workload and script |
| `examples/` | requests to try (Intel-Phi-Jev's examples) |

## License

MIT or Apache-2.0, at your option. See [NOTICE](NOTICE) for what is ported
from where.
