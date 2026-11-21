//! `jevbase`: the command line.
//!
//!   jevbase                   the terminal interface (in a terminal)
//!   jevbase tui [FILE...]     the same, opening request files
//!   jevbase ask [FILE...]     answer requests (.json: one; .jsonl: one a line)
//!   jevbase serve             POST /v1/systemone over HTTP
//!   jevbase bench [FILE...]   decisions per second, cold and warm
//!   jevbase info              the model, the CPU features, every setting

use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use jevbase::config::{Config, KEYS};
use jevbase::readout::{self, Readout};
use jevbase::{Judge, Request, engine, server, tui};

const HELP: &str = "\
jevbase: Jev-like typed judgments (noul / choice / score) from a local LLM, on the CPU

usage: jevbase [options] [command] [args]

commands:
  (none), tui [FILE...]
                      the terminal interface: edit a document and its questions,
                      F5 asks, Ctrl-N/P steps through the requests of the files
                      (.json one, .jsonl one a line); default: examples/*
  ask [FILE...]       answer requests: a .json file is one request, a .jsonl file
                      one per line; no FILE or `-` reads standard input. Prints
                      one JSON response per request.
  serve               POST /v1/systemone, GET /health, GET /v1/models on `bind`
                      (default 127.0.0.1:8097)
  bench [FILE...]     decisions per second: each request cold (state prefilled),
                      warm (state cached) and one question alone (state cached),
                      over --rounds rounds (default 3). Default files:
                      examples/dev_tasks.jsonl examples/long_sessions.jsonl
  info                the model file, the CPU features llama.cpp uses, every setting

options:
  -c, --config FILE   read settings from FILE too (after ./jevbase.conf)
  -m, --model FILE    the GGUF model (setting `model`)
  -t, --threads N     CPU threads (setting `threads`)
      --set KEY=VALUE any setting (see `jevbase info` for the keys)
      --debug         raw log-probabilities, tokens and latency in every answer
  -v, --verbose       show llama.cpp's own log
      --rounds N      bench: rounds (default 3)
      --snapshot WxH  tui: print one frame as text instead of running
      --ask           tui --snapshot: load the model and answer first
  -h, --help          this text
";

struct Cli {
    config: Option<PathBuf>,
    sets: Vec<(String, String)>,
    command: String,
    args: Vec<String>,
    rounds: usize,
    snapshot: Option<(usize, usize)>,
    snapshot_ask: bool,
}

fn parse(argv: &[String]) -> Result<Cli, String> {
    let mut cli = Cli {
        config: None,
        sets: Vec::new(),
        command: String::new(),
        args: Vec::new(),
        rounds: 3,
        snapshot: None,
        snapshot_ask: false,
    };
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match a.as_str() {
            "-h" | "--help" => {
                cli.command = "help".into();
                return Ok(cli);
            }
            "-c" | "--config" => cli.config = Some(PathBuf::from(value(a)?)),
            "-m" | "--model" => cli.sets.push(("model".into(), value(a)?)),
            "-t" | "--threads" => cli.sets.push(("threads".into(), value(a)?)),
            "--rounds" => {
                cli.rounds = value(a)?
                    .parse()
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or("--rounds needs a whole number above 0")?
            }
            "--snapshot" => {
                let v = value(a)?;
                let size = v
                    .split_once('x')
                    .and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))
                    .ok_or_else(|| format!("--snapshot {v}: expected WIDTHxHEIGHT, e.g. 110x34"))?;
                cli.snapshot = Some(size);
            }
            "--ask" => cli.snapshot_ask = true,
            "--set" => {
                let kv = value(a)?;
                let (k, v) = kv
                    .split_once('=')
                    .ok_or_else(|| format!("--set {kv}: expected KEY=VALUE"))?;
                cli.sets.push((k.trim().into(), v.trim().into()));
            }
            "--debug" => cli.sets.push(("debug".into(), "true".into())),
            "-v" | "--verbose" => cli.sets.push(("verbose".into(), "true".into())),
            s if s.starts_with('-') && s != "-" => {
                return Err(format!("unknown option {s} (jevbase --help)"));
            }
            _ if cli.command.is_empty() => cli.command = a.clone(),
            _ => cli.args.push(a.clone()),
        }
    }
    if cli.command.is_empty() {
        let terminal = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
        cli.command = if terminal || cli.snapshot.is_some() {
            "tui"
        } else {
            "help"
        }
        .into();
    }
    Ok(cli)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match run(&argv) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("jevbase: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(argv: &[String]) -> Result<ExitCode, String> {
    let cli = parse(argv)?;
    if cli.command == "help" {
        print!("{HELP}");
        return Ok(ExitCode::SUCCESS);
    }
    let mut cfg = Config::load(cli.config.as_deref())?;
    for (k, v) in &cli.sets {
        cfg.set(k, v).map_err(|e| format!("{k}: {e}"))?;
    }
    match cli.command.as_str() {
        "tui" => {
            let files: Vec<PathBuf> = cli.args.iter().map(PathBuf::from).collect();
            match cli.snapshot {
                Some((w, h)) => print!("{}", tui::snapshot(&cfg, &files, w, h, cli.snapshot_ask)?),
                None => tui::run(&cfg, &files)?,
            }
            Ok(ExitCode::SUCCESS)
        }
        "info" => info(&cfg),
        "ask" => ask(&cfg, &cli.args),
        "serve" => {
            let judge = open(&cfg)?;
            server::serve(&judge, &cfg.bind, |addr| {
                eprintln!("jevbase: serving POST http://{addr}/v1/systemone (Ctrl-C stops)")
            })?;
            Ok(ExitCode::SUCCESS)
        }
        "bench" => bench(&cfg, &cli.args, cli.rounds),
        other => Err(format!("unknown command `{other}` (jevbase --help)")),
    }
}

/// Open the model, saying what is happening on standard error.
fn open(cfg: &Config) -> Result<Judge, String> {
    let bytes = std::fs::metadata(&cfg.model).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "jevbase: loading {} ({:.2} GB, {}) ...",
        cfg.model.display(),
        bytes as f64 / 1e9,
        if cfg.mmap { "mapped" } else { "into RAM" }
    );
    let judge = jevbase::open(cfg)?;
    let i = &judge.engine.info;
    eprintln!(
        "jevbase: {} ({}, {:.2} B parameters) loaded in {:.1} s; {} threads, readout {}",
        i.name,
        i.desc,
        i.params as f64 / 1e9,
        i.load_seconds,
        cfg.threads,
        cfg.readout.name()
    );
    Ok(judge)
}

fn info(cfg: &Config) -> Result<ExitCode, String> {
    println!("jevbase {}", env!("CARGO_PKG_VERSION"));
    let model = match std::fs::metadata(&cfg.model) {
        Ok(m) => format!("{} ({:.2} GB)", cfg.model.display(), m.len() as f64 / 1e9),
        Err(_) => format!("{} (MISSING: set `model`)", cfg.model.display()),
    };
    println!("model     {model}");
    println!("llama.cpp {}", engine::system_info());
    println!(
        "readout   {} (avx2 available: {})",
        cfg.readout.name(),
        Readout::Avx2.available()
    );
    let sources: Vec<String> = cfg
        .sources
        .iter()
        .map(|p| p.display().to_string())
        .collect();
    println!(
        "settings  {}",
        if sources.is_empty() {
            "built-in defaults".into()
        } else {
            sources.join(", ")
        }
    );
    for (k, v) in cfg.table() {
        let doc = KEYS.iter().find(|e| e.0 == k).map_or("", |e| e.2);
        println!("  {k:<13} {v:<28} {doc}");
    }
    Ok(ExitCode::SUCCESS)
}

/// A request, or why it could not be read, with where it came from
/// (`file` or `file:line`).
type Sourced = (String, Result<Request, String>);

/// Requests from files (.json: one, .jsonl: one a line, `#` lines skipped)
/// or standard input.
fn read_requests(files: &[String]) -> Result<Vec<Sourced>, String> {
    let mut out = Vec::new();
    let sources: Vec<String> = if files.is_empty() {
        vec!["-".into()]
    } else {
        files.to_vec()
    };
    for f in &sources {
        let text = if f == "-" {
            let mut s = String::new();
            std::io::stdin()
                .read_to_string(&mut s)
                .map_err(|e| format!("standard input: {e}"))?;
            s
        } else {
            std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?
        };
        if let Ok(one) = serde_json::from_str::<Request>(&text) {
            out.push((f.clone(), Ok(one)));
            continue;
        }
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let req = serde_json::from_str::<Request>(line).map_err(|e| e.to_string());
            out.push((format!("{f}:{}", n + 1), req));
        }
    }
    if out.is_empty() {
        return Err("no requests found".into());
    }
    Ok(out)
}

fn ask(cfg: &Config, files: &[String]) -> Result<ExitCode, String> {
    let requests = read_requests(files)?;
    let judge = open(cfg)?;
    let pretty = std::io::stdout().is_terminal();
    let mut stdout = std::io::stdout().lock();
    let mut failed = 0;
    for (from, req) in requests {
        let result = req
            .map_err(|e| format!("{from}: {e}"))
            .and_then(|r| judge.evaluate(&r).map_err(|e| format!("{from}: {e}")));
        let reply = match result {
            Ok(eval) => serde_json::to_value(eval).expect("evaluation to JSON"),
            Err(e) => {
                failed += 1;
                serde_json::json!({ "error": e })
            }
        };
        let text = if pretty {
            serde_json::to_string_pretty(&reply)
        } else {
            serde_json::to_string(&reply)
        }
        .expect("JSON text");
        writeln!(stdout, "{text}").map_err(|e| e.to_string())?;
    }
    Ok(if failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

/// One bench sample: decisions answered and seconds taken.
#[derive(Default, Clone, Copy)]
struct Sample {
    decisions: usize,
    reads: usize,
    tokens: u64,
    seconds: f64,
}

impl Sample {
    fn add(&mut self, o: Sample) {
        self.decisions += o.decisions;
        self.reads += o.reads;
        self.tokens += o.tokens;
        self.seconds += o.seconds;
    }
}

fn timed(judge: &Judge, req: &Request, permutations: usize) -> Result<Sample, String> {
    let t0 = Instant::now();
    let eval = judge.evaluate(req).map_err(|e| e.to_string())?;
    let seconds = t0.elapsed().as_secs_f64();
    let reads = req
        .questions
        .values()
        .map(|q| match q.get("type").and_then(|t| t.as_str()) {
            Some("choice") => {
                let n = q
                    .get("criteria")
                    .and_then(|c| c.as_object())
                    .map_or(1, |c| c.len());
                permutations.clamp(1, n)
            }
            _ => 1,
        })
        .sum();
    Ok(Sample {
        decisions: eval.answers.len(),
        reads,
        tokens: eval.usage.input_tokens,
        seconds,
    })
}

fn bench(cfg: &Config, files: &[String], rounds: usize) -> Result<ExitCode, String> {
    let files: Vec<String> = if files.is_empty() {
        vec![
            "examples/dev_tasks.jsonl".into(),
            "examples/long_sessions.jsonl".into(),
        ]
    } else {
        files.to_vec()
    };
    for f in &files {
        if !Path::new(f).is_file() {
            return Err(format!(
                "{f} does not exist (run from the project directory, or name files)"
            ));
        }
    }
    let judge = open(cfg)?;
    let flash = match cfg.flash_attn {
        -1 => "auto",
        0 => "off",
        _ => "on",
    };
    println!(
        "jevbase bench: {}, {} threads, repack {}, flash_attn {flash}, ubatch {}, readout {}, {rounds} rounds",
        judge.engine.info.name,
        cfg.threads,
        cfg.repack,
        cfg.ubatch,
        cfg.readout.name()
    );
    println!(
        "{:<34} {:>5} {:>9} {:>6} {:>9} {:>12} {:>12} {:>12}",
        "file / mode",
        "round",
        "decisions",
        "reads",
        "seconds",
        "decisions/s",
        "ms/decision",
        "prompt tok/s"
    );
    const MODES: [&str; 3] = ["cold", "warm", "single (warm)"];
    for f in &files {
        let requests: Vec<Request> = read_requests(std::slice::from_ref(f))?
            .into_iter()
            .map(|(from, r)| r.map_err(|e| format!("{from}: {e}")))
            .collect::<Result<_, _>>()?;
        let name = Path::new(f)
            .file_name()
            .map_or(f.clone(), |n| n.to_string_lossy().into());
        let mut rates: [Vec<f64>; 3] = Default::default();
        for round in 1..=rounds {
            let mut modes = [Sample::default(); 3];
            for req in &requests {
                judge.engine.reset();
                modes[0].add(timed(&judge, req, cfg.permutations)?);
                modes[1].add(timed(&judge, req, cfg.permutations)?);
                let mut one = req.clone();
                let first = one.questions.keys().next().cloned().unwrap_or_default();
                one.questions.retain(|k, _| *k == first);
                modes[2].add(timed(&judge, &one, cfg.permutations)?);
            }
            for (m, s) in modes.iter().enumerate() {
                let rate = s.decisions as f64 / s.seconds;
                rates[m].push(rate);
                println!(
                    "{:<34} {:>5} {:>9} {:>6} {:>9.2} {:>12.2} {:>12.1} {:>12.0}",
                    format!("{name} {}", MODES[m]),
                    round,
                    s.decisions,
                    s.reads,
                    s.seconds,
                    rate,
                    1e3 / rate,
                    s.tokens as f64 / s.seconds
                );
            }
        }
        for (m, r) in rates.iter_mut().enumerate() {
            r.sort_by(f64::total_cmp);
            println!(
                "  {name} {}: decisions/s min {:.2} median {:.2} max {:.2} over {} rounds",
                MODES[m],
                r[0],
                r[r.len() / 2],
                r[r.len() - 1],
                r.len()
            );
        }
    }
    readout_bench(judge.engine.info.n_vocab);
    Ok(ExitCode::SUCCESS)
}

/// The two readouts on a row the size of this vocabulary.
fn readout_bench(n_vocab: usize) {
    let mut s = 0x9e3779b97f4a7c15u64;
    let row: Vec<f32> = (0..n_vocab)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            (-8.0 + 16.0 * u + if i % 997 == 0 { 20.0 } else { 0.0 }) as f32
        })
        .collect();
    let reps = 400;
    let exact = readout::lse(Readout::Exact, &row);
    for mode in [Readout::Exact, Readout::Avx2] {
        if !mode.available() {
            continue;
        }
        let mut best = f64::INFINITY;
        let mut v = 0.0;
        for _ in 0..3 {
            let t0 = Instant::now();
            for _ in 0..reps {
                v = readout::lse(mode, std::hint::black_box(&row));
            }
            best = best.min(t0.elapsed().as_secs_f64() / reps as f64);
        }
        println!(
            "readout {:<5} {:>8.1} us per row of {n_vocab} (best of 3 x {reps}), lse {v:.17} (diff to exact {:.1e})",
            mode.name(),
            best * 1e6,
            (v - exact).abs()
        );
    }
}
