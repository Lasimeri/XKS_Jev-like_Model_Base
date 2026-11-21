//! Settings: one table of keys, read in this order, later ones winning:
//! built-in defaults, `~/.config/jevbase/jevbase.conf`, `./jevbase.conf`,
//! the file given with `-c`, `JEVBASE_<KEY>` environment variables, and
//! command-line flags. A file holds `key = value` lines; `#` starts a
//! comment; `~` and `$HOME` in a value are the home directory.

use std::path::{Path, PathBuf};

use crate::engine::EngineConfig;
use crate::judge::JudgeConfig;
use crate::prompt::{Layout, Template};
use crate::readout::Readout;
use crate::score::Calibration;

/// Every key, its default, and what it does (`jevbase info` prints this).
pub const KEYS: &[(&str, &str, &str)] = &[
    (
        "model",
        "~/models/Qwen3.8-2B-Distill/Qwen3.8-2B-Q8_0.gguf",
        "the GGUF model file",
    ),
    ("threads", "8", "CPU threads for llama.cpp"),
    (
        "ctx",
        "16384",
        "cache cells: the state plus one round of question suffixes",
    ),
    ("batch", "2048", "tokens per llama_decode"),
    ("ubatch", "512", "tokens per physical step of a decode"),
    (
        "forks",
        "15",
        "questions decoded together, each on a copy of the state",
    ),
    (
        "permutations",
        "3",
        "rotations of a Choice's options averaged (position-bias control)",
    ),
    (
        "layout",
        "letters",
        "letters (lettered options) | jev (answers in Jev's JSON form)",
    ),
    (
        "template",
        "chatml",
        "chat template: chatml | gemma | llama3 | raw",
    ),
    (
        "repack",
        "false",
        "let llama.cpp repack weights into its interleaved CPU layouts",
    ),
    ("flash_attn", "auto", "flash attention: auto | on | off"),
    (
        "load",
        "ram",
        "ram (read the weights into memory) | mmap (map the file)",
    ),
    (
        "readout",
        "auto",
        "log-sum-exp: auto (avx2 when the CPU has AVX2+FMA) | avx2 | exact",
    ),
    (
        "temperature",
        "1.0",
        "answer temperature; temperature.<bucket> per bucket, e.g. temperature.choice:3-5",
    ),
    (
        "bind",
        "127.0.0.1:8097",
        "address `jevbase serve` listens on",
    ),
    (
        "api_keys",
        "",
        "comma-separated keys `serve` requires as `Authorization: Bearer KEY` (empty: none)",
    ),
    (
        "debug",
        "false",
        "add raw log-probabilities, tokens and latency to every answer",
    ),
    ("verbose", "false", "show llama.cpp's informational log"),
];

#[derive(Debug, Clone)]
pub struct Config {
    pub model: PathBuf,
    pub threads: i32,
    pub ctx: u32,
    pub batch: u32,
    pub ubatch: u32,
    pub forks: u32,
    pub permutations: usize,
    pub layout: Layout,
    pub template: Template,
    pub repack: bool,
    pub flash_attn: i32,
    pub mmap: bool,
    pub readout: Readout,
    pub calibration: Calibration,
    pub bind: String,
    /// Bearer keys `serve` requires (none: it answers anyone who can reach it).
    pub api_keys: Vec<String>,
    pub debug: bool,
    pub verbose: bool,
    /// The settings files that were read, in order.
    pub sources: Vec<PathBuf>,
}

impl Config {
    /// The built-in defaults (the `KEYS` table).
    pub fn defaults() -> Self {
        let mut c = Self {
            model: PathBuf::new(),
            threads: 0,
            ctx: 0,
            batch: 0,
            ubatch: 0,
            forks: 0,
            permutations: 0,
            layout: Layout::Letters,
            template: Template::ChatMl,
            repack: false,
            flash_attn: -1,
            mmap: false,
            readout: Readout::Exact,
            calibration: Calibration::default(),
            bind: String::new(),
            api_keys: Vec::new(),
            debug: false,
            verbose: false,
            sources: Vec::new(),
        };
        for (k, v, _) in KEYS {
            c.set(k, v).expect("built-in default");
        }
        c
    }

    /// Defaults, then the settings files, then `JEVBASE_*` variables.
    pub fn load(explicit: Option<&Path>) -> Result<Self, String> {
        let mut c = Self::defaults();
        let mut files = Vec::new();
        if let Some(home) = std::env::var_os("HOME") {
            files.push(PathBuf::from(home).join(".config/jevbase/jevbase.conf"));
        }
        files.push(PathBuf::from("jevbase.conf"));
        for f in files {
            if f.is_file() {
                c.read_file(&f)?;
            }
        }
        if let Some(f) = explicit {
            if !f.is_file() {
                return Err(format!("settings file {} does not exist", f.display()));
            }
            c.read_file(f)?;
        }
        for (k, _, _) in KEYS {
            let var = format!("JEVBASE_{}", k.to_ascii_uppercase());
            if let Ok(v) = std::env::var(&var) {
                c.set(k, &v).map_err(|e| format!("{var}: {e}"))?;
            }
        }
        Ok(c)
    }

    fn read_file(&mut self, path: &Path) -> Result<(), String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        for (n, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let (k, v) = line
                .split_once('=')
                .ok_or_else(|| format!("{}:{}: expected `key = value`", path.display(), n + 1))?;
            self.set(k.trim(), v.trim().trim_matches('"'))
                .map_err(|e| format!("{}:{}: {e}", path.display(), n + 1))?;
        }
        self.sources.push(path.to_path_buf());
        Ok(())
    }

    /// Set one key from text.
    pub fn set(&mut self, key: &str, value: &str) -> Result<(), String> {
        let v = expand_home(value);
        let v = v.as_str();
        match key {
            "model" => self.model = PathBuf::from(v),
            "threads" => self.threads = positive(key, v)? as i32,
            "ctx" => self.ctx = positive(key, v)?,
            "batch" => self.batch = positive(key, v)?,
            "ubatch" => self.ubatch = positive(key, v)?,
            "forks" => self.forks = positive(key, v)?,
            "permutations" => self.permutations = positive(key, v)? as usize,
            "layout" => self.layout = v.parse()?,
            "template" => self.template = v.parse()?,
            "repack" => self.repack = boolean(key, v)?,
            "flash_attn" => {
                self.flash_attn = match v {
                    "auto" => -1,
                    "on" | "true" => 1,
                    "off" | "false" => 0,
                    _ => return Err(format!("flash_attn: `{v}` is not auto, on or off")),
                }
            }
            "load" => {
                self.mmap = match v {
                    "ram" => false,
                    "mmap" => true,
                    _ => return Err(format!("load: `{v}` is not ram or mmap")),
                }
            }
            "readout" => self.readout = v.parse()?,
            "temperature" => self.calibration.default = temperature(v)?,
            "bind" => self.bind = v.to_string(),
            "api_keys" => {
                self.api_keys = v
                    .split(',')
                    .map(str::trim)
                    .filter(|k| !k.is_empty())
                    .map(str::to_string)
                    .collect()
            }
            "debug" => self.debug = boolean(key, v)?,
            "verbose" => self.verbose = boolean(key, v)?,
            _ => match key.strip_prefix("temperature.") {
                Some(bucket) if valid_bucket(bucket) => {
                    self.calibration
                        .by_bucket
                        .insert(bucket.to_string(), temperature(v)?);
                }
                Some(bucket) => {
                    return Err(format!(
                        "temperature.{bucket}: buckets are noul:2, choice:2, choice:3-5, \
                         choice:6-10, choice:11+, score:2, score:3-5, score:6-10"
                    ));
                }
                None => {
                    let known: Vec<&str> = KEYS.iter().map(|k| k.0).collect();
                    return Err(format!("unknown key `{key}` (keys: {})", known.join(", ")));
                }
            },
        }
        Ok(())
    }

    pub fn engine(&self) -> EngineConfig {
        EngineConfig {
            model: self.model.clone(),
            n_ctx: self.ctx,
            forks: self.forks,
            n_batch: self.batch,
            n_ubatch: self.ubatch,
            threads: self.threads,
            repack: self.repack,
            flash_attn: self.flash_attn,
            mmap: self.mmap,
            readout: self.readout,
            verbose: self.verbose,
        }
    }

    pub fn judge(&self) -> JudgeConfig {
        JudgeConfig {
            template: self.template,
            layout: self.layout,
            calibration: self.calibration.clone(),
            permutations: self.permutations,
            debug: self.debug,
        }
    }

    /// The effective settings as (key, value) pairs.
    pub fn table(&self) -> Vec<(&'static str, String)> {
        let flash = match self.flash_attn {
            -1 => "auto",
            0 => "off",
            _ => "on",
        };
        let mut temps = format!("{}", self.calibration.default);
        let mut buckets: Vec<_> = self.calibration.by_bucket.iter().collect();
        buckets.sort_by(|a, b| a.0.cmp(b.0));
        for (b, t) in buckets {
            temps.push_str(&format!(", {b}={t}"));
        }
        vec![
            ("model", self.model.display().to_string()),
            ("threads", self.threads.to_string()),
            ("ctx", self.ctx.to_string()),
            ("batch", self.batch.to_string()),
            ("ubatch", self.ubatch.to_string()),
            ("forks", self.forks.to_string()),
            ("permutations", self.permutations.to_string()),
            ("layout", format!("{:?}", self.layout).to_ascii_lowercase()),
            (
                "template",
                format!("{:?}", self.template).to_ascii_lowercase(),
            ),
            ("repack", self.repack.to_string()),
            ("flash_attn", flash.into()),
            ("load", if self.mmap { "mmap" } else { "ram" }.into()),
            ("readout", self.readout.name().into()),
            ("temperature", temps),
            ("bind", self.bind.clone()),
            ("api_keys", format!("{} set", self.api_keys.len())),
            ("debug", self.debug.to_string()),
        ]
    }
}

fn expand_home(v: &str) -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    let v = match v.strip_prefix("~/") {
        Some(rest) => format!("{home}/{rest}"),
        None => v.to_string(),
    };
    v.replace("$HOME", &home)
}

fn positive(key: &str, v: &str) -> Result<u32, String> {
    match v.parse::<u32>() {
        Ok(n) if n > 0 => Ok(n),
        _ => Err(format!("{key}: `{v}` is not a whole number above 0")),
    }
}

fn boolean(key: &str, v: &str) -> Result<bool, String> {
    match v {
        "true" | "on" | "yes" | "1" => Ok(true),
        "false" | "off" | "no" | "0" => Ok(false),
        _ => Err(format!("{key}: `{v}` is not true or false")),
    }
}

fn temperature(v: &str) -> Result<f64, String> {
    match v.parse::<f64>() {
        Ok(t) if t > 0.0 && t.is_finite() => Ok(t),
        _ => Err(format!("temperature: `{v}` is not a number above 0")),
    }
}

fn valid_bucket(b: &str) -> bool {
    matches!(
        b,
        "noul:2"
            | "choice:2"
            | "choice:3-5"
            | "choice:6-10"
            | "choice:11+"
            | "score:2"
            | "score:3-5"
            | "score:6-10"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_parse_and_keys_are_settable() {
        let c = Config::defaults();
        assert_eq!(c.threads, 8);
        assert_eq!(c.forks, 15);
        assert_eq!(c.permutations, 3);
        assert!(!c.mmap);
        let mut c = c;
        c.set("temperature.choice:3-5", "0.6").unwrap();
        assert_eq!(c.calibration.temperature("choice", 4), 0.6);
        assert!(c.set("threads", "0").is_err());
        assert!(c.set("nope", "1").unwrap_err().contains("keys:"));
        assert!(c.set("temperature.choice:7", "1").is_err());
    }
}
