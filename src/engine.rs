//! The engine: llama.cpp in process, reading the next-token distribution
//! after the answer cue for every question of a request at once. The model
//! never generates; one forward pass per question reads it.
//!
//! The decode plan is Intel-Phi-Jev's ARTICHOKE (`src/artichoke/mod.rs`),
//! step for step, so one llama.cpp build gives both the same numbers:
//! - The state (the prefix every rendered question shares) is prefilled once
//!   on sequence 0 and kept between requests; a request on the same state
//!   costs only its questions.
//! - Questions whose labels are one token each are read in rounds of up to
//!   `forks`: sequence 0 is copied to sequences 1..=k (`llama_memory_seq_cp`,
//!   which on a hybrid model copies the recurrent state too), every suffix
//!   goes through one batch, and one output row per question is read: the
//!   logits after the last token of its suffix.
//! - A question whose labels are several tokens is read as a trie: its suffix
//!   on sequence 1 read at the end (the root), then every distinct proper
//!   prefix of its labels forked from there and read one step further; a
//!   label's log-probability is the sum along its path.
//! - Each row is read by `readout`: log-probabilities over the whole
//!   vocabulary, kept only at the label tokens.

use std::collections::{BTreeMap, HashMap};
use std::ffi::{CStr, CString};
use std::fmt;
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;
use std::sync::{Mutex, Once};
use std::time::Instant;

use crate::ffi::{self, LlamaContext, LlamaModel};
use crate::prompt::Segs;
use crate::readout::{self, Readout};

/// How to open the engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// The GGUF file.
    pub model: PathBuf,
    /// Cells of the unified cache: the state plus one round of suffixes.
    pub n_ctx: u32,
    /// Questions forked from the state and decoded together.
    pub forks: u32,
    /// Tokens handed to one `llama_decode`.
    pub n_batch: u32,
    /// Tokens in one physical step of a decode.
    pub n_ubatch: u32,
    /// CPU threads for llama.cpp.
    pub threads: i32,
    /// Let llama.cpp repack weights into its interleaved CPU layouts.
    pub repack: bool,
    /// Flash attention: -1 auto, 0 off, 1 on.
    pub flash_attn: i32,
    /// Memory-map the file instead of reading it into process memory.
    pub mmap: bool,
    pub readout: Readout,
    /// Show llama.cpp's informational log.
    pub verbose: bool,
}

/// Why a request was not answered.
#[derive(Debug, Clone)]
pub enum EngineError {
    /// The request cannot be read as asked (the caller's to fix).
    Rejected(String),
    /// The engine failed.
    Failed(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rejected(m) => write!(f, "rejected: {m}"),
            Self::Failed(m) => write!(f, "engine failed: {m}"),
        }
    }
}

impl std::error::Error for EngineError {}

/// One question's reading: the log-probability of each label, and its cost.
#[derive(Debug, Clone)]
pub struct Scored {
    pub logprobs: Vec<f64>,
    /// Tokens this question put through the model.
    pub evaluated: u64,
    /// State tokens it reused from the cache.
    pub cached: u64,
    pub latency_ms: f64,
}

/// What the loaded model is.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// File stem, the name answers carry.
    pub name: String,
    /// llama.cpp's description (architecture, size, type).
    pub desc: String,
    pub path: PathBuf,
    pub bytes: u64,
    pub params: u64,
    pub n_vocab: usize,
    /// Recurrent state anywhere (pure or hybrid model).
    pub recurrent: bool,
    pub load_seconds: f64,
}

/// TypeSafe's context limits (https://docs.typesafe.ai/models, "Context
/// length"): the state plus any one question within the per-question
/// context, the state plus all questions within the per-request context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub per_question: usize,
    pub per_request: usize,
}

impl Limits {
    /// The documented defaults: 32k per question, 64k per request.
    pub const DEFAULT: Limits = Limits {
        per_question: 32_000,
        per_request: 64_000,
    };
    /// With `TypeSafe-Danger-Zone-Long-Context: true`: both 262k (2^18).
    pub const LONG_CONTEXT: Limits = Limits {
        per_question: 262_144,
        per_request: 262_144,
    };
}

/// One token of a decode plan: (token, position, sequence, output slot).
type Step = (i32, i32, i32, Option<usize>);

/// What only one request may touch at a time.
struct Inner {
    ctx: *mut LlamaContext,
    /// Tokens sequence 0 holds (the last state prefilled).
    cached: Vec<i32>,
    /// Labels already tokenized.
    labels: HashMap<String, Vec<i32>>,
}

pub struct Engine {
    model: *mut LlamaModel,
    inner: Mutex<Inner>,
    forks: usize,
    n_ctx: usize,
    n_batch: usize,
    readout: Readout,
    pub info: ModelInfo,
}

// SAFETY: the context is only touched under `inner`'s lock; the model and
// its vocabulary are read-only after loading and llama.cpp allows sharing.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

static INIT: Once = Once::new();

/// llama.cpp's report of the CPU features its kernels use.
pub fn system_info() -> String {
    // SAFETY: returns a static NUL-terminated string.
    unsafe { CStr::from_ptr(ffi::jl_system_info()) }
        .to_string_lossy()
        .trim()
        .to_string()
}

impl Engine {
    pub fn open(cfg: &EngineConfig) -> Result<Self, String> {
        if !cfg.model.is_file() {
            return Err(format!(
                "model not found: {} (set `model` in jevbase.conf, or pass --model)",
                cfg.model.display()
            ));
        }
        if cfg.forks == 0 || cfg.threads < 1 {
            return Err("forks and threads must be at least 1".into());
        }
        let verbose = cfg.verbose;
        // SAFETY: once per process, before any model is loaded.
        INIT.call_once(|| unsafe { ffi::jl_init(verbose as i32) });
        let t0 = Instant::now();
        let path = CString::new(cfg.model.as_os_str().as_bytes()).map_err(|e| e.to_string())?;
        // SAFETY: a valid path; NULL is checked.
        let model =
            unsafe { ffi::jl_model_load(path.as_ptr(), cfg.mmap as i32, cfg.repack as i32) };
        if model.is_null() {
            return Err(format!("llama.cpp could not load {}", cfg.model.display()));
        }
        // At most one output per fork in a step (the last token of each
        // suffix), plus the trie's root.
        let n_seq = cfg.forks + 1;
        // SAFETY: model is live; NULL is checked.
        let ctx = unsafe {
            ffi::jl_context_new(
                model,
                cfg.n_ctx,
                cfg.n_batch,
                cfg.n_ubatch,
                n_seq,
                n_seq,
                cfg.threads,
                cfg.flash_attn,
            )
        };
        if ctx.is_null() {
            // SAFETY: model is live and no context holds it.
            unsafe { ffi::jl_model_free(model) };
            return Err("could not create a context (lower ctx or forks?)".into());
        }
        let mut desc = [0 as std::ffi::c_char; 256];
        // SAFETY: model is live; the buffer is 256 bytes and NUL-terminated.
        let info = unsafe {
            ffi::jl_model_desc(model, desc.as_mut_ptr(), desc.len());
            ModelInfo {
                name: cfg
                    .model
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                desc: CStr::from_ptr(desc.as_ptr()).to_string_lossy().into_owned(),
                path: cfg.model.clone(),
                bytes: ffi::jl_model_size(model),
                params: ffi::jl_model_n_params(model),
                n_vocab: ffi::jl_n_vocab(model) as usize,
                recurrent: ffi::jl_model_recurrent(model) != 0,
                load_seconds: t0.elapsed().as_secs_f64(),
            }
        };
        Ok(Self {
            model,
            inner: Mutex::new(Inner {
                ctx,
                cached: Vec::new(),
                labels: HashMap::new(),
            }),
            forks: cfg.forks as usize,
            n_ctx: cfg.n_ctx as usize,
            n_batch: cfg.n_batch as usize,
            readout: cfg.readout,
            info,
        })
    }

    /// Forget the kept state: the next request prefills from scratch.
    pub fn reset(&self) {
        let mut g = self.inner.lock().expect("engine lock");
        // SAFETY: the context is live and locked.
        unsafe { ffi::jl_mem_clear(g.ctx) };
        g.cached.clear();
    }

    /// Read every (suffix, labels) item against the shared `prefix`: one
    /// `Scored` per item, its log-probabilities in label order. A request
    /// past `limits` (or this context) is refused before any work.
    pub fn score_many(
        &self,
        prefix: &Segs,
        items: &[(Segs, Vec<String>)],
        limits: Limits,
    ) -> Result<Vec<Scored>, EngineError> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let cands = {
            let mut g = self.inner.lock().expect("engine lock");
            items
                .iter()
                .map(|(_, labels)| {
                    labels
                        .iter()
                        .map(|l| self.label_tokens(&mut g, l))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        let (prompts, shared) = self.prepare(prefix, items)?;
        check_limits(&prompts, &cands, shared, self.n_ctx, limits)?;
        self.fork_and_read(&prompts, &cands, shared)
    }

    /// Count the tokens of a text as user text (for display).
    pub fn count_tokens(&self, text: &str) -> Result<usize, EngineError> {
        Ok(self.tokenize_with(text, false, false)?.len())
    }

    fn tokenize_with(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<i32>, EngineError> {
        let len = i32::try_from(text.len())
            .map_err(|_| EngineError::Rejected("text too long to tokenize".into()))?;
        let mut buf = vec![0i32; text.len() + 8];
        let call = |buf: &mut Vec<i32>| {
            // SAFETY: `text` is valid for `len` bytes and `buf` for its length.
            unsafe {
                ffi::jl_tokenize(
                    self.model,
                    text.as_ptr().cast(),
                    len,
                    add_special as i32,
                    parse_special as i32,
                    buf.as_mut_ptr(),
                    buf.len() as i32,
                )
            }
        };
        let mut n = call(&mut buf);
        if n < 0 {
            buf.resize((-n) as usize, 0);
            n = call(&mut buf);
        }
        if n < 0 {
            return Err(EngineError::Failed("the tokenizer failed".into()));
        }
        buf.truncate(n as usize);
        Ok(buf)
    }

    /// A rendered prompt, segment by segment: template text with special
    /// tokens parsed, user text without, the start token before the first.
    fn tokenize_segs(&self, segs: &Segs) -> Result<Vec<i32>, EngineError> {
        let mut out = Vec::new();
        for (i, s) in segs.0.iter().enumerate() {
            out.extend(self.tokenize_with(&s.text, i == 0, !s.user)?);
        }
        // A template that writes its start token itself after the tokenizer
        // added one: keep one.
        // SAFETY: the model is live.
        let bos = unsafe { ffi::jl_bos(self.model) };
        if out.len() >= 2 && out[0] == bos && out[1] == bos {
            out.remove(0);
        }
        Ok(out)
    }

    /// A label's tokens as it follows the answer cue (no start token, no
    /// special tokens), cached.
    fn label_tokens(&self, inner: &mut Inner, label: &str) -> Result<Vec<i32>, EngineError> {
        if let Some(t) = inner.labels.get(label) {
            return Ok(t.clone());
        }
        let toks = self.tokenize_with(label, false, false)?;
        if toks.is_empty() {
            return Err(EngineError::Rejected(format!(
                "label {label:?} has no tokens"
            )));
        }
        inner.labels.insert(label.to_string(), toks.clone());
        Ok(toks)
    }

    /// The log-probabilities of `ids` in output row `i` of the last decode.
    fn read(&self, ctx: *mut LlamaContext, i: i32, ids: &[i32]) -> Result<Vec<f64>, EngineError> {
        // SAFETY: the context is live and locked by the caller.
        let p = unsafe { ffi::jl_logits(ctx, i) };
        if p.is_null() {
            return Err(EngineError::Failed(format!("no logits for output row {i}")));
        }
        // SAFETY: llama.cpp returns n_vocab floats, valid until the next decode.
        let row = unsafe { std::slice::from_raw_parts(p, self.info.n_vocab) };
        Ok(readout::logprobs(self.readout, row, ids))
    }

    /// Decode `steps` in windows of n_batch, calling `out(slot, row)` for
    /// every output right after the window that produced it.
    fn decode(
        &self,
        ctx: *mut LlamaContext,
        steps: &[Step],
        mut out: impl FnMut(usize, i32) -> Result<(), EngineError>,
    ) -> Result<(), EngineError> {
        let mut tok = Vec::with_capacity(self.n_batch);
        let mut pos = Vec::with_capacity(self.n_batch);
        let mut seq = Vec::with_capacity(self.n_batch);
        let mut flag = Vec::with_capacity(self.n_batch);
        for window in steps.chunks(self.n_batch) {
            tok.clear();
            pos.clear();
            seq.clear();
            flag.clear();
            for &(t, p, s, slot) in window {
                tok.push(t);
                pos.push(p);
                seq.push(s);
                flag.push(slot.is_some() as i8);
            }
            // SAFETY: four arrays of window.len() entries; the context is
            // live and locked by the caller.
            let rc = unsafe {
                ffi::jl_decode(
                    ctx,
                    window.len() as i32,
                    tok.as_ptr(),
                    pos.as_ptr(),
                    seq.as_ptr(),
                    flag.as_ptr(),
                )
            };
            if rc != 0 {
                return Err(EngineError::Failed(format!("llama_decode returned {rc}")));
            }
            for (w, &(.., slot)) in window.iter().enumerate() {
                if let Some(s) = slot {
                    out(s, w as i32)?;
                }
            }
        }
        Ok(())
    }

    fn fork_and_read(
        &self,
        prompts: &[Vec<i32>],
        cands: &[Vec<Vec<i32>>],
        shared: usize,
    ) -> Result<Vec<Scored>, EngineError> {
        let t0 = Instant::now();
        let mut g = self.inner.lock().expect("engine lock");
        let ctx = g.ctx;
        let session = &prompts[0][..shared];

        // Reuse what sequence 0 already holds. A recurrent state cannot be
        // cut back, so a partial match there means starting over.
        let reuse = g
            .cached
            .iter()
            .zip(session)
            .take_while(|(a, b)| a == b)
            .count();
        if reuse < g.cached.len() {
            // SAFETY: the context is live and locked.
            let cut = unsafe { ffi::jl_seq_rm(ctx, 0, reuse as i32, -1) } != 0;
            if cut {
                g.cached.truncate(reuse);
            } else {
                // SAFETY: as above.
                unsafe { ffi::jl_mem_clear(ctx) };
                g.cached.clear();
            }
        }
        let start = g.cached.len();
        if shared > start {
            let plan: Vec<Step> = session[start..]
                .iter()
                .enumerate()
                .map(|(i, &t)| (t, (start + i) as i32, 0, None))
                .collect();
            if let Err(e) = self.decode(ctx, &plan, |_, _| Ok(())) {
                // SAFETY: as above.
                unsafe { ffi::jl_mem_clear(ctx) };
                g.cached.clear();
                return Err(e);
            }
        }
        g.cached = session.to_vec();

        // One-token labels in rounds of forks, as many as there are
        // sequences and cells for; several-token labels one question at a
        // time as a trie.
        let single: Vec<usize> = (0..prompts.len())
            .filter(|&n| cands[n].iter().all(|c| c.len() == 1))
            .collect();
        let firsts: Vec<Vec<i32>> = cands
            .iter()
            .map(|cs| cs.iter().map(|c| c[0]).collect())
            .collect();
        let mut readings: Vec<Option<(Vec<f64>, usize)>> = vec![None; prompts.len()];
        let mut i = 0;
        while i < single.len() {
            let mut j = i;
            let mut cells = shared;
            while j < single.len()
                && j - i < self.forks
                && cells + prompts[single[j]].len() - shared <= self.n_ctx
            {
                cells += prompts[single[j]].len() - shared;
                j += 1;
            }
            if j == i {
                return Err(EngineError::Rejected(format!(
                    "a prompt of {} tokens does not fit a context of {}",
                    prompts[single[i]].len(),
                    self.n_ctx
                )));
            }
            let mut plan = Vec::new();
            for (k, &n) in single[i..j].iter().enumerate() {
                let seq = (k + 1) as i32;
                // SAFETY: the context is live and locked.
                unsafe { ffi::jl_seq_cp(ctx, 0, seq, -1, -1) };
                let suffix = &prompts[n][shared..];
                for (m, &t) in suffix.iter().enumerate() {
                    let slot = (m + 1 == suffix.len()).then_some(n);
                    plan.push((t, (shared + m) as i32, seq, slot));
                }
            }
            let result = self.decode(ctx, &plan, |slot, row| {
                let r = self.read(ctx, row, &firsts[slot])?;
                readings[slot] = Some((r, prompts[slot].len() - shared));
                Ok(())
            });
            for k in 0..(j - i) {
                // SAFETY: as above.
                unsafe { ffi::jl_seq_rm(ctx, (k + 1) as i32, -1, -1) };
            }
            result?;
            i = j;
        }
        for n in 0..prompts.len() {
            if readings[n].is_none() {
                readings[n] = Some(self.read_trie(ctx, &prompts[n], shared, &cands[n])?);
            }
        }
        let each = t0.elapsed().as_secs_f64() * 1e3 / prompts.len() as f64;
        readings
            .into_iter()
            .enumerate()
            .map(|(n, r)| {
                let (logprobs, evaluated) =
                    r.ok_or_else(|| EngineError::Failed(format!("no output for prompt {n}")))?;
                Ok(Scored {
                    logprobs,
                    // The state is charged to the first question.
                    evaluated: (evaluated + if n == 0 { shared - start } else { 0 }) as u64,
                    cached: if n == 0 { start } else { shared } as u64,
                    latency_ms: each,
                })
            })
            .collect()
    }

    /// One question whose labels are several tokens: its suffix forked from
    /// the state onto sequence 1 and read at its end (the root), then every
    /// distinct proper prefix of the labels forked from sequence 1 and read
    /// one step further, in rounds on the other sequences. Returns the
    /// labels' log-probabilities and the tokens evaluated.
    fn read_trie(
        &self,
        ctx: *mut LlamaContext,
        prompt: &[i32],
        shared: usize,
        cands: &[Vec<i32>],
    ) -> Result<(Vec<f64>, usize), EngineError> {
        // What each node (a label prefix) needs read: the next tokens of the
        // labels through it.
        let mut needs: BTreeMap<Vec<i32>, Vec<i32>> = BTreeMap::new();
        for c in cands {
            for d in 0..c.len() {
                let e = needs.entry(c[..d].to_vec()).or_default();
                if !e.contains(&c[d]) {
                    e.push(c[d]);
                }
            }
        }
        let mut read: BTreeMap<Vec<i32>, Vec<f64>> = BTreeMap::new();
        let len = prompt.len();
        let mut evaluated = len - shared;

        // The root on sequence 1.
        // SAFETY: the context is live and locked by the caller.
        unsafe { ffi::jl_seq_cp(ctx, 0, 1, -1, -1) };
        let suffix = &prompt[shared..];
        let plan: Vec<Step> = suffix
            .iter()
            .enumerate()
            .map(|(m, &t)| {
                (
                    t,
                    (shared + m) as i32,
                    1,
                    (m + 1 == suffix.len()).then_some(0),
                )
            })
            .collect();
        let root_ids = needs.get(&Vec::new()).cloned().unwrap_or_default();
        let mut root = None;
        let r = self.decode(ctx, &plan, |_, row| {
            root = Some(self.read(ctx, row, &root_ids)?);
            Ok(())
        });
        if let Err(e) = r {
            // SAFETY: as above.
            unsafe { ffi::jl_seq_rm(ctx, 1, -1, -1) };
            return Err(e);
        }
        read.insert(
            Vec::new(),
            root.ok_or_else(|| EngineError::Failed("no root row".into()))?,
        );

        // The probes, forked from sequence 1 in rounds of the free sequences.
        let probes: Vec<Vec<i32>> = needs.keys().filter(|p| !p.is_empty()).cloned().collect();
        if self.forks < 2 && !probes.is_empty() {
            // SAFETY: as above.
            unsafe { ffi::jl_seq_rm(ctx, 1, -1, -1) };
            return Err(EngineError::Rejected(
                "labels of several tokens need forks = 2 or more".into(),
            ));
        }
        let free = self.forks - 1;
        let mut result = Ok(());
        for round in probes.chunks(free) {
            let mut plan = Vec::new();
            for (k, p) in round.iter().enumerate() {
                let seq = (k + 2) as i32;
                // SAFETY: as above.
                unsafe { ffi::jl_seq_cp(ctx, 1, seq, -1, -1) };
                for (m, &t) in p.iter().enumerate() {
                    let slot = (m + 1 == p.len()).then_some(k);
                    plan.push((t, (len + m) as i32, seq, slot));
                }
                evaluated += p.len();
            }
            let mut rows = vec![Vec::new(); round.len()];
            result = self.decode(ctx, &plan, |slot, row| {
                rows[slot] = self.read(ctx, row, &needs[&round[slot]])?;
                Ok(())
            });
            for k in 0..round.len() {
                // SAFETY: as above.
                unsafe { ffi::jl_seq_rm(ctx, (k + 2) as i32, -1, -1) };
            }
            if result.is_err() {
                break;
            }
            for (p, r) in round.iter().zip(rows) {
                read.insert(p.clone(), r);
            }
        }
        // SAFETY: as above.
        unsafe { ffi::jl_seq_rm(ctx, 1, -1, -1) };
        result?;
        let logprobs = cands
            .iter()
            .map(|c| {
                (0..c.len())
                    .map(|d| {
                        let node = &c[..d];
                        let at = needs[node].iter().position(|&t| t == c[d]).expect("need");
                        read[node][at]
                    })
                    .sum()
            })
            .collect();
        Ok((logprobs, evaluated))
    }

    /// Every item's whole prompt, and where they fork: where their tokens
    /// part, never past the end of the state itself (the prefix's own
    /// tokens), so the kept state is the state alone and reusable by the
    /// next request on it.
    fn prepare(
        &self,
        prefix: &Segs,
        items: &[(Segs, Vec<String>)],
    ) -> Result<(Vec<Vec<i32>>, usize), EngineError> {
        let prompts = items
            .iter()
            .map(|(s, _)| self.tokenize_segs(&prefix.concat(s)))
            .collect::<Result<Vec<_>, _>>()?;
        if prompts.iter().any(|p| p.is_empty()) {
            return Err(EngineError::Rejected("empty prompt".into()));
        }
        let own = self.tokenize_segs(prefix)?;
        let session_end = own
            .iter()
            .zip(&prompts[0])
            .take_while(|(a, b)| a == b)
            .count();
        let shared = fork_point(&prompts).min(session_end);
        Ok((prompts, shared))
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let g = self.inner.get_mut().expect("engine lock");
        // SAFETY: both are live and nothing else holds them.
        unsafe {
            ffi::jl_context_free(g.ctx);
            ffi::jl_model_free(self.model);
        }
    }
}

/// The longest prefix every prompt shares, short of each prompt's last
/// token (every fork decodes at least one).
fn fork_point(prompts: &[Vec<i32>]) -> usize {
    let mut shared = prompts[0].len() - 1;
    for p in &prompts[1..] {
        let lcp = prompts[0].iter().zip(p).take_while(|(a, b)| a == b).count();
        shared = shared.min(lcp).min(p.len() - 1);
    }
    shared
}

/// Refuse, before any prefill, a request that cannot be read: past
/// TypeSafe's context limits, or a question that with its longest label
/// does not fit this server's context.
fn check_limits(
    prompts: &[Vec<i32>],
    cands: &[Vec<Vec<i32>>],
    shared: usize,
    n_ctx: usize,
    limits: Limits,
) -> Result<(), EngineError> {
    let longest = prompts
        .iter()
        .zip(cands)
        .map(|(p, cs)| {
            p.len()
                + cs.iter()
                    .map(|c| c.len().saturating_sub(1))
                    .max()
                    .unwrap_or(0)
        })
        .max()
        .unwrap_or(0);
    let request = shared + prompts.iter().map(|p| p.len() - shared).sum::<usize>();
    if longest > limits.per_question {
        return Err(EngineError::Rejected(format!(
            "the state and a question come to {longest} tokens, past the per-question context of {}",
            limits.per_question
        )));
    }
    if request > limits.per_request {
        return Err(EngineError::Rejected(format!(
            "the request comes to {request} tokens, past the per-request context of {}",
            limits.per_request
        )));
    }
    if longest > n_ctx {
        return Err(EngineError::Rejected(format!(
            "the state and a question come to {longest} tokens, past this context of {n_ctx} \
             (raise `ctx`)"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_requests_are_refused_before_any_prefill() {
        let p = |n: usize| vec![1i32; n];
        let one = vec![vec![1i32]];
        let d = Limits::DEFAULT;
        assert!(check_limits(&[p(100), p(120)], &[one.clone(), one.clone()], 90, 16384, d).is_ok());
        let e =
            check_limits(&[p(20_000)], std::slice::from_ref(&one), 19_990, 16384, d).unwrap_err();
        assert!(matches!(e, EngineError::Rejected(ref m) if m.contains("ctx")));
        assert!(check_limits(&[p(33_000)], std::slice::from_ref(&one), 32_990, 65_536, d).is_err());
        let many: Vec<_> = (0..5).map(|_| p(31_000)).collect();
        let cands: Vec<_> = (0..5).map(|_| one.clone()).collect();
        assert!(matches!(
            check_limits(&many, &cands, 10_000, 131_072, d),
            Err(EngineError::Rejected(ref m)) if m.contains("per-request context of 64000")
        ));
        // Long context lifts both documented limits; this context still binds.
        let long = Limits::LONG_CONTEXT;
        assert!(
            check_limits(
                &[p(33_000)],
                std::slice::from_ref(&one),
                32_990,
                65_536,
                long
            )
            .is_ok()
        );
        assert!(check_limits(&many, &cands, 10_000, 131_072, long).is_ok());
        let label = vec![vec![1i32; 10]];
        assert!(check_limits(&[p(16_380)], &[label], 16_000, 16_384, d).is_err());
    }

    #[test]
    fn fork_point_stops_short_of_every_last_token() {
        assert_eq!(fork_point(&[vec![1, 2, 3, 4], vec![1, 2, 3, 5]]), 3);
        assert_eq!(fork_point(&[vec![1, 2, 3], vec![1, 2, 3]]), 2);
        assert_eq!(fork_point(&[vec![1, 2], vec![9]]), 0);
    }
}
