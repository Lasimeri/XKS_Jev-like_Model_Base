//! XKS_Jev-like_Model_Base: typed judgments (noul / choice / score) read
//! from a local LLM's next-token distribution, on the CPU with llama.cpp.
//!
//! The method is Intel-Phi-Jev's (`xks`, the oracle this crate is checked
//! against): render each question so that its answer is the next token,
//! prefill the shared state once, fork it per question, read the logits of
//! the answer labels after the cue, and turn them into typed answers.
//!
//! - [`protocol`]: the System One request and response (wire format).
//! - [`prompt`]: questions rendered as prompts whose next token is the answer.
//! - [`engine`]: llama.cpp in process; prefill, forks, trie, readout.
//! - [`readout`]: log-sum-exp over the vocabulary (exact, or AVX2 + FMA).
//! - [`score`]: softmax, temperature, confidence, expected level.
//! - [`judge`]: a request in, typed answers out.
//! - [`config`]: every setting, from files, environment and flags.
//! - [`server`]: the same wire over HTTP (`jevbase serve`).
//! - [`tui`]: the terminal interface (`jevbase`).

pub mod config;
pub mod engine;
mod ffi;
pub mod judge;
pub mod prompt;
pub mod protocol;
pub mod readout;
pub mod score;
pub mod server;
pub mod tui;

pub use config::Config;
pub use engine::{Engine, EngineError};
pub use judge::Judge;
pub use protocol::{Answer, Evaluation, Request};

/// Open the model and the judge with these settings.
pub fn open(cfg: &Config) -> Result<Judge, String> {
    let engine = Engine::open(&cfg.engine())?;
    Ok(Judge::new(engine, cfg.judge()))
}
