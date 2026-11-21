//! XKS_Jev-like_Model_Base: typed judgments (noul / choice / score) read
//! from a local LLM's next-token distribution, on the CPU with llama.cpp.
//!
//! The method is Intel-Phi-Jev's (`xks`, the oracle this crate is checked
//! against): render each question so that its answer is the next token,
//! prefill the shared state once, fork it per question, read the logits of
//! the answer labels after the cue, and turn them into typed answers.
//!
//! Always built (std and serde only):
//! - [`protocol`]: the System One request and response (wire format).
//! - [`client`]: ask a running `jevbase serve` over HTTP.
//! - [`prompt`]: questions rendered as prompts whose next token is the answer.
//! - [`readout`]: log-sum-exp over the vocabulary (exact, or AVX2 + FMA).
//! - [`score`]: softmax, temperature, confidence, expected level.
//!
//! With the `engine` feature (the default; llama.cpp linked statically):
//! - [`engine`]: llama.cpp in process; prefill, forks, trie, readout.
//! - [`judge`]: a request in, typed answers out.
//! - [`config`]: every setting, from files, environment and flags.
//! - [`server`]: the same wire over HTTP (`jevbase serve`).
//! - [`tui`]: the terminal interface (`jevbase`).

pub mod client;
pub mod prompt;
pub mod protocol;
pub mod readout;
pub mod score;

#[cfg(feature = "engine")]
pub mod config;
#[cfg(feature = "engine")]
pub mod engine;
#[cfg(feature = "engine")]
mod ffi;
#[cfg(feature = "engine")]
pub mod judge;
#[cfg(feature = "engine")]
pub mod server;
#[cfg(feature = "engine")]
pub mod tui;

pub use client::{Client, ClientError};
pub use protocol::{Answer, Evaluation, Request};

#[cfg(feature = "engine")]
pub use config::Config;
#[cfg(feature = "engine")]
pub use engine::{Engine, EngineError};
#[cfg(feature = "engine")]
pub use judge::Judge;

/// Open the model and the judge with these settings.
#[cfg(feature = "engine")]
pub fn open(cfg: &Config) -> Result<Judge, String> {
    let engine = Engine::open(&cfg.engine())?;
    Ok(Judge::new(engine, cfg.judge()))
}
