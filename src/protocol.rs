//! Wire types of the System One request and response, as TypeSafe documents
//! `POST /v1/systemone` (OpenAPI 0.2.0, https://docs.typesafe.ai/api/system-one,
//! read 2026-11-21), so TypeSafe's SDKs, Intel-Phi-Jev's `xks` and Mechanical
//! Jev's `mjev` talk to this crate's server and client unchanged.
//!
//! A request is a `model` (a name or alias), a `state` (free text, or any
//! JSON, rendered as pretty JSON) and named typed questions: `noul` (yes/no),
//! `choice` (one of 1 to 255 named options) and `score` (a level on an
//! ordered scale of 1 to 10). `instructions` is optional on every question.
//! The wire's `box` questions and image `assets` need a vision model; the
//! server refuses them (`validate`).

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A request: the model asked for, the situation (`state`) and named typed
/// questions.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Request {
    /// Model name or alias (required on the wire; `validate` checks it).
    /// A client fills its default when it is None.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Free text, or any JSON object or array (rendered as pretty JSON).
    pub state: Value,
    /// Question id -> question, in the order given.
    pub questions: Map<String, Value>,
}

/// One typed judgment asked of the model.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// Yes/no ("noul" is TypeSafe's name for the primitive).
    Noul {
        #[serde(default)]
        instructions: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
    /// One of N named options; each value is an optional description.
    Choice {
        #[serde(default)]
        instructions: Value,
        criteria: Map<String, Value>,
    },
    /// An ordered scale; `criteria[i]` describes level `i`.
    Score {
        #[serde(default)]
        instructions: Value,
        criteria: Vec<Value>,
    },
}

/// What `true` and `false` mean for a Noul (each optional).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct NoulCriteria {
    #[serde(rename = "true", default)]
    pub is_true: Value,
    #[serde(rename = "false", default)]
    pub is_false: Value,
}

/// One typed answer.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul {
        /// P(yes) in [0, 1].
        noul: f64,
    },
    Choice {
        choice: String,
        probabilities: Map<String, Value>,
        confidence: f64,
    },
    Score {
        /// Probability-weighted mean level index.
        score: f64,
        legend: Map<String, Value>,
        probabilities: Map<String, Value>,
        confidence: f64,
    },
}

/// Tokens the model evaluated for a request (cached session tokens excluded).
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// The response envelope.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Evaluation {
    pub model: String,
    pub answers: Map<String, Value>,
    pub usage: Usage,
    /// Per-question diagnostics (raw log-probabilities, label mass, tokens,
    /// latency); present only when debug output is on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub debug: Option<Value>,
}

impl Evaluation {
    /// The typed answer to question `id`.
    pub fn answer(&self, id: &str) -> Option<Answer> {
        self.answers
            .get(id)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }
}

/// An `instructions` or `criteria` value as prompt text: strings verbatim,
/// anything else as compact JSON, null as nothing.
pub fn text_of(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// `state` as prompt text: strings verbatim, structures as pretty JSON.
pub fn render_state(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => serde_json::to_string_pretty(other).unwrap_or_default(),
    }
}

/// Parse and check every question, in request order.
pub fn parse_questions(raw: &Map<String, Value>) -> Result<Vec<(String, Question)>, String> {
    let mut out = Vec::with_capacity(raw.len());
    for (id, v) in raw {
        let q: Question =
            serde_json::from_value(v.clone()).map_err(|e| format!("question `{id}`: {e}"))?;
        match &q {
            Question::Choice { criteria, .. } if criteria.is_empty() => {
                return Err(format!("question `{id}`: choice needs at least one option"));
            }
            Question::Score { criteria, .. } if criteria.is_empty() || criteria.len() > 10 => {
                return Err(format!("question `{id}`: score needs 1 to 10 levels"));
            }
            _ => {}
        }
        out.push((id.clone(), q));
    }
    if out.is_empty() {
        return Err("no questions".into());
    }
    Ok(out)
}
