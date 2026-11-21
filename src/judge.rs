//! The judge: render each question against the shared state prefix, read
//! the labels with the engine, average a Choice over rotations of its
//! options, calibrate, and build the typed answers.
//! Ported from Intel-Phi-Jev's `src/judge.rs`.

use serde_json::{Map, Value, json};

use crate::engine::{Engine, EngineError, Limits};
use crate::prompt::{self, Layout, MAX_OPTIONS, Template};
use crate::protocol::{
    Answer, Evaluation, Question, Request, Usage, parse_questions, render_state,
};
use crate::score::{
    Calibration, argmax, choice_confidence, expected_level, score_confidence, softmax,
};

#[derive(Debug, Clone)]
pub struct JudgeConfig {
    pub template: Template,
    pub layout: Layout,
    pub calibration: Calibration,
    /// For a Choice: read this many cyclic rotations of the option order and
    /// average the per-option probabilities (position-bias control).
    pub permutations: usize,
    /// Add per-question diagnostics to every evaluation.
    pub debug: bool,
}

/// One question's uncalibrated result.
#[derive(Debug, Clone)]
pub struct RawQuestion {
    pub id: String,
    pub kind: &'static str,
    pub keys: Vec<String>,
    /// ln of the rotation-mean probability of each option.
    pub logprobs: Vec<f64>,
    pub evaluated: u64,
    pub cached: u64,
    /// Probability the model put on the labels at all (mean over rotations):
    /// well under one means the layout is off its distribution.
    pub label_mass: f64,
    pub latency_ms: f64,
}

pub struct Judge {
    pub engine: Engine,
    pub cfg: JudgeConfig,
}

impl Judge {
    pub fn new(engine: Engine, cfg: JudgeConfig) -> Self {
        Self { engine, cfg }
    }

    /// Read every question and return its uncalibrated log-probabilities;
    /// a request past `limits` is refused before any work.
    pub fn raw(&self, req: &Request, limits: Limits) -> Result<Vec<RawQuestion>, EngineError> {
        let questions = parse_questions(&req.questions).map_err(EngineError::Rejected)?;
        let prefix = prompt::prefix(
            self.cfg.layout,
            self.cfg.template,
            &render_state(&req.state),
        );
        struct Plan {
            id: String,
            kind: &'static str,
            keys: Vec<String>,
            orders: Vec<Vec<usize>>,
        }
        // Render every question and rotation first, so the engine sees the
        // whole request at once and forks the state for all of them.
        let mut plans = Vec::with_capacity(questions.len());
        let mut items: Vec<(prompt::Segs, Vec<String>)> = Vec::new();
        for (id, q) in &questions {
            let (kind, n) = match q {
                Question::Noul { .. } => ("noul", 2),
                Question::Choice { criteria, .. } => ("choice", criteria.len()),
                Question::Score { criteria, .. } => ("score", criteria.len()),
            };
            if n > MAX_OPTIONS {
                return Err(EngineError::Rejected(format!(
                    "question `{id}`: {n} options; a Choice takes at most {MAX_OPTIONS}"
                )));
            }
            let perms = if kind == "choice" {
                self.cfg.permutations.clamp(1, n)
            } else {
                1
            };
            let mut plan = Plan {
                id: id.clone(),
                kind,
                keys: Vec::new(),
                orders: Vec::with_capacity(perms),
            };
            for r in 0..perms {
                let order: Vec<usize> = (0..n).map(|i| (i + r) % n).collect();
                let rendered = prompt::render(self.cfg.layout, self.cfg.template, q, &order);
                if r == 0 {
                    // Rotation 0 is the identity: keys in request order.
                    plan.keys = rendered.keys.clone();
                }
                items.push((rendered.suffix, rendered.labels));
                plan.orders.push(order);
            }
            plans.push(plan);
        }
        let mut scored = self.engine.score_many(&prefix, &items, limits)?.into_iter();
        let mut out = Vec::with_capacity(plans.len());
        for plan in plans {
            let n = plan.keys.len();
            let perms = plan.orders.len();
            let mut mean = vec![0.0f64; n];
            let (mut evaluated, mut cached, mut latency_ms, mut mass) = (0, 0, 0.0, 0.0);
            for order in &plan.orders {
                let s = scored.next().ok_or_else(|| {
                    EngineError::Failed("the engine returned too few readings".into())
                })?;
                mass += s.logprobs.iter().map(|l| l.exp()).sum::<f64>() / perms as f64;
                let p = softmax(&s.logprobs, 1.0);
                // Rendered position `pos` holds original option `order[pos]`.
                for (pos, &orig) in order.iter().enumerate() {
                    mean[orig] += p[pos] / perms as f64;
                }
                evaluated += s.evaluated;
                cached += s.cached;
                latency_ms += s.latency_ms;
            }
            out.push(RawQuestion {
                id: plan.id,
                kind: plan.kind,
                keys: plan.keys,
                logprobs: mean.iter().map(|p| p.max(1e-300).ln()).collect(),
                evaluated,
                cached,
                label_mass: mass,
                latency_ms,
            });
        }
        Ok(out)
    }

    /// Every answer with its whole distribution, within TypeSafe's default
    /// context limits.
    pub fn evaluate(&self, req: &Request) -> Result<Evaluation, EngineError> {
        self.evaluate_with(req, Limits::DEFAULT)
    }

    /// `evaluate` within `limits` (`Limits::LONG_CONTEXT` for a request
    /// sent with `TypeSafe-Danger-Zone-Long-Context: true`).
    pub fn evaluate_with(&self, req: &Request, limits: Limits) -> Result<Evaluation, EngineError> {
        let raws = self.raw(req, limits)?;
        let mut answers = Map::new();
        let mut usage = Usage::default();
        let mut debug = Vec::new();
        for r in &raws {
            let t = self.cfg.calibration.temperature(r.kind, r.keys.len());
            let probs = softmax(&r.logprobs, t);
            let answer = match r.kind {
                "noul" => Answer::Noul {
                    noul: round(probs[0]),
                },
                "choice" => Answer::Choice {
                    choice: r.keys[argmax(&probs)].clone(),
                    probabilities: prob_map(&r.keys, &probs),
                    confidence: round(choice_confidence(&probs)),
                },
                _ => {
                    let legend: Map<String, Value> = req
                        .questions
                        .get(&r.id)
                        .and_then(|q| q.get("criteria"))
                        .and_then(Value::as_array)
                        .map(|a| {
                            a.iter()
                                .enumerate()
                                .map(|(i, v)| (i.to_string(), v.clone()))
                                .collect()
                        })
                        .unwrap_or_default();
                    Answer::Score {
                        score: round(expected_level(&probs)),
                        legend,
                        probabilities: prob_map(&r.keys, &probs),
                        confidence: round(score_confidence(&probs)),
                    }
                }
            };
            answers.insert(
                r.id.clone(),
                serde_json::to_value(answer).expect("answer to JSON"),
            );
            // Only tokens the model evaluated; reused state tokens are free.
            usage.input_tokens += r.evaluated;
            if self.cfg.debug {
                debug.push(json!({
                    "id": r.id, "kind": r.kind, "temperature": t,
                    "raw_logprobs": r.logprobs,
                    "prompt_evaluated": r.evaluated,
                    "prompt_cached": r.cached,
                    "label_mass": round(r.label_mass),
                    "latency_ms": round(r.latency_ms),
                }));
            }
        }
        Ok(Evaluation {
            model: self.engine.info.name.clone(),
            answers,
            usage,
            debug: self.cfg.debug.then_some(Value::Array(debug)),
        })
    }
}

fn prob_map(keys: &[String], probs: &[f64]) -> Map<String, Value> {
    keys.iter()
        .zip(probs)
        .map(|(k, p)| (k.clone(), json!(round(*p))))
        .collect()
}

/// Four decimals, as answers are reported.
fn round(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}
