//! From label log-probabilities to typed answers: restricted softmax,
//! temperature per question bucket, confidence, and the expected level.
//! Ported unchanged from Intel-Phi-Jev's `src/score.rs`.

use std::collections::HashMap;

/// Temperatures per bucket: `noul:2`, `choice:2`, `choice:3-5`,
/// `choice:6-10`, `choice:11+`, `score:2`, `score:3-5`, `score:6-10`.
/// A bucket without one uses `default` (1.0: the model's own distribution).
#[derive(Debug, Clone)]
pub struct Calibration {
    pub default: f64,
    pub by_bucket: HashMap<String, f64>,
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            default: 1.0,
            by_bucket: HashMap::new(),
        }
    }
}

impl Calibration {
    /// The bucket a question of `kind` with `n` options falls in.
    pub fn bucket(kind: &str, n: usize) -> String {
        let range = match n {
            0..=2 => "2",
            3..=5 => "3-5",
            6..=10 => "6-10",
            _ => "11+",
        };
        format!("{kind}:{range}")
    }

    /// The temperature for a question; anything not positive and finite is 1.
    pub fn temperature(&self, kind: &str, n: usize) -> f64 {
        let t = self
            .by_bucket
            .get(&Self::bucket(kind, n))
            .copied()
            .unwrap_or(self.default);
        if t > 0.0 && t.is_finite() { t } else { 1.0 }
    }
}

/// Softmax of `logprobs / t`; `-inf` entries get probability 0.
pub fn softmax(logprobs: &[f64], t: f64) -> Vec<f64> {
    let m = logprobs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        let n = logprobs.len().max(1);
        return vec![1.0 / n as f64; logprobs.len()];
    }
    let e: Vec<f64> = logprobs.iter().map(|&l| ((l - m) / t).exp()).collect();
    let z: f64 = e.iter().sum();
    e.into_iter().map(|x| x / z).collect()
}

/// Normalise to a distribution; uniform when the total is zero.
fn normalized(probs: &[f64]) -> Vec<f64> {
    let total: f64 = probs.iter().sum();
    if total <= 0.0 {
        return vec![1.0 / probs.len() as f64; probs.len()];
    }
    probs.iter().map(|p| p / total).collect()
}

/// Choice confidence as TypeSafe computes it: the peak probability scaled
/// from uniform (0) to certainty (1), `(p_max - 1/n) / (1 - 1/n)`.
pub fn choice_confidence(probs: &[f64]) -> f64 {
    if probs.len() < 2 {
        return 1.0;
    }
    let p = normalized(probs);
    let u = 1.0 / p.len() as f64;
    let pmax = p.iter().copied().fold(0.0, f64::max);
    ((pmax - u) / (1.0 - u)).clamp(0.0, 1.0)
}

/// Score confidence as TypeSafe computes it: one minus the mean distance
/// from the modal level over the mean absolute deviation of a uniform
/// distribution, floored at 0. A near miss costs less than a far one.
pub fn score_confidence(probs: &[f64]) -> f64 {
    if probs.len() < 2 {
        return 1.0;
    }
    let p = normalized(probs);
    let mode = argmax(&p) as f64;
    let spread: f64 = p
        .iter()
        .enumerate()
        .map(|(i, q)| q * (i as f64 - mode).abs())
        .sum();
    let n = p.len() as f64;
    let center = (n - 1.0) / 2.0;
    let uniform_mad: f64 = (0..p.len()).map(|i| (i as f64 - center).abs()).sum::<f64>() / n;
    (1.0 - spread / uniform_mad).max(0.0)
}

/// Probability-weighted mean level index.
pub fn expected_level(probs: &[f64]) -> f64 {
    probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum()
}

/// Index of the largest probability (the first one on a tie).
pub fn argmax(probs: &[f64]) -> usize {
    let mut best = 0;
    for (i, &p) in probs.iter().enumerate() {
        if p > probs[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_matches_typesafe() {
        // The cases of system-one-adapter's confidence tests, and the docs'
        // three-option demo, (3 p_max - 1) / 2.
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        assert!(close(score_confidence(&[0.2; 5]), 0.0));
        assert!(close(score_confidence(&[0.04; 5]), 0.0));
        assert!(close(score_confidence(&[0.01, 0.02, 0.07, 0.3, 0.6]), 0.55));
        assert!(close(choice_confidence(&[0.5, 0.5]), 0.0));
        assert!(close(choice_confidence(&[0.2, 0.2]), 0.0));
        assert!(close(choice_confidence(&[0.82, 0.18]), 0.64));
        assert!(close(score_confidence(&[1.0]), 1.0));
        assert!(close(choice_confidence(&[1.0]), 1.0));
        assert!(close(choice_confidence(&[0.6, 0.3, 0.1]), 0.4));
    }

    #[test]
    fn softmax_handles_missing_labels() {
        let p = softmax(&[0.0, f64::NEG_INFINITY], 1.0);
        assert!((p[0] - 1.0).abs() < 1e-12 && p[1] == 0.0);
        let p = softmax(&[-1.0, -1.0], 2.0);
        assert!((p[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn buckets() {
        assert_eq!(Calibration::bucket("noul", 2), "noul:2");
        assert_eq!(Calibration::bucket("choice", 30), "choice:11+");
        let mut c = Calibration::default();
        c.by_bucket.insert("score:3-5".into(), 0.5);
        assert_eq!(c.temperature("score", 5), 0.5);
        assert_eq!(c.temperature("choice", 5), 1.0);
    }
}
