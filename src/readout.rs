//! The readout: log-probabilities of chosen tokens from one row of logits,
//! normalised over the whole vocabulary,
//!
//!   log p(t) = z[t] - lse(z),   lse(z) = max + ln sum_i exp(z[i] - max).
//!
//! Two implementations of the one sum:
//! - `Exact` is Intel-Phi-Jev's arithmetic bit for bit: the f32 maximum, then
//!   exp(z[i] - max) in f64 added in vocabulary order. Same build, same bits.
//! - `Avx2` uses AVX2 + FMA: the maximum with vmaxps (exact, a maximum does not
//!   depend on order), exp in f64 by Cody-Waite range reduction and a degree-13
//!   Taylor polynomial evaluated with FMA, 2^n built in the exponent bits, the
//!   sum carried in 16 f64 lanes. Its exp is within a few ulp of libm's and
//!   the sum is added in another order, so the result differs from `Exact` in
//!   the last bits only (`tests` below, and `jevbase bench`).

/// Which implementation reads a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Readout {
    Exact,
    Avx2,
}

impl Readout {
    /// AVX2 + FMA when this CPU has both, else the exact scalar sum.
    pub fn best() -> Self {
        if Self::Avx2.available() {
            Self::Avx2
        } else {
            Self::Exact
        }
    }

    /// Whether this CPU can run it.
    pub fn available(self) -> bool {
        match self {
            Self::Exact => true,
            Self::Avx2 => avx2_fma(),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Avx2 => "avx2",
        }
    }
}

impl std::str::FromStr for Readout {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::best()),
            "exact" => Ok(Self::Exact),
            "avx2" if Self::Avx2.available() => Ok(Self::Avx2),
            "avx2" => Err("readout `avx2`: this CPU lacks AVX2 or FMA (use auto or exact)".into()),
            other => Err(format!("unknown readout `{other}` (auto|avx2|exact)")),
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn avx2_fma() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_fma() -> bool {
    false
}

/// Full-vocabulary log-probabilities of `ids` in `row`.
pub fn logprobs(mode: Readout, row: &[f32], ids: &[i32]) -> Vec<f64> {
    let lse = lse(mode, row);
    ids.iter().map(|&t| row[t as usize] as f64 - lse).collect()
}

/// log sum exp of `row`.
pub fn lse(mode: Readout, row: &[f32]) -> f64 {
    match mode {
        #[cfg(target_arch = "x86_64")]
        Readout::Avx2 if avx2_fma() => {
            // SAFETY: the CPU has AVX2 and FMA (checked just above).
            unsafe { x86::lse_avx2(row) }
        }
        _ => lse_exact(row),
    }
}

/// Intel-Phi-Jev's `read`, term for term.
fn lse_exact(row: &[f32]) -> f64 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    max + row
        .iter()
        .map(|&x| (x as f64 - max).exp())
        .sum::<f64>()
        .ln()
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// ln 2 split for Cody-Waite reduction with FMA: HI is ln 2 rounded to
    /// f64, LO the rest.
    const LN2_HI: f64 = std::f64::consts::LN_2;
    const LN2_LO: f64 = 2.319_046_813_846_299_6e-17;
    /// Below this exp is under 3.3e-308; such terms cannot move a sum that
    /// holds exp(0) = 1 for the maximum itself, so inputs are clamped here,
    /// which keeps 2^n a normal number.
    const FLOOR: f64 = -708.0;
    /// 1/k!, k = 0..=13: |r| <= ln2/2 leaves a truncation error near 6e-18.
    const C: [f64; 14] = [
        1.0,
        1.0,
        1.0 / 2.0,
        1.0 / 6.0,
        1.0 / 24.0,
        1.0 / 120.0,
        1.0 / 720.0,
        1.0 / 5040.0,
        1.0 / 40320.0,
        1.0 / 362880.0,
        1.0 / 3628800.0,
        1.0 / 39916800.0,
        1.0 / 479001600.0,
        1.0 / 6227020800.0,
    ];
    /// 2^52 + 1023: n + MAGIC holds the biased exponent n + 1023 in its low
    /// mantissa bits, for integer n in [-1022, 0].
    const MAGIC: f64 = 4503599627370496.0 + 1023.0;

    /// exp of four f64 values, each <= 0.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub fn exp4(x: __m256d) -> __m256d {
        let x = _mm256_max_pd(x, _mm256_set1_pd(FLOOR));
        let n = _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            _mm256_mul_pd(x, _mm256_set1_pd(std::f64::consts::LOG2_E)),
        );
        let r = _mm256_fnmadd_pd(n, _mm256_set1_pd(LN2_HI), x);
        let r = _mm256_fnmadd_pd(n, _mm256_set1_pd(LN2_LO), r);
        let mut p = _mm256_set1_pd(C[13]);
        for &c in C[..13].iter().rev() {
            p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(c));
        }
        let biased = _mm256_castpd_si256(_mm256_add_pd(n, _mm256_set1_pd(MAGIC)));
        let scale = _mm256_castsi256_pd(_mm256_slli_epi64::<52>(biased));
        _mm256_mul_pd(p, scale)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn hmax(v: __m256) -> f32 {
        let v = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let v = _mm_max_ps(v, _mm_movehl_ps(v, v));
        let v = _mm_max_ss(v, _mm_shuffle_ps::<1>(v, v));
        _mm_cvtss_f32(v)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn hsum(v: __m256d) -> f64 {
        let v = _mm_add_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd::<1>(v));
        _mm_cvtsd_f64(_mm_add_sd(v, _mm_unpackhi_pd(v, v)))
    }

    /// log sum exp of `row` with AVX2 + FMA.
    #[target_feature(enable = "avx2,fma")]
    pub fn lse_avx2(row: &[f32]) -> f64 {
        let n = row.len();
        // Maximum: four 8-lane accumulators. The loaded value is the first
        // operand, so a NaN in the row is skipped as f32::max skips it.
        let mut m = [_mm256_set1_ps(f32::NEG_INFINITY); 4];
        let mut i = 0;
        while i + 32 <= n {
            for (k, acc) in m.iter_mut().enumerate() {
                // SAFETY: i + 8k + 8 <= i + 32 <= n.
                let v = unsafe { _mm256_loadu_ps(row.as_ptr().add(i + 8 * k)) };
                *acc = _mm256_max_ps(v, *acc);
            }
            i += 32;
        }
        let both = _mm256_max_ps(_mm256_max_ps(m[0], m[1]), _mm256_max_ps(m[2], m[3]));
        let max = row[i..].iter().copied().fold(hmax(both), f32::max) as f64;

        // Sum: four 4-lane f64 accumulators, sixteen terms a step.
        let vmax = _mm256_set1_pd(max);
        let mut s = [_mm256_setzero_pd(); 4];
        let mut i = 0;
        while i + 16 <= n {
            for (k, acc) in s.iter_mut().enumerate() {
                // SAFETY: i + 4k + 4 <= i + 16 <= n.
                let f = unsafe { _mm_loadu_ps(row.as_ptr().add(i + 4 * k)) };
                let x = _mm256_sub_pd(_mm256_cvtps_pd(f), vmax);
                *acc = _mm256_add_pd(*acc, exp4(x));
            }
            i += 16;
        }
        let tail: f64 = row[i..].iter().map(|&x| (x as f64 - max).exp()).sum();
        let total = hsum(_mm256_add_pd(
            _mm256_add_pd(s[0], s[1]),
            _mm256_add_pd(s[2], s[3]),
        )) + tail;
        max + total.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic row with the spread of real logits (a few large
    /// values over a broad negative mass), from a fixed linear congruence.
    fn row(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|i| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (s >> 11) as f64 / (1u64 << 53) as f64;
                let base = -8.0 + 16.0 * u;
                (if i % 997 == 0 { base + 20.0 } else { base }) as f32
            })
            .collect()
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn exp4_is_within_a_few_ulp_of_libm() {
        use std::arch::x86_64::*;
        if !Readout::Avx2.available() {
            return;
        }
        let mut worst = 0.0f64;
        let steps = 200_000;
        for k in 0..steps {
            let x0 = -700.0 * k as f64 / steps as f64;
            let xs = [x0, x0 - 1e-3, x0 - 0.3465, x0 - 0.3466];
            let mut out = [0.0f64; 4];
            // SAFETY: AVX2 and FMA are present (checked above); `out` holds 4.
            unsafe {
                let v = x86::exp4(_mm256_loadu_pd(xs.as_ptr()));
                _mm256_storeu_pd(out.as_mut_ptr(), v);
            }
            for (x, y) in xs.iter().zip(out) {
                let want = x.exp();
                worst = worst.max(((y - want) / want).abs());
            }
        }
        // 4 ulp of 1.0 is 8.9e-16.
        assert!(worst < 8.9e-16, "worst relative error {worst:e}");
    }

    #[test]
    fn avx2_lse_matches_exact() {
        if !Readout::Avx2.available() {
            return;
        }
        for (n, seed) in [(248_320, 1), (151_936, 2), (37, 3), (1, 4)] {
            let r = row(n, seed);
            let a = lse(Readout::Exact, &r);
            let b = lse(Readout::Avx2, &r);
            assert!(
                (a - b).abs() <= 1e-13 * a.abs().max(1.0),
                "n {n}: exact {a} avx2 {b}"
            );
        }
    }

    #[test]
    fn logprobs_are_row_minus_lse() {
        let r = vec![0.0f32, 1.0, 2.0];
        let lp = logprobs(Readout::Exact, &r, &[2, 0]);
        let z = (1.0f64 + 1f64.exp() + 2f64.exp()).ln();
        assert!((lp[0] - (2.0 - z)).abs() < 1e-15);
        assert!((lp[1] - (0.0 - z)).abs() < 1e-15);
    }
}
