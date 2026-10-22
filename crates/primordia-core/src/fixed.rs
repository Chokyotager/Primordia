//! Deterministic fixed-point helpers and CPU-precomputed tables.
//!
//! Transcendentals are evaluated with our own series using only IEEE-754
//! `+ - * /`, which are bit-exact on every platform, so the tables (and
//! therefore every run) do not depend on the host libm.

pub const Q16: u64 = 1 << 16;
pub const Q32: u64 = 1 << 32;

const LN2: f64 = 0.693_147_180_559_945_3;
const PI: f64 = 3.141_592_653_589_793;

/// exp(x) using range reduction and a Taylor series (basic ops only).
pub fn det_exp(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < -745.0 {
        return 0.0;
    }
    if x > 709.0 {
        return f64::INFINITY;
    }
    // x = k ln2 + r, |r| <= ln2/2
    let kf = x / LN2;
    let k = if kf >= 0.0 { (kf + 0.5) as i64 } else { (kf - 0.5) as i64 };
    let r = x - (k as f64) * LN2;
    let mut term = 1.0;
    let mut sum = 1.0;
    for i in 1..30 {
        term = term * r / (i as f64);
        sum += term;
    }
    let mut p = 1.0f64;
    if k >= 0 {
        for _ in 0..k {
            p *= 2.0;
        }
    } else {
        for _ in 0..(-k) {
            p *= 0.5;
        }
    }
    sum * p
}

/// sin(x) with range reduction to [-pi, pi] and a Taylor series.
pub fn det_sin(x: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let n = (x / two_pi) as i64;
    let mut r = x - (n as f64) * two_pi;
    if r > PI {
        r -= two_pi;
    }
    if r < -PI {
        r += two_pi;
    }
    let r2 = r * r;
    let mut term = r;
    let mut sum = r;
    for i in 1..20 {
        term = -term * r2 / (((2 * i) * (2 * i + 1)) as f64);
        sum += term;
    }
    sum
}

pub fn det_powi(base: f64, n: u32) -> f64 {
    let mut p = 1.0;
    for _ in 0..n {
        p *= base;
    }
    p
}

/// Convert a probability/rate to Q32 (saturating at 2^32).
pub fn to_q32(p: f64) -> u64 {
    if p <= 0.0 {
        0
    } else if p >= 1.0 {
        Q32
    } else {
        (p * Q32 as f64) as u64
    }
}

/// Convert a non-negative real to Q16 (saturating at u32::MAX).
pub fn to_q16(x: f64) -> u32 {
    if x <= 0.0 {
        0
    } else {
        let v = x * Q16 as f64;
        if v >= u32::MAX as f64 { u32::MAX } else { v as u32 }
    }
}

/// Signed Q16.
pub fn to_q16s(x: f64) -> i64 {
    (x * Q16 as f64) as i64
}

/// Arrhenius factor table, 4096 entries indexed by T >> 4 (T in 0.01 K),
/// values Q16.16 clamped to 2^31.
pub const ARR_SIZE: usize = 4096;

pub fn arrhenius_table(activation_ev: f64, t_ref_k: f64) -> Vec<u32> {
    const K_B: f64 = 8.617_333_262e-5; // eV/K
    let mut t = Vec::with_capacity(ARR_SIZE);
    for i in 0..ARR_SIZE {
        if activation_ev == 0.0 {
            t.push(Q16 as u32);
            continue;
        }
        // centre of the bucket in Kelvin
        let tk = ((i as f64) * 16.0 + 8.0) / 100.0;
        if tk < 1.0 {
            t.push(0);
            continue;
        }
        let a = det_exp(-(activation_ev / K_B) * (1.0 / tk - 1.0 / t_ref_k));
        let v = a * Q16 as f64;
        t.push(if v >= 2147483648.0 { 2147483648u32 } else { v as u32 });
    }
    t
}

#[inline]
pub fn arr_lookup(table: &[u32], t: u32) -> u32 {
    let i = (t >> 4) as usize;
    table[i.min(ARR_SIZE - 1)]
}

/// exp(-lambda) table for Poisson sampling: index = lambda in Q10 (lambda < 64).
pub const POIS_SIZE: usize = 64 * 1024;

pub fn poisson_exp_table() -> Vec<u32> {
    (0..POIS_SIZE)
        .map(|i| {
            let v = det_exp(-(i as f64) / 1024.0) * Q32 as f64;
            if v >= u32::MAX as f64 { u32::MAX } else { v as u32 }
        })
        .collect()
}

/// Inverse-CDF Poisson sample. `lambda_q32` is the mean in Q32 (u64), `u` a
/// uniform u32. Deterministic integer arithmetic only.
#[inline]
pub fn poisson_sample(table: &[u32], lambda_q32: u64, u: u32, kmax: u32) -> u32 {
    if lambda_q32 == 0 {
        return 0;
    }
    let lam = lambda_q32.min(((POIS_SIZE - 1) as u64) << 22) as u128;
    // P(K=0) = exp(-lambda): series for small lambda, table otherwise.
    let mut p: u128 = if lam < (1u128 << 28) {
        let l2 = (lam * lam) >> 32;
        let l3 = (l2 * lam) >> 32;
        (1u128 << 32) - lam + l2 / 2 - l3 / 6
    } else {
        table[(lam >> 22) as usize] as u128
    };
    let mut cdf = p;
    let u = u as u128;
    let mut k = 0u32;
    while u >= cdf && k < kmax {
        k += 1;
        p = ((p * lam) >> 32) / k as u128;
        if p == 0 {
            break;
        }
        cdf += p;
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exp_sin_close_to_std() {
        for i in -200..200 {
            let x = i as f64 * 0.137;
            assert!((det_exp(x) - x.exp()).abs() / x.exp() < 1e-12, "exp {x}");
            assert!((det_sin(x) - x.sin()).abs() < 1e-9, "sin {x}");
        }
    }
    #[test]
    fn poisson_mean() {
        let t = poisson_exp_table();
        let lam = 0.7f64;
        let lq = (lam * Q32 as f64) as u64;
        let mut s = 0u64;
        let n = 200_000u32;
        for i in 0..n {
            let u = crate::rng::philox_u32(1, 2, [i, 0, 0, 0]);
            s += poisson_sample(&t, lq, u, 64) as u64;
        }
        let m = s as f64 / n as f64;
        assert!((m - lam).abs() < 0.02, "mean {m}");
        // tiny lambda
        let lq = (0.0005 * Q32 as f64) as u64;
        let mut s = 0u64;
        for i in 0..n {
            let u = crate::rng::philox_u32(3, 2, [i, 0, 0, 0]);
            s += poisson_sample(&t, lq, u, 64) as u64;
        }
        let m = s as f64 / n as f64;
        assert!((m - 0.0005).abs() < 0.0003, "tiny mean {m}");
    }
}

/// floor(a * 2^16 / b) for a <= b, using only 32-bit operations (shared with WGSL).
#[inline]
pub fn ratio16(mut a: u32, mut b: u32) -> u32 {
    if b == 0 {
        return 0;
    }
    while b >= 65536 {
        a >>= 1;
        b >>= 1;
    }
    (a << 16) / b
}

/// (a * b) >> 16 reduced mod 2^32 (shared with WGSL's 16-bit-limb version).
#[inline]
pub fn mulshr16(a: u32, b: u32) -> u32 {
    ((a as u64 * b as u64) >> 16) as u32
}

/// a * b for a, b <= 2^16, saturating at u32::MAX.
#[inline]
pub fn mul16_sat(a: u32, b: u32) -> u32 {
    if a >= 65536 && b >= 65536 { u32::MAX } else { a.wrapping_mul(b) }
}
