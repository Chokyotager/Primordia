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
        for lam in [0.0005f64, 0.02, 0.7, 3.0] {
            let lq = (lam * (1u64 << 24) as f64) as u32;
            let n = 200_000u32;
            let s: u64 = (0..n).map(|i| poisson24(&t, lq, crate::rng::philox_u32(1, 2, [i, 0, 0, 0]), 64) as u64).sum();
            let m = s as f64 / n as f64;
            assert!((m - lam).abs() < 0.02 * lam.max(0.05), "lambda {lam}: mean {m}");
        }
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

// ---- 32-bit-exact helpers shared with WGSL (phys.wgsl). Each is defined as
// the low 32 bits (or a saturation) of an exact wide computation, which WGSL
// reproduces with 16-bit limbs.

#[inline]
pub fn sat32(x: u64) -> u32 {
    x.min(u32::MAX as u64) as u32
}
/// min((a*b) >> 16, 2^32-1)
#[inline]
pub fn mulshr16_sat(a: u32, b: u32) -> u32 {
    sat32((a as u64 * b as u64) >> 16)
}
/// ((a*b + d) >> 16) mod 2^32
#[inline]
pub fn mad_shr16(a: u32, b: u32, d: u32) -> u32 {
    ((a as u64 * b as u64 + d as u64) >> 16) as u32
}
/// (a*b + d) >> 32 (always fits)
#[inline]
pub fn mad_shr32(a: u32, b: u32, d: u32) -> u32 {
    ((a as u64 * b as u64 + d as u64) >> 32) as u32
}
/// min((a*b) >> 8, 2^32-1)
#[inline]
pub fn mulshr8_sat(a: u32, b: u32) -> u32 {
    sat32((a as u64 * b as u64) >> 8)
}
/// floor((k*(qa-qb) + r) / 2^16) mod 2^32 as i32, k >= 0, r < 2^16.
#[inline]
pub fn flux_j(k: u32, qa: u32, qb: u32, r: u32) -> i32 {
    ((k as i64 * (qa as i64 - qb as i64) + r as i64) >> 16) as i32
}
/// Limit a positive outflow j when the cell's total outflow o exceeds its content q.
#[inline]
pub fn flux_lim(j: u32, q: u32, o: u32) -> u32 {
    if o > q { mulshr16(j, ratio16(q, o)) } else { j }
}
#[inline]
pub fn sat_add(a: u32, b: u32) -> u32 {
    a.saturating_add(b)
}

/// Poisson sample with mean `lam24` (Q24). Shared with WGSL.
pub fn poisson24(table: &[u32], lam24: u32, u: u32, kmax: u32) -> u32 {
    if lam24 == 0 {
        return 0;
    }
    let mut p: u32 = if lam24 < (1 << 20) {
        let l = lam24 << 8; // Q32
        let l2 = ((l as u64 * l as u64) >> 32) as u32;
        let l3 = ((l2 as u64 * l as u64) >> 32) as u32;
        0xFFFF_FFFFu32.wrapping_sub(l).wrapping_add(l2 / 2).wrapping_sub(l3 / 6)
    } else {
        table[((lam24 >> 14) as usize).min(POIS_SIZE - 1)]
    };
    let mut cdf = p;
    let mut k = 0;
    while u >= cdf && k < kmax {
        k += 1;
        p = (((p as u64 * lam24 as u64) >> 24) as u32) / k;
        if p == 0 {
            break;
        }
        cdf = cdf.saturating_add(p);
    }
    k
}
