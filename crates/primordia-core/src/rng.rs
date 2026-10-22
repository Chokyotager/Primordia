//! Philox4x32-10 counter-based RNG. No state: every draw is a pure function
//! of (key, counter). The WGSL port in `primordia-gpu` must stay identical.

const M0: u32 = 0xD251_1F53;
const M1: u32 = 0xCD9E_8D57;
const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;

#[inline]
fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let p = (a as u64) * (b as u64);
    ((p >> 32) as u32, p as u32)
}

#[inline]
pub fn philox4x32(key: [u32; 2], ctr: [u32; 4]) -> [u32; 4] {
    let (mut k0, mut k1) = (key[0], key[1]);
    let mut c = ctr;
    for _ in 0..10 {
        let (hi0, lo0) = mulhilo(M0, c[0]);
        let (hi1, lo1) = mulhilo(M1, c[2]);
        c = [hi1 ^ c[1] ^ k0, lo1, hi0 ^ c[3] ^ k1, lo0];
        k0 = k0.wrapping_add(W0);
        k1 = k1.wrapping_add(W1);
    }
    c
}

#[inline]
pub fn philox_u32(k0: u32, k1: u32, ctr: [u32; 4]) -> u32 {
    philox4x32([k0, k1], ctr)[0]
}

/// Phase ids (part of the counter, so streams of different phases never collide).
pub mod phase {
    pub const INIT: u32 = 1;
    pub const SOURCES: u32 = 2;
    pub const TRANSPORT: u32 = 3;
    pub const CHEM: u32 = 4;
    pub const ABSORB: u32 = 5;
    pub const EXEC: u32 = 6;
    pub const SCHED: u32 = 7;
    pub const BODY: u32 = 8;
    pub const MUTATE: u32 = 9;
    pub const BONDBREAK: u32 = 10;
    pub const INTERVENE: u32 = 11;
    pub const LINEAGE: u32 = 12;
    pub const PAIRING: u32 = 13;
}

/// Keyed generator for one (seed, epoch, phase). Counter layout:
/// (canonical cell or face id, epoch, phase<<24 | stream, draw index).
#[derive(Clone, Copy, Debug)]
pub struct Rng {
    pub key: [u32; 2],
    pub epoch: u32,
    pub phase: u32,
}

impl Rng {
    pub fn new(seed: u64, epoch: u64, phase: u32) -> Self {
        Rng { key: [seed as u32, (seed >> 32) as u32], epoch: epoch as u32, phase }
    }
    #[inline]
    pub fn x4(&self, id: u32, stream: u32, draw: u32) -> [u32; 4] {
        philox4x32(self.key, [id, self.epoch, (self.phase << 24) | (stream & 0xFF_FFFF), draw])
    }
    #[inline]
    pub fn u32(&self, id: u32, stream: u32, draw: u32) -> u32 {
        self.x4(id, stream, draw)[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_answer() {
        // Random123 known-answer test for philox4x32-10.
        let r = philox4x32([0, 0], [0, 0, 0, 0]);
        assert_eq!(r, [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]);
        let r = philox4x32([0xa4093822, 0x299f31d0], [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344]);
        assert_eq!(r, [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1]);
    }
}
