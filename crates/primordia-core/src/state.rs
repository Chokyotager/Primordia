//! World state (structure of arrays), initialisation, the mirror map M,
//! state hashing and snapshots.

use crate::config::{Config, InitEe, parse_init_ee};
use crate::geom::mirror_dir;
use crate::isa;
use crate::params::*;
use crate::rng::{Rng, phase};
use anyhow::{Result, ensure};
use rayon::prelude::*;
use std::io::{Read, Write};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Vent {
    pub id: u32,
    pub pos: u32,
    pub age: u64,
    pub lifetime: u64,
}

/// Per-epoch ledger tallies (reset at the start of every epoch).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    pub energy_source: i64,
    pub energy_sink: i64,
    pub dissipation: i64,
    pub spending: i64,
    pub reaction_net: i64,
    pub matter_source: [i64; 3],
    pub matter_sink: [i64; 3],
    pub atom_source: i64,
    pub atom_sink: i64,
    pub births: u64,
    pub deaths: u64,
    pub body_moves: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LedgerState {
    pub energy_prev: i64,
    pub matter_expected: [i64; 3],
    pub atoms_expected: i64,
    pub energy_residual: i64,
    pub matter_residual: [i64; 3],
    pub atom_residual: i64,
    /// Residuals ever seen non-zero (sticky).
    pub violations: u64,
}

#[derive(Clone, Debug)]
pub struct State {
    pub n: usize,
    pub lmax: usize,
    pub nf: usize,
    pub epoch: u64,
    pub mirrored: bool,
    pub tapes: Vec<u8>,
    pub hdr: Vec<u32>,
    pub fields: Vec<u32>,
    pub site: Vec<u32>,
    pub vents: Vec<Vent>,
    pub next_vent: u32,
    pub tally: Tally,
    pub ledger: LedgerState,
    /// Light reaching each cell this epoch (Q16), CPU-only, recomputed in phase 2.
    pub light: Vec<u32>,
}

impl State {
    pub fn empty(p: &Params) -> State {
        let n = p.geom.n;
        State {
            n,
            lmax: p.lmax,
            nf: p.nf,
            epoch: 0,
            mirrored: false,
            tapes: vec![0; n * p.lmax],
            hdr: vec![0; n * HDR],
            fields: vec![0; n * p.nf],
            site: vec![0; n * SITE],
            vents: vec![],
            next_vent: 0,
            tally: Tally::default(),
            ledger: LedgerState::default(),
            light: vec![0; n],
        }
    }

    #[inline]
    pub fn h(&self, c: usize, w: usize) -> u32 {
        self.hdr[c * HDR + w]
    }
    #[inline]
    pub fn hm(&mut self, c: usize, w: usize) -> &mut u32 {
        &mut self.hdr[c * HDR + w]
    }
    #[inline]
    pub fn flags(&self, c: usize) -> u32 {
        self.hdr[c * HDR + H_FLAGS]
    }
    #[inline]
    pub fn occupied(&self, c: usize) -> bool {
        self.flags(c) & FL_OCC != 0
    }
    #[inline]
    pub fn len(&self, c: usize) -> usize {
        self.hdr[c * HDR + H_LEN] as usize
    }
    #[inline]
    pub fn tape(&self, c: usize) -> &[u8] {
        &self.tapes[c * self.lmax..c * self.lmax + self.len(c)]
    }
    #[inline]
    pub fn f(&self, c: usize, i: usize) -> u32 {
        self.fields[c * self.nf + i]
    }
    #[inline]
    pub fn fm(&mut self, c: usize, i: usize) -> &mut u32 {
        &mut self.fields[c * self.nf + i]
    }
    #[inline]
    pub fn wall(&self, c: usize) -> u32 {
        self.site[c * SITE + S_WALL]
    }
    #[inline]
    pub fn orient(&self, c: usize) -> u32 {
        (self.flags(c) & ORIENT_MASK) >> ORIENT_SHIFT
    }
    #[inline]
    pub fn bonds(&self, c: usize) -> u32 {
        (self.flags(c) & BOND_MASK) >> BOND_SHIFT
    }

    /// Initial random world.
    pub fn init(cfg: &Config, p: &Params) -> State {
        let mut st = State::empty(p);
        let rng = Rng::new(p.seed, 0, phase::INIT);
        let occ_q32 = crate::fixed::to_q32(cfg.world.init_occupancy);
        let lmax = p.lmax;
        let nf = p.nf;
        let valid = &p.orient.valid[p.geom.dims as usize];
        let init_ee = parse_init_ee(&cfg.chirality.init_ee).unwrap_or(InitEe::Random);
        let bias_ee = cfg.chirality.bias.initial_ee;
        let l_init = p.l_init as usize;
        st.tapes
            .par_chunks_mut(lmax)
            .zip(st.hdr.par_chunks_mut(HDR))
            .zip(st.fields.par_chunks_mut(nf))
            .enumerate()
            .for_each(|(c, ((tape, hdr), fld))| {
                let id = c as u32;
                fld[F_F] = if p.energy_on { cfg.energy.init_f } else { 0 };
                fld[F_T] = p.t_amb;
                for k in 0..p.k {
                    fld[p.f_chem(k, 0)] = cfg.chemistry.init[k];
                    fld[p.f_chem(k, 1)] = cfg.chemistry.init[k];
                }
                if p.matter_on {
                    fld[p.f_pool(0)] = cfg.matter.pool_l;
                    fld[p.f_pool(1)] = cfg.matter.pool_d;
                    fld[p.f_pool(2)] = cfg.matter.pool_a;
                }
                let r0 = rng.x4(id, 0, 0);
                if (r0[0] as u64) >= occ_q32 {
                    return;
                }
                for i in 0..l_init {
                    if i % 4 == 0 {
                        let r = rng.x4(id, 1, (i / 4) as u32);
                        for j in 0..4.min(l_init - i) {
                            tape[i + j] = (r[j] & 0xFF) as u8;
                        }
                    }
                }
                let t = &mut tape[..l_init];
                match init_ee {
                    InitEe::Random => {}
                    InitEe::Racemic => {
                        // flip excess-hand bytes (starting at a random offset) until |L-D| <= 1
                        let nl = t.iter().filter(|&&b| isa::is_chiral(b) && b & 1 == 0).count() as i64;
                        let nd = t.iter().filter(|&&b| isa::is_chiral(b) && b & 1 == 1).count() as i64;
                        let (from, mut need) = if nl > nd { (0u8, (nl - nd) / 2) } else { (1u8, (nd - nl) / 2) };
                        let off = (r0[2] as usize) % l_init.max(1);
                        for i in 0..l_init {
                            if need == 0 {
                                break;
                            }
                            let j = (off + i) % l_init;
                            if isa::is_chiral(t[j]) && t[j] & 1 == from {
                                t[j] ^= 1;
                                need -= 1;
                            }
                        }
                    }
                    InitEe::Seeded(x) => {
                        let pl = crate::fixed::to_q32((1.0 + x) / 2.0);
                        for (i, b) in t.iter_mut().enumerate() {
                            if isa::is_chiral(*b) {
                                let u = rng.u32(id, 2, i as u32) as u64;
                                *b = (*b & !1) | if u < pl { 0 } else { 1 };
                            }
                        }
                    }
                }
                if bias_ee != 0.0 {
                    let pf = crate::fixed::to_q32(bias_ee.abs() / 2.0);
                    let from = if bias_ee > 0.0 { 1 } else { 0 };
                    for (i, b) in t.iter_mut().enumerate() {
                        if isa::is_chiral(*b) && *b & 1 == from && (rng.u32(id, 3, i as u32) as u64) < pf {
                            *b ^= 1;
                        }
                    }
                }
                let o = valid[(r0[1] as usize) % valid.len()];
                hdr[H_LEN] = l_init as u32;
                hdr[H_FLAGS] = FL_OCC | (o << ORIENT_SHIFT);
                hdr[H_E] = if p.energy_on { cfg.energy.init_e } else { 0 };
                hdr[H_LINEAGE] = r0[3] | 1;
                hdr[H_BIRTH] = 0;
            });
        if bias_ee != 0.0 {
            // chemical / monomer excess: move a fraction eps/2 of D to L (or the reverse)
            let from = if bias_ee > 0.0 { 1 } else { 0 };
            let q = crate::fixed::to_q16(bias_ee.abs() / 2.0) as u64;
            for c in 0..st.n {
                for k in 0..p.k {
                    let (a, b) = (p.f_chem(k, from), p.f_chem(k, 1 - from));
                    let mv = ((st.f(c, a) as u64 * q) >> 16) as u32;
                    *st.fm(c, a) -= mv;
                    *st.fm(c, b) += mv;
                }
                if p.matter_on {
                    let (a, b) = (p.f_pool(from), p.f_pool(1 - from));
                    let mv = ((st.f(c, a) as u64 * q) >> 16) as u32;
                    *st.fm(c, a) -= mv;
                    *st.fm(c, b) += mv;
                }
            }
        }
        // initial vents
        if p.vents_on {
            for i in 0..cfg.energy.vents.count {
                let pos = crate::phases::random_vent_pos(p, &st, &rng, 100 + i);
                st.vents.push(Vent { id: st.next_vent, pos: pos as u32, age: 0, lifetime: cfg.energy.vents.lifetime });
                st.next_vent += 1;
            }
        }
        st.reset_ledger(p);
        st
    }

    /// Recompute ledger baselines from the current state.
    pub fn reset_ledger(&mut self, p: &Params) {
        self.ledger.energy_prev = self.energy_total();
        let m = self.matter_totals(p);
        self.ledger.matter_expected = m;
        self.ledger.atoms_expected = self.atom_total(p);
    }

    pub fn energy_total(&self) -> i64 {
        let e: u64 = (0..self.n).into_par_iter().map(|c| self.hdr[c * HDR + H_E] as u64 + self.fields[c * self.nf + F_F] as u64).sum();
        e as i64
    }

    pub fn matter_totals(&self, p: &Params) -> [i64; 3] {
        if !p.matter_on {
            return [0; 3];
        }
        let lmax = self.lmax;
        let r = (0..self.n)
            .into_par_iter()
            .map(|c| {
                let mut m = [0i64; 3];
                for cl in 0..3 {
                    m[cl] += self.fields[c * self.nf + p.f_pool(cl)] as i64;
                }
                let len = self.hdr[c * HDR + H_LEN] as usize;
                for &b in &self.tapes[c * lmax..c * lmax + len] {
                    let cl = isa::class(b);
                    if cl < 3 {
                        m[cl as usize] += 1;
                    }
                }
                m
            })
            .reduce(|| [0; 3], |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]]);
        r
    }

    pub fn atom_total(&self, p: &Params) -> i64 {
        if !p.chem_on {
            return 0;
        }
        (0..self.n)
            .into_par_iter()
            .map(|c| {
                let mut s = 0i64;
                for k in 0..p.k {
                    s += p.mass[k] as i64 * (self.fields[c * self.nf + p.f_chem(k, 0)] as i64 + self.fields[c * self.nf + p.f_chem(k, 1)] as i64);
                }
                s
            })
            .sum()
    }

    /// Apply the mirror transformation M to the whole state.
    pub fn mirror(&self, p: &Params) -> State {
        let g = &p.geom;
        let mut out = self.clone();
        out.mirrored = !self.mirrored;
        let lmax = self.lmax;
        let nf = self.nf;
        let o = &p.orient;
        out.tapes.par_chunks_mut(lmax).enumerate().for_each(|(c, t)| {
            let src = g.mirror_cell(c);
            for (i, b) in t.iter_mut().enumerate() {
                *b = isa::mirror(self.tapes[src * lmax + i]);
            }
        });
        out.hdr.par_chunks_mut(HDR).enumerate().for_each(|(c, h)| {
            let src = g.mirror_cell(c);
            h.copy_from_slice(&self.hdr[src * HDR..src * HDR + HDR]);
            let fl = h[H_FLAGS];
            if fl & FL_OCC == 0 {
                return;
            }
            let or = (fl & ORIENT_MASK) >> ORIENT_SHIFT;
            let bonds = (fl & BOND_MASK) >> BOND_SHIFT;
            let mut nb = 0;
            for d in 0..6 {
                if bonds & (1 << d) != 0 {
                    nb |= 1 << mirror_dir(d);
                }
            }
            h[H_FLAGS] = (fl & !(ORIENT_MASK | BOND_MASK)) | (o.mirror[or as usize] << ORIENT_SHIFT) | (nb << BOND_SHIFT);
        });
        let swap_hand = |i: usize| -> usize {
            // field index under L<->D exchange
            if i < 2 {
                return i;
            }
            let j = i - 2;
            if j < 2 * p.k {
                return 2 + (j ^ 1);
            }
            let j2 = j - 2 * p.k;
            if j2 < 3 {
                return if j2 < 2 { 2 + 2 * p.k + (j2 ^ 1) } else { i };
            }
            let j3 = j2 - 3;
            2 + 2 * p.k + 3 + (j3 ^ 1)
        };
        out.fields.par_chunks_mut(nf).enumerate().for_each(|(c, f)| {
            let src = g.mirror_cell(c);
            for i in 0..nf {
                f[swap_hand(i)] = self.fields[src * nf + i];
            }
        });
        out.site.par_chunks_mut(SITE).enumerate().for_each(|(c, s)| {
            let src = g.mirror_cell(c);
            s.copy_from_slice(&self.site[src * SITE..src * SITE + SITE]);
            if s[S_VOTE] != 0 {
                s[S_VOTE] = mirror_dir(s[S_VOTE] - 1) + 1;
            }
        });
        for v in &mut out.vents {
            v.pos = g.mirror_cell(v.pos as usize) as u32;
        }
        for c in 0..self.n {
            out.light[c] = self.light[g.mirror_cell(c)];
        }
        out
    }

    /// 64-bit FNV-1a over all buffers in a fixed order.
    pub fn hash(&self) -> u64 {
        const OFF: u64 = 0xcbf2_9ce4_8422_2325;
        const PR: u64 = 0x0000_0100_0000_01b3;
        // Hash chunks in parallel, then fold chunk hashes in order (still FNV-1a
        // over a fixed byte sequence: the per-chunk digests).
        fn fnv(bytes: &[u8]) -> u64 {
            let mut h = OFF;
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(PR);
            }
            h
        }
        fn words(v: &[u32]) -> &[u8] {
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
        }
        let mut parts: Vec<&[u8]> = vec![];
        let ep = self.epoch.to_le_bytes();
        parts.push(&ep);
        let chunk = 1 << 20;
        for buf in [&self.tapes[..], words(&self.hdr), words(&self.fields), words(&self.site)] {
            for c in buf.chunks(chunk) {
                parts.push(c);
            }
        }
        let mut vb = vec![];
        for v in &self.vents {
            vb.extend_from_slice(&v.id.to_le_bytes());
            vb.extend_from_slice(&v.pos.to_le_bytes());
            vb.extend_from_slice(&v.age.to_le_bytes());
            vb.extend_from_slice(&v.lifetime.to_le_bytes());
        }
        parts.push(&vb);
        let hs: Vec<u64> = parts.par_iter().map(|p| fnv(p)).collect();
        let mut h = OFF;
        for x in hs {
            for b in x.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(PR);
            }
        }
        h
    }

    /// First differing cell between two states (for twin / parity diagnostics).
    pub fn first_diff(&self, o: &State) -> Option<(usize, &'static str)> {
        for c in 0..self.n {
            if self.hdr[c * HDR..(c + 1) * HDR] != o.hdr[c * HDR..(c + 1) * HDR] {
                return Some((c, "hdr"));
            }
            if self.tapes[c * self.lmax..(c + 1) * self.lmax] != o.tapes[c * self.lmax..(c + 1) * self.lmax] {
                return Some((c, "tape"));
            }
            if self.fields[c * self.nf..(c + 1) * self.nf] != o.fields[c * self.nf..(c + 1) * self.nf] {
                return Some((c, "fields"));
            }
            if self.site[c * SITE..(c + 1) * SITE] != o.site[c * SITE..(c + 1) * SITE] {
                return Some((c, "site"));
            }
        }
        if self.vents != o.vents {
            return Some((0, "vents"));
        }
        None
    }

    // ---------------------------------------------------------- snapshots

    pub fn write_snapshot(&self, path: &std::path::Path, config_hash: u64) -> Result<()> {
        let f = std::fs::File::create(path)?;
        let mut z = zstd::Encoder::new(f, 3)?;
        z.write_all(b"PSNP")?;
        z.write_all(&1u32.to_le_bytes())?;
        z.write_all(&config_hash.to_le_bytes())?;
        z.write_all(&self.epoch.to_le_bytes())?;
        z.write_all(&[self.mirrored as u8])?;
        let put = |z: &mut zstd::Encoder<std::fs::File>, b: &[u8]| -> Result<()> {
            z.write_all(&(b.len() as u64).to_le_bytes())?;
            z.write_all(b)?;
            Ok(())
        };
        let w32 = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        put(&mut z, &self.tapes)?;
        put(&mut z, &w32(&self.hdr))?;
        put(&mut z, &w32(&self.fields))?;
        put(&mut z, &w32(&self.site))?;
        let mut vb = vec![];
        for v in &self.vents {
            vb.extend_from_slice(&v.id.to_le_bytes());
            vb.extend_from_slice(&v.pos.to_le_bytes());
            vb.extend_from_slice(&v.age.to_le_bytes());
            vb.extend_from_slice(&v.lifetime.to_le_bytes());
        }
        put(&mut z, &vb)?;
        let l = &self.ledger;
        let mut lb = vec![];
        for x in [l.energy_prev, l.matter_expected[0], l.matter_expected[1], l.matter_expected[2], l.atoms_expected, self.next_vent as i64] {
            lb.extend_from_slice(&x.to_le_bytes());
        }
        put(&mut z, &lb)?;
        z.finish()?;
        Ok(())
    }

    pub fn read_snapshot(path: &std::path::Path, p: &Params) -> Result<(State, u64)> {
        let f = std::fs::File::open(path)?;
        let mut z = zstd::Decoder::new(f)?;
        let mut all = vec![];
        z.read_to_end(&mut all)?;
        ensure!(&all[0..4] == b"PSNP", "bad snapshot magic");
        let ver = u32::from_le_bytes(all[4..8].try_into()?);
        ensure!(ver == 1, "snapshot version {ver}");
        let cfg_hash = u64::from_le_bytes(all[8..16].try_into()?);
        let epoch = u64::from_le_bytes(all[16..24].try_into()?);
        let mirrored = all[24] != 0;
        let mut pos = 25;
        let mut get = || -> Result<Vec<u8>> {
            let n = u64::from_le_bytes(all[pos..pos + 8].try_into()?) as usize;
            pos += 8;
            let v = all[pos..pos + n].to_vec();
            pos += n;
            Ok(v)
        };
        let r32 = |b: Vec<u8>| -> Vec<u32> { b.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect() };
        let mut st = State::empty(p);
        st.epoch = epoch;
        st.mirrored = mirrored;
        st.tapes = get()?;
        st.hdr = r32(get()?);
        st.fields = r32(get()?);
        st.site = r32(get()?);
        ensure!(st.tapes.len() == p.geom.n * p.lmax && st.fields.len() == p.geom.n * p.nf, "snapshot does not match config geometry");
        let vb = get()?;
        for ch in vb.chunks(24) {
            st.vents.push(Vent {
                id: u32::from_le_bytes(ch[0..4].try_into()?),
                pos: u32::from_le_bytes(ch[4..8].try_into()?),
                age: u64::from_le_bytes(ch[8..16].try_into()?),
                lifetime: u64::from_le_bytes(ch[16..24].try_into()?),
            });
        }
        let lb = get()?;
        let li = |i: usize| i64::from_le_bytes(lb[i * 8..i * 8 + 8].try_into().unwrap());
        st.ledger.energy_prev = li(0);
        st.ledger.matter_expected = [li(1), li(2), li(3)];
        st.ledger.atoms_expected = li(4);
        st.next_vent = li(5) as u32;
        Ok((st, cfg_hash))
    }
}
