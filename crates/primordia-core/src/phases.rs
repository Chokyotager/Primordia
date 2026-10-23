//! CPU implementation of every non-execution phase: sources, transport,
//! chemistry, absorption/upkeep, body movement, mutation/decay, ledgers.
//! Every per-cell update reads only pre-phase values of its neighbours, so
//! the rayon-parallel loops are deterministic.

use crate::fixed::*;
use crate::geom::{Boundary, mirror_dir};
use crate::isa;
use crate::params::*;
use crate::rng::{Rng, phase};
use crate::state::{State, Vent};
use rayon::prelude::*;

const TWO_PI: f64 = 6.283_185_307_179_586;

pub fn random_vent_pos(p: &Params, st: &State, rng: &Rng, id: u32) -> usize {
    let g = &p.geom;
    let r = rng.x4(id, 7, 0);
    let mut pc = [0u32; 3];
    for a in 0..(g.dims.max(1) as usize) {
        pc[a] = r[a] % g.size[a];
    }
    if g.dims == 3 && p.vent_cfg.floor_only {
        pc[2] = 0;
    }
    let c = g.index(pc);
    if st.mirrored { g.mirror_cell(c) } else { c }
}

/// Exposure e in Q16 for every cell: 1 - (b + w_o * o) / faces.
pub fn exposure(st: &State, p: &Params) -> Vec<u32> {
    let faces = p.geom.faces();
    let w = p.w_occ_q16;
    (0..st.n)
        .into_par_iter()
        .map(|c| {
            if faces == 0 || !st.occupied(c) {
                return 1 << 16;
            }
            let bonds = st.bonds(c);
            let (mut b, mut o) = (0u64, 0u64);
            for d in 0..faces {
                if bonds & (1 << d) != 0 {
                    b += 1;
                } else {
                    let t = p.nbr[c * 6 + d as usize];
                    if t != NONE && st.site[t as usize * SITE + S_WALL] == 0 && st.occupied(t as usize) {
                        o += 1;
                    }
                }
            }
            let cov = (b * 65536 + w * o) / faces as u64;
            (65536u64.saturating_sub(cov)) as u32
        })
        .collect()
}

// ------------------------------------------------------------ phase 2

fn cycle_factor(p: &Params, epoch: u64) -> f64 {
    let mut f = 1.0;
    if p.day_period > 0 {
        let s = det_sin(TWO_PI * (epoch % p.day_period) as f64 / p.day_period as f64);
        f = if s > 0.0 { s } else { 0.0 };
    }
    if p.season_period > 0 {
        f *= 1.0 + p.season_amp * det_sin(TWO_PI * (epoch % p.season_period) as f64 / p.season_period as f64);
    }
    f.max(0.0)
}

/// Light at the top of each y row this epoch (Q16, clamped to u32). The GPU
/// path uploads this small array instead of evaluating the day cycle.
pub fn row_lights(p: &Params, epoch: u64) -> Vec<u32> {
    let g = &p.geom;
    let cyc = to_q16(cycle_factor(p, epoch)) as u64;
    let base = (p.sun_i0_q16 * cyc) >> 16;
    let ny = g.size[1].max(1);
    (0..ny)
        .map(|y| {
            if !p.sun_on {
                return 0;
            }
            let v = if g.dims < 2 || ny < 2 {
                base
            } else {
                // 1 + g * (2y/(ny-1) - 1)
                let num = 2 * y as i64 - (ny as i64 - 1);
                let fac = 65536 + (p.sun_grad_q16 * num) / (ny as i64 - 1);
                ((base as i64 * fac.max(0)) >> 16) as u64
            };
            sat32(v)
        })
        .collect()
}

pub fn sources(st: &mut State, p: &Params) {
    let g = &p.geom;
    let epoch = st.epoch;
    let rng = Rng::new(p.seed, epoch, phase::SOURCES);
    let mirrored = st.mirrored;
    let rows = row_lights(p, epoch);
    let ny = g.size[1].max(1);
    if !p.sun_on {
        st.light.iter_mut().for_each(|l| *l = 0);
    } else if g.dims == 3 {
        let (sx, sy, sz) = (g.size[0] as usize, g.size[1] as usize, g.size[2] as usize);
        let att = 65536 - (p.sun_atten_q16.min(1 << 16) as u32);
        let occ: Vec<bool> = (0..st.n).map(|c| st.occupied(c)).collect();
        let light = &mut st.light;
        for y in 0..sy {
            for x in 0..sx {
                let mut i = rows[y];
                for z in (0..sz).rev() {
                    let c = x + sx * (y + sy * z);
                    light[c] = i;
                    if occ[c] {
                        i = mulshr16(i, att);
                    }
                }
            }
        }
    } else {
        let sx = g.size[0] as usize;
        st.light.par_iter_mut().enumerate().for_each(|(c, l)| {
            let y = if g.dims >= 2 { (c / sx) % ny as usize } else { 0 };
            *l = rows[y];
        });
    }
    // sunlight injects free energy and heat everywhere
    let nf = st.nf;
    if p.sun_on {
        let light = &st.light;
        let injected: u64 = st
            .fields
            .par_chunks_mut(nf)
            .enumerate()
            .map(|(c, f)| {
                let l = light[c] as u64;
                if l == 0 {
                    return 0;
                }
                let d = (rng.u32(p.geom.canon(c, mirrored), 0, 0) & 0xFFFF) as u64;
                let add = (l + d) >> 16;
                f[F_F] = (f[F_F] as u64 + add).min(u32::MAX as u64) as u32;
                f[F_T] = f[F_T].saturating_add(((l * p.sun_heat_q16) >> 32) as u32);
                add
            })
            .sum();
        st.tally.energy_source += injected as i64;
    }
    vent_inject(st, p);
    vent_update(st, p);
}

/// Inject free energy, heat and X4 at every active vent (sequential: two
/// vents may share a cell). Logs the ledger sources.
pub fn vent_inject(st: &mut State, p: &Params) {
    if !p.vents_on {
        return;
    }
    let vc = &p.vent_cfg;
    for vi in 0..st.vents.len() {
        let c = st.vents[vi].pos as usize;
        if !p.chem_only {
            *st.fm(c, F_F) = st.f(c, F_F).saturating_add(vc.rate);
        }
        *st.fm(c, F_T) = st.f(c, F_T).saturating_add(vc.heat);
        if let Some(x4) = p.x4_species {
            for h in 0..2 {
                let i = p.f_chem(x4, h);
                *st.fm(c, i) += vc.x4_rate;
            }
        }
    }
    let nv = st.vents.len() as i64;
    if !p.chem_only {
        st.tally.energy_source += nv * vc.rate as i64;
    }
    if let Some(x4) = p.x4_species {
        st.tally.atom_source += nv * 2 * vc.x4_rate as i64 * p.mass[x4] as i64;
    }
}

/// Age, drift, retire and spawn vents (CPU-only bookkeeping).
pub fn vent_update(st: &mut State, p: &Params) {
    if !p.vents_on {
        return;
    }
    let g = &p.geom;
    let rng = Rng::new(p.seed, st.epoch, phase::SOURCES);
    let mirrored = st.mirrored;
    let vc = &p.vent_cfg;
    let mut vents = std::mem::take(&mut st.vents);
    for v in vents.iter_mut() {
        let c = v.pos as usize;
        v.age += 1;
        if (rng.u32(v.id, 2, 0) as u64) < to_q32(vc.drift) && g.dims > 0 {
            let nd = if g.dims == 3 && vc.floor_only { 4 } else { g.faces() };
            let mut d = rng.u32(v.id, 3, 0) % nd;
            if mirrored {
                d = mirror_dir(d);
            }
            if let Some(t) = g.neighbor(c, d) {
                v.pos = t as u32;
            }
        }
    }
    vents.retain(|v| v.age < v.lifetime);
    if (rng.u32(0, 1, 0) as u64) < to_q32(vc.spawn_rate) {
        let pos = random_vent_pos(p, st, &rng, 1_000_000 + st.next_vent);
        vents.push(Vent { id: st.next_vent, pos: pos as u32, age: 0, lifetime: vc.lifetime });
        st.next_vent += 1;
    }
    st.vents = vents;
}

// ------------------------------------------------------------ phase 3

#[derive(Clone, Copy)]
pub struct FluxSpec {
    pub kappa: i64,
    pub gamma: Option<i64>,
    pub bonded_only: bool,
    pub stream: u32,
    /// value of the reservoir behind absorbing boundaries (None = no absorption)
    pub outside: Option<u32>,
}

/// Conservative antisymmetric per-face transport of one quantity stored at
/// `vals[c*stride + off]`. Returns the net amount lost through absorbing
/// boundaries.
pub fn flux(vals: &mut [u32], stride: usize, off: usize, spec: FluxSpec, bonds: &[u8], walls: &[u8], p: &Params, rng: &Rng, mirrored: bool) -> i64 {
    flux_batch(vals, stride, &[(off, spec, 0)], spec.stream, bonds, walls, p, rng, mirrored)[0]
}

/// Transport up to 8 quantities at once. Each entry is (offset, spec, dither
/// slot); one Philox call per face supplies eight 16-bit dithers. Mirror twins
/// pass the hand-swapped slot so both worlds draw the same dither for
/// mirror-image fields.
#[allow(clippy::too_many_arguments)]
pub fn flux_batch(vals: &mut [u32], stride: usize, fl: &[(usize, FluxSpec, usize)], stream: u32, bonds: &[u8], walls: &[u8], p: &Params, rng: &Rng, mirrored: bool) -> Vec<i64> {
    let g = &p.geom;
    let dims = g.dims as usize;
    let nfld = fl.len();
    if dims == 0 || nfld == 0 {
        return vec![0; nfld];
    }
    let n = g.n;
    let canon = |c: usize| g.canon(c, mirrored);
    let dith = |w: &[u32; 4], slot: usize| (w[slot / 2] >> (16 * (slot % 2))) & 0xFFFF;
    let any_out = fl.iter().any(|f| f.1.outside.is_some());
    let fs = dims * nfld;
    // pass A: raw face fluxes and boundary outflows
    let mut jf = vec![0i32; n * fs];
    let mut bo = vec![0u32; n * nfld];
    jf.par_chunks_mut(fs).zip(bo.par_chunks_mut(nfld)).enumerate().for_each(|(c, (j, b))| {
        if walls[c] != 0 {
            return;
        }
        for a in 0..dims {
            let t = p.nbr[c * 6 + 2 * a];
            if t == NONE || walls[t as usize] != 0 {
                continue;
            }
            let t = t as usize;
            let bonded = bonds[c] & (1 << (2 * a)) != 0;
            let (id, flip) = if mirrored && a == 0 { (canon(t) * 3, true) } else { (canon(c) * 3 + a as u32, false) };
            let w = rng.x4(id, stream, 0);
            for (k, (off, spec, slot)) in fl.iter().enumerate() {
                if spec.kappa == 0 || (spec.bonded_only && !bonded) {
                    continue;
                }
                let kk = match spec.gamma {
                    Some(gm) if bonded => (spec.kappa * gm) >> 16,
                    _ => spec.kappa,
                } as u32;
                let mut r = dith(&w, *slot);
                if flip {
                    r = 0xFFFF - r;
                }
                j[a * nfld + k] = flux_j(kk, vals[c * stride + off], vals[t * stride + off], r);
            }
        }
        if any_out {
            for d in 0..2 * dims as u32 {
                let t = p.nbr[c * 6 + d as usize];
                let absorbing = if t == NONE { g.boundary[(d / 2) as usize] == Boundary::Absorbing } else { walls[t as usize] == WALL_ABSORB as u8 };
                if !absorbing {
                    continue;
                }
                let cd = if mirrored { mirror_dir(d) } else { d };
                let w = rng.x4(canon(c) * 8 + cd, stream | 0x8000, 0);
                for (k, (off, spec, slot)) in fl.iter().enumerate() {
                    if let Some(out) = spec.outside {
                        // only outflow is counted (the reservoir never pushes matter in)
                        let x = flux_j(spec.kappa as u32, vals[c * stride + off], out, dith(&w, *slot));
                        if x > 0 {
                            b[k] = b[k].wrapping_add(x as u32);
                        }
                    }
                }
            }
        }
    });
    // pass B: total outflow per cell and field (saturating)
    let outs: Vec<u32> = (0..n)
        .into_par_iter()
        .flat_map_iter(|c| {
            let jf = &jf;
            let bo = &bo;
            (0..nfld).map(move |k| {
                let mut o = bo[c * nfld + k];
                for a in 0..dims {
                    let x = jf[c * fs + a * nfld + k];
                    if x > 0 {
                        o = sat_add(o, x as u32);
                    }
                    let s = p.nbr[c * 6 + 2 * a + 1];
                    if s != NONE {
                        let y = jf[s as usize * fs + a * nfld + k];
                        if y < 0 {
                            o = sat_add(o, y.unsigned_abs());
                        }
                    }
                }
                o
            })
        })
        .collect();
    // pass C: limit outflows so no cell goes negative
    {
        let vals = &*vals;
        jf.par_chunks_mut(fs).zip(bo.par_chunks_mut(nfld)).enumerate().for_each(|(c, (j, b))| {
            for (k, (off, _, _)) in fl.iter().enumerate() {
                let qc = vals[c * stride + off];
                for a in 0..dims {
                    let x = j[a * nfld + k];
                    if x > 0 {
                        j[a * nfld + k] = flux_lim(x as u32, qc, outs[c * nfld + k]) as i32;
                    } else if x < 0 {
                        let t = p.nbr[c * 6 + 2 * a] as usize;
                        j[a * nfld + k] = -(flux_lim(x.unsigned_abs(), vals[t * stride + off], outs[t * nfld + k]) as i32);
                    }
                }
                b[k] = flux_lim(b[k], qc, outs[c * nfld + k]);
            }
        });
    }
    // pass D: apply (wrapping u32 arithmetic; the limits keep the true value >= 0)
    let lost = vals
        .par_chunks_mut(stride)
        .enumerate()
        .map(|(c, v)| {
            let mut l = vec![0i64; nfld];
            for (k, (off, _, _)) in fl.iter().enumerate() {
                let mut q = v[*off].wrapping_sub(bo[c * nfld + k]);
                for a in 0..dims {
                    q = q.wrapping_sub(jf[c * fs + a * nfld + k] as u32);
                    let s = p.nbr[c * 6 + 2 * a + 1];
                    if s != NONE {
                        q = q.wrapping_add(jf[s as usize * fs + a * nfld + k] as u32);
                    }
                }
                v[*off] = q;
                l[k] = bo[c * nfld + k] as i64;
            }
            l
        })
        .reduce(|| vec![0i64; nfld], |a, b| a.iter().zip(&b).map(|(x, y)| x + y).collect());
    lost
}

pub fn bond_masks(st: &State) -> Vec<u8> {
    (0..st.n).map(|c| st.bonds(c) as u8).collect()
}
pub fn wall_masks(st: &State) -> Vec<u8> {
    (0..st.n).map(|c| st.site[c * SITE + S_WALL] as u8).collect()
}

pub fn transport(st: &mut State, p: &Params) {
    if p.geom.dims == 0 {
        return;
    }
    let rng = Rng::new(p.seed, st.epoch, phase::TRANSPORT);
    let bonds = bond_masks(st);
    let walls = wall_masks(st);
    let nf = st.nf;
    let mirrored = st.mirrored;
    for (bi, batch) in transport_batches(p, mirrored).iter().enumerate() {
        let fl: Vec<(usize, FluxSpec, usize)> = batch.iter().map(|&(f, spec, slot, _)| (f, spec, slot)).collect();
        let lost = flux_batch(&mut st.fields, nf, &fl, bi as u32, &bonds, &walls, p, &rng, mirrored);
        for (k, &(i, _, _, kind)) in batch.iter().enumerate() {
            let l = lost[k];
            match kind {
                0 => st.tally.energy_sink += l,
                2 => st.tally.atom_sink += l * p.mass[(i - 2) / 2] as i64,
                3 => st.tally.matter_sink[i - p.f_pool(0)] += l,
                _ => {}
            }
        }
    }
    // dissipation, heat relaxation, signal decay
    let t_amb = p.t_amb;
    let diss: i64 = st
        .fields
        .par_chunks_mut(nf)
        .enumerate()
        .map(|(c, f)| {
            let id = p.geom.canon(c, mirrored);
            let r = rng.x4(id, 0x10000, 0);
            let mut lost = 0;
            if p.energy_on && p.diss_q16 > 0 {
                let l = mad_shr16(f[F_F], p.diss_q16 as u32, r[0] & 0xFFFF);
                f[F_F] -= l;
                f[F_T] = sat_add(f[F_T], mulshr16(l, p.heat_q16 as u32));
                lost = l as i64;
            }
            if p.relax_q16 != 0 {
                f[F_T] = relax_t(f[F_T], t_amb, p.relax_q16 as u32, r[1] & 0xFFFF);
            }
            for j in 0..p.s {
                for h in 0..2 {
                    let i = p.f_sig(j, h);
                    let hc = if mirrored { 1 - h } else { h };
                    let w = rng.u32(id, 0x10100 + (j * 2 + hc) as u32, 0) & 0xFFFF;
                    let l = mad_shr16(f[i], p.sig_decay_q16 as u32, w);
                    f[i] -= l.min(f[i]);
                }
            }
            lost
        })
        .sum();
    st.tally.dissipation += diss;
}

/// The transport plan: batches of <= 8 (field, spec, dither slot, kind)
/// sharing one Philox call per face. L/D pairs are never split, so the
/// hand-swapped dither slot of a field is always in its own batch.
/// kind: 0 energy, 1 heat, 2 chemical, 3 monomer pool, 4 signal.
pub fn transport_batches(p: &Params, mirrored: bool) -> Vec<Vec<(usize, FluxSpec, usize, u8)>> {
    let g = if p.bonds_on { Some(p.gamma_q16) } else { None };
    let mut specs: Vec<(usize, FluxSpec, u8)> = vec![];
    if p.energy_on {
        specs.push((F_F, FluxSpec { kappa: p.kappa_f, gamma: None, bonded_only: false, stream: 0, outside: Some(0) }, 0));
    }
    specs.push((F_T, FluxSpec { kappa: p.kappa_t, gamma: None, bonded_only: false, stream: 1, outside: Some(p.t_amb) }, 1));
    for k in 0..p.k {
        for h in 0..2 {
            let i = p.f_chem(k, h);
            specs.push((i, FluxSpec { kappa: p.kappa_c, gamma: g, bonded_only: false, stream: i as u32, outside: Some(0) }, 2));
        }
    }
    if p.matter_on {
        for cl in 0..3 {
            let i = p.f_pool(cl);
            specs.push((i, FluxSpec { kappa: p.kappa_m, gamma: g, bonded_only: false, stream: i as u32, outside: Some(0) }, 3));
        }
    }
    for j in 0..p.s {
        for h in 0..2 {
            let i = p.f_sig(j, h);
            specs.push((i, FluxSpec { kappa: p.kappa_s, gamma: None, bonded_only: false, stream: i as u32, outside: Some(0) }, 4));
        }
    }
    let mut units: Vec<Vec<(usize, FluxSpec, u8)>> = vec![];
    let mut i = 0;
    while i < specs.len() {
        let f = specs[i].0;
        let paired = i + 1 < specs.len() && swap_hand_field(p, f) == specs[i + 1].0 && swap_hand_field(p, f) != f;
        if paired {
            units.push(vec![specs[i], specs[i + 1]]);
            i += 2;
        } else {
            units.push(vec![specs[i]]);
            i += 1;
        }
    }
    let mut batches: Vec<Vec<(usize, FluxSpec, u8)>> = vec![vec![]];
    for u in units {
        if batches.last().unwrap().len() + u.len() > 8 {
            batches.push(vec![]);
        }
        batches.last_mut().unwrap().extend(u);
    }
    batches
        .into_iter()
        .map(|batch| {
            let fields: Vec<usize> = batch.iter().map(|x| x.0).collect();
            batch
                .iter()
                .map(|&(f, spec, kind)| {
                    let src = if mirrored { swap_hand_field(p, f) } else { f };
                    (f, spec, fields.iter().position(|&x| x == src).unwrap(), kind)
                })
                .collect()
        })
        .collect()
}

/// Heat relaxation T + floor((relax*(T_amb - T) + d) / 2^16), clamped to u32.
pub fn relax_t(t: u32, t_amb: u32, relax: u32, d: u32) -> u32 {
    let j = flux_j(relax, t_amb, t, d);
    if j >= 0 {
        sat_add(t, j as u32)
    } else if j.unsigned_abs() > t {
        0
    } else {
        t - j.unsigned_abs()
    }
}

/// Field index under L<->D exchange.
pub fn swap_hand_field(p: &Params, i: usize) -> usize {
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
}

// ------------------------------------------------------------ phase 4

pub fn chemistry(st: &mut State, p: &Params) {
    if !p.chem_on && !p.matter_on {
        return;
    }
    let rng = Rng::new(p.seed, st.epoch, phase::CHEM);
    let mirrored = st.mirrored;
    let nf = st.nf;
    let light = &st.light;
    let walls: Vec<u8> = wall_masks(st);
    let i0i = ((p.sun_i0_q16 >> 16) as u32).max(1);
    #[derive(Default, Clone, Copy)]
    struct T {
        rxn: i64,
        sink: i64,
        msrc: [i64; 2],
        msink: [i64; 2],
    }
    let tot = st
        .fields
        .par_chunks_mut(nf)
        .enumerate()
        .map(|(c, f)| {
            let mut t = T::default();
            if walls[c] != 0 {
                return t;
            }
            let id = p.geom.canon(c, mirrored);
            let ch = |h: usize| if mirrored { 1 - h } else { h };
            let temp = f[F_T];
            if p.chem_on {
                for (ri, rx) in p.reactions.iter().enumerate() {
                    let a = arr_lookup(&p.arr[rx.arr], temp);
                    let rate = if rx.photo { mulshr16_sat(rx.rate_q32, light[c] / i0i) } else { mulshr16_sat(rx.rate_q32, a) };
                    // desired events per hand
                    let mut ev = [0u32; 2];
                    for h in 0..2 {
                        let fi = |k: u32| 2 + 2 * k as usize + h;
                        let c0 = f[fi(rx.r[0])];
                        let d = rng.u32(id, (ri * 2 + ch(h)) as u32, 0);
                        let prod = if rx.photo || rx.r[1] == NONE { c0 } else { sat32(c0 as u64 * f[fi(rx.r[1])] as u64) };
                        let cap = if rx.r[1] == NONE {
                            c0
                        } else if rx.r[1] == rx.r[0] {
                            c0 / 2
                        } else {
                            c0.min(f[fi(rx.r[1])])
                        };
                        ev[h] = mad_shr32(rate, prod, d).min(cap);
                    }
                    // endothermic reactions draw from F: share the cap symmetrically
                    if rx.energy < 0 {
                        let e = rx.energy.unsigned_abs();
                        let need = sat32((ev[0] as u64 + ev[1] as u64) * e as u64);
                        let have = if p.energy_on { f[F_F] } else { 0 };
                        if need > have {
                            let r = ratio16(have, need);
                            for h in 0..2 {
                                ev[h] = mulshr16(ev[h], r);
                            }
                        }
                        let used = (ev[0] + ev[1]) * e;
                        f[F_F] -= used;
                        t.rxn -= used as i64;
                    }
                    for h in 0..2 {
                        let e = ev[h];
                        if e == 0 {
                            continue;
                        }
                        for k in rx.r {
                            if k != NONE {
                                f[2 + 2 * k as usize + h] -= e;
                            }
                        }
                        for k in rx.p {
                            if k != NONE {
                                f[2 + 2 * k as usize + h] += e;
                            }
                        }
                        if rx.energy > 0 {
                            let heat = mulshr16(e, rx.energy_q16[h].max(0) as u32);
                            f[F_T] = sat_add(f[F_T], mulshr16(heat, p.rxn_heat_q16 as u32));
                        }
                    }
                }
                // global sink
                if p.sink_q16 > 0 {
                    for k in 0..p.k {
                        for h in 0..2 {
                            let i = 2 + 2 * k + h;
                            let d = rng.u32(id, (500 + k * 2 + ch(h)) as u32, 0) & 0xFFFF;
                            let l = mad_shr16(f[i], p.sink_q16 as u32, d);
                            f[i] -= l;
                            t.sink += l as i64 * p.mass[k] as i64;
                        }
                    }
                }
            }
            // racemization of chemicals and free monomers
            if p.rac_q32 > 0 {
                let rate = mulshr16_sat(p.rac_q32.min(u32::MAX as u64) as u32, arr_lookup(&p.arr[p.arr_rac], temp));
                let r_ld = mulshr16(rate, p.rac_bias_q16 as u32);
                let rac = |li: usize, di: usize, sid: usize, f: &mut [u32]| -> (u32, u32) {
                    let (cl, cd) = (f[li], f[di]);
                    let dl = rng.u32(id, (1000 + sid * 2 + ch(0)) as u32, 0);
                    let dd = rng.u32(id, (1000 + sid * 2 + ch(1)) as u32, 0);
                    let n_ld = mad_shr32(r_ld, cl, dl).min(cl);
                    let n_dl = mad_shr32(rate, cd, dd).min(cd);
                    f[li] = f[li] - n_ld + n_dl;
                    f[di] = f[di] - n_dl + n_ld;
                    (n_ld, n_dl)
                };
                if p.chem_on {
                    for k in 0..p.k {
                        rac(2 + 2 * k, 3 + 2 * k, k, f);
                    }
                }
                if p.matter_on {
                    let (li, di) = (p.f_pool(0), p.f_pool(1));
                    let (ld, dl) = rac(li, di, 100, f);
                    t.msink[0] += ld as i64;
                    t.msrc[1] += ld as i64;
                    t.msink[1] += dl as i64;
                    t.msrc[0] += dl as i64;
                }
            }
            t
        })
        .reduce(T::default, |a, b| T {
            rxn: a.rxn + b.rxn,
            sink: a.sink + b.sink,
            msrc: [a.msrc[0] + b.msrc[0], a.msrc[1] + b.msrc[1]],
            msink: [a.msink[0] + b.msink[0], a.msink[1] + b.msink[1]],
        });
    st.tally.reaction_net += tot.rxn;
    st.tally.atom_sink += tot.sink;
    for h in 0..2 {
        st.tally.matter_source[h] += tot.msrc[h];
        st.tally.matter_sink[h] += tot.msink[h];
    }
}

// ------------------------------------------------------------ phase 5

pub fn absorb_upkeep(st: &mut State, p: &Params) {
    let n = st.n;
    // clear per-epoch flags first (always, even without energy)
    st.hdr.par_chunks_mut(HDR).for_each(|h| h[H_FLAGS] &= !(FL_EXEC | FL_NEWBORN));
    if !p.energy_on {
        st.hdr.par_chunks_mut(HDR).for_each(|h| {
            if h[H_SLEEP] > 0 {
                h[H_SLEEP] -= 1;
            }
        });
        return;
    }
    let expo = exposure(st, p);
    let rng = Rng::new(p.seed, st.epoch, phase::ABSORB);
    let mirrored = st.mirrored;
    let nf = st.nf;
    let spent: i64 = st
        .hdr
        .par_chunks_mut(HDR)
        .zip(st.fields.par_chunks_mut(nf))
        .enumerate()
        .map(|(c, (h, f))| {
            if h[H_FLAGS] & FL_OCC == 0 {
                return 0;
            }
            // absorption
            let a = mulshr16(p.absorb_q16 as u32, expo[c]);
            let de = mulshr16(a, f[F_F]);
            f[F_F] -= de;
            h[H_E] = sat_add(h[H_E], de);
            // upkeep
            let d = rng.u32(p.geom.canon(c, mirrored), 0, 0) & 0xFFFF;
            let mut up = (p.upkeep_base_q16 as u32).wrapping_add((p.upkeep_byte_q16 as u32).wrapping_mul(h[H_LEN]));
            if h[H_SLEEP] > 0 {
                up /= 8;
                h[H_SLEEP] -= 1;
            }
            let up = (up.wrapping_add(d)) >> 16;
            let paid = up.min(h[H_E]);
            h[H_E] -= paid;
            f[F_T] = sat_add(f[F_T], mulshr16(paid, p.heat_q16 as u32));
            paid as i64
        })
        .sum();
    st.tally.spending += spent;
    // bonded energy equalisation
    if p.bonds_on && p.beta_q16 > 0 {
        let bonds = bond_masks(st);
        let walls = wall_masks(st);
        let spec = FluxSpec { kappa: p.beta_q16, gamma: None, bonded_only: true, stream: 0x20000, outside: None };
        flux(&mut st.hdr, HDR, H_E, spec, &bonds, &walls, p, &rng, mirrored);
    }
    let wake = p.wake_energy;
    st.hdr.par_chunks_mut(HDR).for_each(|h| {
        if h[H_FLAGS] & FL_OCC == 0 {
            return;
        }
        if h[H_E] == 0 {
            h[H_FLAGS] |= FL_HALT;
        } else if h[H_FLAGS] & FL_HALT != 0 && h[H_E] >= wake {
            h[H_FLAGS] &= !FL_HALT;
        }
    });
    let _ = n;
}

// ------------------------------------------------------------ phase 7

pub struct Bodies {
    /// component label per cell (min canonical index), NONE for unbonded/empty
    pub label: Vec<u32>,
    pub count: usize,
    pub sizes: Vec<(u32, u32)>,
}

fn find(par: &mut [u32], mut x: u32) -> u32 {
    while par[x as usize] != x {
        let nx = par[x as usize];
        par[x as usize] = par[nx as usize];
        x = nx;
    }
    x
}

/// Bond-graph components. Labels are the minimum canonical index of the
/// members, so they are identical between mirror twins.
pub fn components(st: &State, p: &Params) -> Bodies {
    let n = st.n;
    let mut par: Vec<u32> = (0..n as u32).collect();
    let dims = p.geom.dims as usize;
    for c in 0..n {
        let b = st.bonds(c);
        if b == 0 {
            continue;
        }
        for a in 0..dims {
            if b & (1 << (2 * a)) != 0 {
                let t = p.nbr[c * 6 + 2 * a];
                if t != NONE {
                    let (x, y) = (find(&mut par, c as u32), find(&mut par, t));
                    if x != y {
                        let (lo, hi) = if x < y { (x, y) } else { (y, x) };
                        par[hi as usize] = lo;
                    }
                }
            }
        }
    }
    let mut minc = vec![NONE; n];
    let mut size = vec![0u32; n];
    for c in 0..n {
        if st.bonds(c) == 0 {
            continue;
        }
        let r = find(&mut par, c as u32) as usize;
        let cc = p.geom.canon(c, st.mirrored);
        if cc < minc[r] {
            minc[r] = cc;
        }
        size[r] += 1;
    }
    let mut label = vec![NONE; n];
    let mut sizes = vec![];
    for c in 0..n {
        if st.bonds(c) != 0 {
            let r = find(&mut par, c as u32) as usize;
            label[c] = minc[r];
        }
    }
    for r in 0..n {
        if size[r] > 0 && par[r] == r as u32 {
            sizes.push((minc[r], size[r]));
        }
    }
    let count = sizes.len();
    Bodies { label, count, sizes }
}

pub fn body_move(st: &mut State, p: &Params) {
    if !p.bonds_on || p.geom.dims == 0 {
        st.site.par_chunks_mut(SITE).for_each(|s| s[S_VOTE] = 0);
        return;
    }
    let any_vote = st.site.par_chunks(SITE).any(|s| s[S_VOTE] != 0);
    if !any_vote {
        return;
    }
    let bodies = components(st, p);
    use std::collections::BTreeMap;
    let mut members: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for c in 0..st.n {
        if bodies.label[c] != NONE {
            members.entry(bodies.label[c]).or_default().push(c);
        }
    }
    let mut cand: Vec<(u32, u32)> = vec![]; // (label, dir)
    for (&lab, mem) in &members {
        let mut votes = [0u64; 6];
        for &c in mem {
            let v = st.site[c * SITE + S_VOTE];
            if v != 0 {
                votes[(v - 1) as usize] += 1;
            }
        }
        let size = mem.len() as u64;
        if size > p.max_body as u64 {
            continue;
        }
        let best = (0..6).max_by_key(|&d| votes[d]).unwrap();
        let bv = votes[best];
        if bv == 0 || (0..6).any(|d| d != best && votes[d] >= bv) {
            continue;
        }
        if (bv << 16) < p.vote_q16 * size {
            continue;
        }
        let d = best as u32;
        let ok = mem.iter().all(|&c| match crate::exec::nb(st, p, c, d) {
            Some(t) => !st.occupied(t) || bodies.label[t] == lab,
            None => false,
        });
        if ok {
            cand.push((lab, d));
        }
    }
    // claims: cancel any candidate whose destination is claimed by another
    let mut claims: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
    for &(lab, d) in &cand {
        for &c in &members[&lab] {
            let t = crate::exec::nb(st, p, c, d).unwrap();
            *claims.entry(t).or_insert(0) += 1;
        }
    }
    for &(lab, d) in &cand {
        let mem = &members[&lab];
        if mem.iter().any(|&c| claims[&crate::exec::nb(st, p, c, d).unwrap()] > 1) {
            continue;
        }
        // translate
        let l = st.lmax;
        let data: Vec<(usize, Vec<u32>, Vec<u8>)> = mem
            .iter()
            .map(|&c| (crate::exec::nb(st, p, c, d).unwrap(), st.hdr[c * HDR..(c + 1) * HDR].to_vec(), st.tapes[c * l..(c + 1) * l].to_vec()))
            .collect();
        for &c in mem {
            st.hdr[c * HDR..(c + 1) * HDR].iter_mut().for_each(|x| *x = 0);
            st.tapes[c * l..(c + 1) * l].iter_mut().for_each(|x| *x = 0);
        }
        for (t, h, tp) in data {
            st.hdr[t * HDR..(t + 1) * HDR].copy_from_slice(&h);
            st.tapes[t * l..(t + 1) * l].copy_from_slice(&tp);
        }
        st.tally.body_moves += 1;
    }
    st.site.par_chunks_mut(SITE).for_each(|s| s[S_VOTE] = 0);
}

// ------------------------------------------------------------ phase 8

/// Spatial mutation multiplier map (Q16).
pub fn mutation_map(st: &State, p: &Params, image: Option<&Vec<u32>>) -> Option<Vec<u32>> {
    match p.map_kind {
        1 => {
            let g = &p.geom;
            let mut m = vec![1u32 << 16; st.n];
            let r0 = p.map_r0.max(0.1);
            let rmax = (5.0 * r0) as i64 + 1;
            let table: Vec<u32> = (0..=(rmax * rmax * 3) as usize)
                .map(|r2| {
                    // sqrt by bisection (basic ops only)
                    let x = r2 as f64;
                    let (mut lo, mut hi) = (0.0f64, x.max(1.0));
                    for _ in 0..60 {
                        let mid = 0.5 * (lo + hi);
                        if mid * mid > x { hi = mid } else { lo = mid }
                    }
                    to_q16(p.map_a * det_exp(-lo / r0))
                })
                .collect();
            for v in &st.vents {
                let pc = g.coords(v.pos as usize);
                let dims = g.dims as usize;
                let rng_ax = |a: usize| {
                    let h = rmax.min((g.size[a] as i64 - 1) / 2);
                    if a < dims { -h..=h } else { 0..=0 }
                };
                for dz in rng_ax(2) {
                    for dy in rng_ax(1) {
                        for dx in rng_ax(0) {
                            let d = [dx, dy, dz];
                            let mut q = [0u32; 3];
                            let mut ok = true;
                            for a in 0..3 {
                                let s = g.size[a] as i64;
                                let mut x = pc[a] as i64 + d[a];
                                if x < 0 || x >= s {
                                    if a < dims && g.boundary[a] == Boundary::Torus {
                                        x = x.rem_euclid(s);
                                    } else {
                                        ok = false;
                                    }
                                }
                                q[a] = x as u32;
                            }
                            if !ok {
                                continue;
                            }
                            let r2 = (dx * dx + dy * dy + dz * dz) as usize;
                            let c = g.index(q);
                            m[c] = m[c].saturating_add(table[r2]);
                        }
                    }
                }
            }
            Some(m)
        }
        2 => image.cloned(),
        _ => None,
    }
}

/// Expected events (Q24) for the four per-byte channels of one cell:
/// background flips, radiation flips, hand flips, decay. Shared with WGSL.
#[allow(clippy::too_many_arguments)]
pub fn mutation_lambdas(p: &Params, n_ne: u32, n_ch: u32, temp: u32, e: u32, light: u32, mx: u32, i0i: u32) -> [u32; 4] {
    let irel = light / i0i;
    let a_mut = arr_lookup(&p.arr[p.arr_mut], temp);
    let a_rac = arr_lookup(&p.arr[p.arr_rac], temp);
    let a_dec = arr_lookup(&p.arr[p.arr_decay], temp);
    let bg = mulshr16_sat(mulshr16_sat(p.mu_bg_q32, a_mut), mx);
    let uv = mulshr16_sat(mulshr16(p.mu_uv_q32, e), irel);
    let fl = mulshr16_sat(p.mu_flip_q32, a_rac);
    let lf = 65536u32.saturating_add(mulshr16_sat(mulshr16(p.decay_light_q16 as u32, e), irel));
    let dc = mulshr16_sat(mulshr16_sat(p.decay_q32, a_dec), lf);
    [mulshr8_sat(bg, n_ne), mulshr8_sat(uv, n_ne), mulshr8_sat(fl, n_ch), mulshr8_sat(dc, n_ne)]
}

pub fn mutate_decay(st: &mut State, p: &Params, map: Option<&Vec<u32>>) {
    let expo = exposure(st, p);
    let rng = Rng::new(p.seed, st.epoch, phase::MUTATE);
    let mirrored = st.mirrored;
    let lmax = st.lmax;
    let nf = st.nf;
    let light = &st.light;
    let i0i = ((p.sun_i0_q16 >> 16) as u32).max(1);
    let matter = p.matter_on;
    let flip_frame = |v: u8, bit: u32| -> u8 { if mirrored { isa::mirror(isa::mirror(v) ^ (1 << bit)) } else { v ^ (1 << bit) } };
    st.tapes
        .par_chunks_mut(lmax)
        .zip(st.hdr.par_chunks_mut(HDR))
        .zip(st.fields.par_chunks_mut(nf))
        .enumerate()
        .for_each(|(c, ((tape, h), f))| {
            if h[H_FLAGS] & FL_OCC == 0 {
                return;
            }
            let len = h[H_LEN] as usize;
            let t = &mut tape[..len];
            let n_ne = t.iter().filter(|&&b| b != 0).count() as u32;
            let n_ch = t.iter().filter(|&&b| isa::is_chiral(b)).count() as u32;
            let temp = f[F_T];
            let id = p.geom.canon(c, mirrored);
            let lam = mutation_lambdas(p, n_ne, n_ch, temp, expo[c], light[c], map.map(|m| m[c]).unwrap_or(1 << 16), i0i);
            let mut draw = 0u32;
            let mut rnd = || {
                let r = rng.u32(id, 0, draw);
                draw += 1;
                r
            };
            for (ch, &l) in lam.iter().enumerate() {
                if l == 0 {
                    continue;
                }
                let k = poisson24(&p.pois, l, rnd(), 64);
                for _ in 0..k {
                    let elig = |b: u8| if ch == 2 { isa::is_chiral(b) } else { b != 0 };
                    let ne = t.iter().filter(|&&b| elig(b)).count();
                    if ne == 0 {
                        break;
                    }
                    let j = (rnd() as usize) % ne;
                    let pos = t.iter().enumerate().filter(|(_, b)| elig(**b)).nth(j).unwrap().0;
                    let old = t[pos];
                    let new = match ch {
                        0 | 1 => flip_frame(old, rnd() % 8),
                        2 => isa::mirror(old),
                        _ => 0,
                    };
                    if matter {
                        let co = isa::class(old) as usize;
                        let cn = isa::class(new) as usize;
                        if co < 3 {
                            f[2 + 2 * p.k + co] += 1;
                        }
                        if cn < 3 {
                            let pi = 2 + 2 * p.k + cn;
                            if f[pi] == 0 {
                                if co < 3 {
                                    f[2 + 2 * p.k + co] -= 1;
                                }
                                continue;
                            }
                            f[pi] -= 1;
                        }
                    }
                    t[pos] = new;
                }
            }
            // dead-cell bookkeeping
            // dying counter: all-empty tape, or (with starvation on) dormant
            let starving = p.starve_after > 0 && h[H_FLAGS] & FL_HALT != 0;
            if starving || t.iter().all(|&b| b == 0) {
                h[H_ZERO] += 1;
            } else {
                h[H_ZERO] = 0;
            }
        });
    // thermal bond breaking
    if p.bonds_on && p.bond_break_q32 > 0 && p.geom.dims > 0 {
        let rngb = Rng::new(p.seed, st.epoch, phase::BONDBREAK);
        let dims = p.geom.dims as usize;
        let breaks: Vec<(usize, usize)> = (0..st.n)
            .into_par_iter()
            .flat_map_iter(|c| {
                let b = st.bonds(c);
                let mut out = vec![];
                for a in 0..dims {
                    if b & (1 << (2 * a)) == 0 {
                        continue;
                    }
                    let t = p.nbr[c * 6 + 2 * a] as usize;
                    let tm = (st.f(c, F_T) / 2 + st.f(t, F_T) / 2) as u32;
                    let pr = mulshr16_sat(p.bond_break_q32, arr_lookup(&p.arr[p.arr_bond], tm));
                    let id = if mirrored && a == 0 { p.geom.canon(t, true) * 3 } else { p.geom.canon(c, mirrored) * 3 + a as u32 };
                    if rngb.u32(id, 0, 0) < pr {
                        out.push((c, a));
                    }
                }
                out
            })
            .collect();
        for (c, a) in breaks {
            let t = p.nbr[c * 6 + 2 * a] as usize;
            st.hdr[c * HDR + H_FLAGS] &= !(1 << (BOND_SHIFT + 2 * a as u32));
            st.hdr[t * HDR + H_FLAGS] &= !(1 << (BOND_SHIFT + 2 * a as u32 + 1));
        }
    }
    // dead-cell clearing (sequential: touches neighbours' bond bits)
    let dead: Vec<usize> = (0..st.n).into_par_iter().filter(|&c| st.occupied(c) && dying(st, p, c)).collect();
    for c in dead {
        clear_cell(st, p, c);
        st.tally.deaths += 1;
    }
}

/// The cell's dying counter has reached its limit: `dead_after` for an
/// all-empty tape, otherwise `starve_after` (if starvation is on).
pub fn dying(st: &State, p: &Params, c: usize) -> bool {
    let z = st.h(c, H_ZERO);
    if st.tape(c).iter().all(|&b| b == 0) {
        z >= p.dead_after
    } else {
        p.starve_after > 0 && z >= p.starve_after
    }
}

/// Remove the individual at c: tape monomers to the pool, energy to F,
/// bonds cut on both sides.
pub fn clear_cell(st: &mut State, p: &Params, c: usize) {
    if !st.occupied(c) {
        return;
    }
    let l = st.lmax;
    let len = st.len(c);
    for i in 0..len {
        let b = st.tapes[c * l + i];
        if p.matter_on {
            let cl = isa::class(b);
            if cl < 3 {
                *st.fm(c, p.f_pool(cl as usize)) += 1;
            }
        }
    }
    st.tapes[c * l..(c + 1) * l].iter_mut().for_each(|x| *x = 0);
    let e = st.h(c, H_E);
    *st.fm(c, F_F) = st.f(c, F_F).saturating_add(e);
    let bonds = st.bonds(c);
    for d in 0..6u32 {
        if bonds & (1 << d) != 0 {
            let t = p.nbr[c * 6 + d as usize];
            if t != NONE {
                st.hdr[t as usize * HDR + H_FLAGS] &= !(1 << (BOND_SHIFT + (d ^ 1)));
            }
        }
    }
    st.hdr[c * HDR..(c + 1) * HDR].iter_mut().for_each(|x| *x = 0);
}

// ------------------------------------------------------------ phase 9

/// Fold the per-site execution counters (spent energy, catalysis gains) into
/// the epoch tally and zero them.
pub fn collect_site_counters(st: &mut State) {
    let (sp, rx) = st
        .site
        .par_chunks_mut(SITE)
        .map(|s| {
            let r = (s[S_SPENT] as i64, s[S_RXN] as i64);
            s[S_SPENT] = 0;
            s[S_RXN] = 0;
            r
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    st.tally.spending += sp;
    st.tally.reaction_net += rx;
}

/// Check all ledgers against the tallies accumulated since the last check.
pub fn ledger_check(st: &mut State, p: &Params) {
    let e_now = st.energy_total();
    let m = st.matter_totals(p);
    let a = st.atom_total(p);
    ledger_apply(st, e_now, m, a);
}

/// Compare measured totals with the expected ones from the tallies.
pub fn ledger_apply(st: &mut State, e_now: i64, m: [i64; 3], a: i64) {
    let t = &st.tally;
    let expected = st.ledger.energy_prev + t.energy_source - t.energy_sink - t.dissipation - t.spending + t.reaction_net;
    st.ledger.energy_residual = e_now - expected;
    st.ledger.energy_prev = e_now;
    for cl in 0..3 {
        st.ledger.matter_expected[cl] += t.matter_source[cl] - t.matter_sink[cl];
        st.ledger.matter_residual[cl] = m[cl] - st.ledger.matter_expected[cl];
    }
    st.ledger.atoms_expected += t.atom_source - t.atom_sink;
    st.ledger.atom_residual = a - st.ledger.atoms_expected;
    if st.ledger.energy_residual != 0 || st.ledger.matter_residual.iter().any(|&x| x != 0) || st.ledger.atom_residual != 0 {
        st.ledger.violations += 1;
        // re-anchor so one bug does not cascade into every later epoch
        st.ledger.matter_expected = m;
        st.ledger.atoms_expected = a;
    }
}
