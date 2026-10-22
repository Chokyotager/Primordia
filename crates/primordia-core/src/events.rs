//! Interventions: logged events `{epoch, kind, params}` applied at phase 1.

use crate::geom::Boundary;
use crate::isa;
use crate::params::*;
use crate::phases::clear_cell;
use crate::rng::{Rng, phase};
use crate::state::{State, Vent};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Event {
    pub epoch: u64,
    pub kind: String,
    #[serde(default)]
    pub params: Value,
}

fn get_u(v: &Value, k: &str, d: u64) -> u64 {
    v.get(k).and_then(|x| x.as_u64()).unwrap_or(d)
}
fn get_f(v: &Value, k: &str, d: f64) -> f64 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(d)
}
fn get_xyz(v: &Value, k: &str) -> [i64; 3] {
    let mut p = [0i64; 3];
    if let Some(a) = v.get(k).and_then(|x| x.as_array()) {
        for (i, x) in a.iter().take(3).enumerate() {
            p[i] = x.as_i64().unwrap_or(0);
        }
    }
    p
}

/// Cells within Euclidean radius r of a centre (torus-aware).
pub fn region(p: &Params, center: [i64; 3], r: f64) -> Vec<usize> {
    let g = &p.geom;
    if g.dims == 0 {
        // 0D: "radius" = number of slots starting at center[0]
        let n = (r.max(1.0)) as usize;
        return (0..n).map(|i| (center[0] as usize + i) % g.n).collect();
    }
    let ri = r.ceil() as i64;
    let mut out = vec![];
    let dims = g.dims as usize;
    let span = |a: usize| if a < dims { -ri..=ri } else { 0..=0 };
    for dz in span(2) {
        for dy in span(1) {
            for dx in span(0) {
                if ((dx * dx + dy * dy + dz * dz) as f64) > r * r {
                    continue;
                }
                let d = [dx, dy, dz];
                let mut q = [0u32; 3];
                let mut ok = true;
                for a in 0..3 {
                    let s = g.size[a] as i64;
                    let mut x = center[a] + d[a];
                    if x < 0 || x >= s {
                        if a < dims && g.boundary[a] == Boundary::Torus {
                            x = x.rem_euclid(s);
                        } else {
                            ok = false;
                        }
                    }
                    q[a] = x as u32;
                }
                if ok {
                    out.push(g.index(q));
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Place a genome at cell c (replacing any occupant). Matter for the bytes is
/// a logged matter source.
pub fn place_genome(st: &mut State, p: &Params, c: usize, genome: &[u8], energy: u32, epoch: u64) {
    clear_cell(st, p, c);
    let l = st.lmax;
    let n = genome.len().min(l);
    st.tapes[c * l..c * l + n].copy_from_slice(&genome[..n]);
    if p.matter_on {
        for &b in &genome[..n] {
            let cl = isa::class(b);
            if cl < 3 {
                st.tally.matter_source[cl as usize] += 1;
            }
        }
    }
    let lin = Rng::new(p.seed, epoch, phase::INTERVENE).u32(c as u32, 9, 0) | 1;
    let e = if p.energy_on { energy } else { 0 };
    let h = &mut st.hdr[c * HDR..(c + 1) * HDR];
    h[H_LEN] = n as u32;
    h[H_FLAGS] = FL_OCC | (p.orient.valid[p.geom.dims as usize][0] << ORIENT_SHIFT);
    h[H_E] = e;
    h[H_LINEAGE] = lin;
    h[H_BIRTH] = epoch as u32;
    st.tally.energy_source += e as i64;
}

pub fn apply(st: &mut State, p: &Params, ev: &Event) -> Result<()> {
    let v = &ev.params;
    let g = &p.geom;
    match ev.kind.as_str() {
        "paint" => {
            let cells = region(p, get_xyz(v, "center"), get_f(v, "radius", 3.0));
            let amount = get_u(v, "amount", 100) as u32;
            let field = v.get("field").and_then(|x| x.as_str()).unwrap_or("F");
            let k = get_u(v, "species", 0) as usize;
            let h = get_u(v, "hand", 0) as usize;
            for c in cells {
                if st.wall(c) != 0 {
                    continue;
                }
                match field {
                    "F" => {
                        *st.fm(c, F_F) += amount;
                        st.tally.energy_source += amount as i64;
                    }
                    "E" => {
                        if st.occupied(c) {
                            *st.hm(c, H_E) += amount;
                            st.tally.energy_source += amount as i64;
                        }
                    }
                    "T" => *st.fm(c, F_T) += amount,
                    "chem" => {
                        anyhow::ensure!(k < p.k, "species out of range");
                        *st.fm(c, p.f_chem(k, h.min(1))) += amount;
                        st.tally.atom_source += amount as i64 * p.mass[k] as i64;
                    }
                    "pool" => {
                        anyhow::ensure!(p.matter_on, "matter is off");
                        let cl = h.min(2);
                        *st.fm(c, p.f_pool(cl)) += amount;
                        st.tally.matter_source[cl] += amount as i64;
                    }
                    "signal" => {
                        anyhow::ensure!(k < p.s, "signal out of range");
                        *st.fm(c, p.f_sig(k, h.min(1))) += amount;
                    }
                    f => bail!("unknown paint field {f}"),
                }
            }
        }
        "vent_add" => {
            let c = point(p, get_xyz(v, "center"));
            st.vents.push(Vent { id: st.next_vent, pos: c as u32, age: 0, lifetime: get_u(v, "lifetime", p.vent_cfg.lifetime) });
            st.next_vent += 1;
        }
        "vent_remove" => {
            let c = point(p, get_xyz(v, "center")) as u32;
            let r = get_f(v, "radius", 2.0);
            let near = region(p, get_xyz(v, "center"), r);
            st.vents.retain(|x| x.pos != c && !near.contains(&(x.pos as usize)));
        }
        "vent_move" => {
            let id = get_u(v, "id", 0) as u32;
            let c = point(p, get_xyz(v, "center")) as u32;
            for x in st.vents.iter_mut() {
                if x.id == id {
                    x.pos = c;
                }
            }
        }
        "seed" => {
            let genome = isa::parse_hex_genome(v.get("genome").and_then(|x| x.as_str()).context("seed needs genome")?)?;
            let cells = region(p, get_xyz(v, "center"), get_f(v, "radius", 0.0));
            let e = get_u(v, "energy", 1000) as u32;
            let mirror = v.get("mirror").and_then(|x| x.as_bool()).unwrap_or(false);
            for &c in &cells {
                if st.wall(c) == 0 {
                    place_genome(st, p, c, &genome, e, ev.epoch);
                }
            }
            if mirror {
                let mg: Vec<u8> = genome.iter().map(|&b| isa::mirror(b)).collect();
                for &c in &cells {
                    let m = g.mirror_cell(c);
                    if st.wall(m) == 0 {
                        place_genome(st, p, m, &mg, e, ev.epoch);
                        let o = p.orient.mirror[st.orient(c) as usize];
                        let f = st.hm(m, H_FLAGS);
                        *f = (*f & !ORIENT_MASK) | (o << ORIENT_SHIFT);
                    }
                }
            }
        }
        "clear" => {
            for c in region(p, get_xyz(v, "center"), get_f(v, "radius", 3.0)) {
                clear_cell(st, p, c);
            }
        }
        "wall" => {
            let kind = match v.get("type").and_then(|x| x.as_str()).unwrap_or("reflect") {
                "reflect" => WALL_REFLECT,
                "absorb" => WALL_ABSORB,
                "none" => 0,
                k => bail!("unknown wall type {k}"),
            };
            for c in region(p, get_xyz(v, "center"), get_f(v, "radius", 2.0)) {
                clear_cell(st, p, c);
                st.site[c * SITE + S_WALL] = kind;
            }
        }
        "mirror_region" => mirror_region(st, p, get_xyz(v, "min"), get_xyz(v, "max"))?,
        "set" | "snapshot" | "branch" => {} // handled by the Sim / CLI
        k => bail!("unknown event kind {k}"),
    }
    Ok(())
}

fn point(p: &Params, c: [i64; 3]) -> usize {
    let g = &p.geom;
    let mut q = [0u32; 3];
    for a in 0..3 {
        q[a] = (c[a].rem_euclid(g.size[a] as i64)) as u32;
    }
    g.index(q)
}

/// Apply M to an axis-aligned box: reflect x within [min.x, max.x], mirror
/// tapes, orientations, bonds and swap hands of fields.
pub fn mirror_region(st: &mut State, p: &Params, lo: [i64; 3], hi: [i64; 3]) -> Result<()> {
    let g = &p.geom;
    anyhow::ensure!(g.dims >= 1, "mirror_region needs space");
    let mut lo2 = [0u32; 3];
    let mut hi2 = [0u32; 3];
    for a in 0..3 {
        let s = g.size[a] as i64;
        lo2[a] = lo[a].clamp(0, s - 1) as u32;
        hi2[a] = if (a as u32) < g.dims { hi[a].clamp(0, s - 1) as u32 } else { 0 };
        if (a as u32) >= g.dims {
            lo2[a] = 0;
        }
    }
    let inside = |c: usize| {
        let q = g.coords(c);
        (0..3).all(|a| q[a] >= lo2[a] && q[a] <= hi2[a])
    };
    let cells: Vec<usize> = (0..st.n).filter(|&c| inside(c)).collect();
    // cut bonds crossing the region boundary
    for &c in &cells {
        let b = st.bonds(c);
        for d in 0..6u32 {
            if b & (1 << d) != 0 {
                let t = p.nbr[c * 6 + d as usize];
                if t != NONE && !inside(t as usize) {
                    st.hdr[c * HDR + H_FLAGS] &= !(1 << (BOND_SHIFT + d));
                    st.hdr[t as usize * HDR + H_FLAGS] &= !(1 << (BOND_SHIFT + (d ^ 1)));
                }
            }
        }
    }
    let before = st.matter_totals(p);
    let src = st.clone();
    let l = st.lmax;
    let nf = st.nf;
    for &c in &cells {
        let mut q = g.coords(c);
        q[0] = lo2[0] + hi2[0] - q[0];
        let s = g.index(q);
        for i in 0..l {
            st.tapes[c * l + i] = isa::mirror(src.tapes[s * l + i]);
        }
        st.hdr[c * HDR..(c + 1) * HDR].copy_from_slice(&src.hdr[s * HDR..(s + 1) * HDR]);
        for i in 0..nf {
            st.fields[c * nf + crate::phases::swap_hand_field(p, i)] = src.fields[s * nf + i];
        }
        let fl = st.hdr[c * HDR + H_FLAGS];
        if fl & FL_OCC == 0 {
            continue;
        }
        let or = (fl & ORIENT_MASK) >> ORIENT_SHIFT;
        let bonds = (fl & BOND_MASK) >> BOND_SHIFT;
        let mut nb = 0;
        for d in 0..6 {
            if bonds & (1 << d) != 0 {
                nb |= 1 << crate::geom::mirror_dir(d);
            }
        }
        st.hdr[c * HDR + H_FLAGS] = (fl & !(ORIENT_MASK | BOND_MASK)) | (p.orient.mirror[or as usize] << ORIENT_SHIFT) | (nb << BOND_SHIFT);
    }
    // a region mirror moves matter between the L and D classes: log the change
    let after = st.matter_totals(p);
    for cl in 0..3 {
        st.tally.matter_source[cl] += after[cl] - before[cl];
    }
    Ok(())
}

pub fn read_events(path: &std::path::Path) -> Result<Vec<Event>> {
    if !path.exists() {
        return Ok(vec![]);
    }
    let s = std::fs::read_to_string(path)?;
    s.lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).map_err(|e| e.into())).collect()
}
