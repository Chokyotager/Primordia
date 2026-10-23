//! CPU colouring of the lattice. `Mode::Life` is an illustrated view meant to
//! be readable at a glance; the other modes show one quantity each.

use crate::{CellInfo, Frame, Image};
use primordia_core::isa;
use primordia_core::params::*;
use primordia_core::state::State;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Life,
    Group,
    Hand,
    Species,
    Lineage,
    Energy,
    Free,
    Temp,
    Chem,
    ChemEe,
    Pools,
    Signals,
    Exposure,
}

impl Mode {
    pub fn all() -> [Mode; 13] {
        use Mode::*;
        [Life, Group, Hand, Species, Lineage, Energy, Free, Temp, Chem, ChemEe, Pools, Signals, Exposure]
    }
    pub fn name(&self) -> &'static str {
        match self {
            Mode::Life => "Life (illustrated)",
            Mode::Group => "What programs do (opcode group)",
            Mode::Hand => "Handedness of code (L red / D blue)",
            Mode::Species => "Families (genome similarity)",
            Mode::Lineage => "Generations (lineage depth)",
            Mode::Energy => "Stored energy E",
            Mode::Free => "Food: free energy F",
            Mode::Temp => "Temperature",
            Mode::Chem => "Chemical amount",
            Mode::ChemEe => "Chemical handedness",
            Mode::Pools => "Building blocks (monomers)",
            Mode::Signals => "Signals",
            Mode::Exposure => "Exposure (how surrounded)",
        }
    }
    /// (swatch colour, meaning) pairs for the legend.
    pub fn legend(&self) -> Vec<([u8; 3], &'static str)> {
        match self {
            Mode::Life => vec![
                (PALETTE[0], "a living cell; the colour is its family (similar programs share a colour)"),
                (PALETTE[5], "bright = well fed, dim = low on energy"),
                ([150, 150, 150], "grey = dormant (out of energy, waiting for light)"),
                ([255, 255, 255], "white rim = copier: its program contains a copy loop"),
                ([24, 60, 52], "dark green glow = food (free energy) in the environment"),
                ([90, 40, 20], "warm tint = hot area"),
                ([255, 170, 40], "orange beacon = hydrothermal vent (energy, heat, X4)"),
                ([110, 90, 70], "brown = wall"),
            ],
            Mode::Group => vec![
                (GROUP_COL[1], "stack & control"),
                (GROUP_COL[2], "copying / heads (CPY, RD, WR)"),
                (GROUP_COL[3], "chiral ops (turn, bond, catalyse)"),
                (GROUP_COL[4], "numbers (PUSH)"),
                (GROUP_COL[5], "body & life cycle (MOVE, ALLOC, DIVIDE)"),
                (GROUP_COL[7], "jumps"),
                (GROUP_COL[0], "mostly empty"),
            ],
            Mode::Hand => vec![([230, 60, 60], "mostly L-handed code"), ([60, 90, 235], "mostly D-handed code"), ([70, 70, 70], "balanced")],
            Mode::Energy | Mode::Free | Mode::Temp | Mode::Chem | Mode::Exposure => vec![(ramp(0.0), "low"), (ramp(0.5), "medium"), (ramp(1.0), "high")],
            Mode::ChemEe => vec![([230, 60, 60], "excess of L molecules"), ([60, 60, 230], "excess of D molecules")],
            Mode::Pools => vec![([255, 0, 0], "L monomers"), ([0, 255, 0], "A monomers"), ([0, 0, 255], "D monomers")],
            Mode::Signals => vec![([255, 20, 0], "L signal"), ([0, 20, 255], "D signal")],
            Mode::Species => vec![(PALETTE[2], "each colour is a family of similar genomes")],
            Mode::Lineage => vec![(hue(0.0, 0.8, 0.9), "early generations"), (hue(0.5, 0.8, 0.9), "later generations")],
        }
    }
}

#[derive(Clone)]
pub struct ViewSettings {
    pub mode: Mode,
    pub chem_k: usize,
    pub chem_h: usize,
    pub sig_j: usize,
    pub slice: [u32; 3],
    pub params: std::sync::Arc<Params>,
}

impl ViewSettings {
    pub fn default(p: &Params) -> ViewSettings {
        let s = p.geom.size;
        ViewSettings { mode: Mode::Life, chem_k: 0, chem_h: 0, sig_j: 0, slice: [s[0] / 2, s[1] / 2, s[2] / 2], params: std::sync::Arc::new(p.clone()) }
    }
    /// UI-side copy: only geometry is needed for coordinate mapping.
    pub fn from_out(dims: u32, size: [u32; 3]) -> ViewSettings {
        let mut c = primordia_core::Config::preset("paper-0d").unwrap();
        c.world.dims = dims;
        c.world.size = if dims == 0 { vec![size[0]] } else { size[..dims as usize].to_vec() };
        let p = Params::new(&c);
        ViewSettings { mode: Mode::Life, chem_k: 0, chem_h: 0, sig_j: 0, slice: [size[0] / 2, size[1] / 2, size[2] / 2], params: std::sync::Arc::new(p) }
    }
}

/// Curated, distinguishable family colours.
pub const PALETTE: [[u8; 3]; 16] = [
    [94, 201, 98],
    [79, 172, 230],
    [240, 101, 160],
    [250, 196, 72],
    [167, 120, 240],
    [64, 214, 190],
    [242, 128, 76],
    [196, 226, 92],
    [118, 140, 255],
    [255, 140, 200],
    [140, 220, 140],
    [230, 90, 90],
    [90, 200, 240],
    [220, 170, 255],
    [255, 210, 140],
    [120, 230, 210],
];

pub const GROUP_COL: [[u8; 3]; 8] = [[40, 40, 40], [240, 200, 60], [80, 200, 120], [220, 80, 200], [90, 160, 240], [250, 120, 60], [140, 140, 140], [200, 200, 230]];

pub fn hue(h: f32, s: f32, v: f32) -> [u8; 3] {
    let h = (h.fract() + 1.0).fract() * 6.0;
    let i = h.floor() as i32;
    let f = h - i as f32;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f)));
    let (r, g, b) = match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

pub fn ramp(x: f32) -> [u8; 3] {
    // dark blue -> teal -> yellow
    let x = x.clamp(0.0, 1.0);
    let r = (255.0 * (1.5 * x - 0.3).clamp(0.0, 1.0)) as u8;
    let g = (255.0 * (1.2 * x).clamp(0.0, 1.0).powf(0.8)) as u8;
    let b = (255.0 * (0.5 + 0.8 * x - 1.2 * x * x).clamp(0.0, 1.0)) as u8;
    [r, g, b]
}

/// Family key: programs run from their first byte and copies are written
/// from the start of the partner's tape, so relatives share their opening
/// bytes even when their tails differ.
pub fn genome_hash(t: &[u8]) -> u32 {
    let mut h = 0x811c_9dc5u32;
    for &b in t.iter().take(12) {
        h = (h ^ b as u32).wrapping_mul(0x0100_0193);
    }
    h ^ (h >> 15)
}

/// Heuristic: the program contains a copy loop (CPY, a head increment and a backward jump).
pub fn is_copier(t: &[u8]) -> bool {
    let cpy = t.iter().any(|&b| b == isa::op::CPY);
    let inc = t.iter().any(|&b| b == isa::op::H1INC || b == isa::op::H0INC);
    let back = t.iter().any(|&b| b >= 0xC0);
    cpy && inc && back
}

fn mix(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    [0, 1, 2].map(|i| (a[i] as f32 * (1.0 - t) + b[i] as f32 * t) as u8)
}

fn scalar(st: &State, p: &Params, v: &ViewSettings, c: usize, expo: &[u32]) -> f32 {
    match v.mode {
        Mode::Energy => (st.h(c, H_E) as f32 + 1.0).ln(),
        Mode::Free | Mode::Life => (st.f(c, F_F) as f32 + 1.0).ln(),
        Mode::Temp => st.f(c, F_T) as f32 / 100.0,
        Mode::Chem if v.chem_k < p.k => st.f(c, p.f_chem(v.chem_k, v.chem_h)) as f32,
        Mode::Exposure => expo.get(c).map(|&e| e as f32 / 65536.0).unwrap_or(1.0),
        _ => 0.0,
    }
}

struct Norm {
    lo: f32,
    hi: f32,
    mx: [f32; 3],
    e_hi: f32,
    t_amb: f32,
}

fn life_color(st: &State, c: usize, nm: &Norm, info: &CellInfo) -> [u8; 3] {
    if st.site[c * SITE + S_WALL] != 0 {
        return [110, 90, 70];
    }
    // environment
    let food = if nm.hi > nm.lo { ((scalar_free(st, c) - nm.lo) / (nm.hi - nm.lo)).clamp(0.0, 1.0) } else { 0.0 };
    let mut bg = mix([10, 13, 22], [24, 70, 58], food * 0.9);
    let heat = ((st.f(c, F_T) as f32 / 100.0 - nm.t_amb) / 15.0).clamp(0.0, 1.0);
    bg = mix(bg, [110, 45, 18], heat * 0.7);
    if !st.occupied(c) {
        return bg;
    }
    let base = PALETTE[(info.family % PALETTE.len() as u32) as usize];
    if info.halted {
        return mix([70, 72, 78], base, 0.25);
    }
    let e = ((st.h(c, H_E) as f32 + 1.0).ln() / nm.e_hi.max(1.0)).clamp(0.0, 1.0);
    let col = mix([30, 30, 34], base, 0.45 + 0.55 * e);
    // copiers in full colour, inert programs muted
    if info.copier { col } else { mix(col, [60, 62, 68], 0.45) }
}

fn scalar_free(st: &State, c: usize) -> f32 {
    (st.f(c, F_F) as f32 + 1.0).ln()
}

fn color(st: &State, p: &Params, v: &ViewSettings, c: usize, nm: &Norm, expo: &[u32], info: &CellInfo) -> [u8; 3] {
    if v.mode == Mode::Life {
        return life_color(st, c, nm, info);
    }
    if st.site[c * SITE + S_WALL] != 0 {
        return if st.site[c * SITE + S_WALL] == WALL_ABSORB { [10, 10, 60] } else { [110, 90, 70] };
    }
    let occ = st.occupied(c);
    let dim = |col: [u8; 3]| if st.flags(c) & FL_HALT != 0 { [col[0] / 2, col[1] / 2, col[2] / 2] } else { col };
    match v.mode {
        Mode::Group | Mode::Hand | Mode::Species | Mode::Lineage => {
            if !occ {
                return [12, 12, 16];
            }
            let t = st.tape(c);
            match v.mode {
                Mode::Group => {
                    let mut cnt = [0u32; 8];
                    for &b in t {
                        cnt[isa::group(b) as usize] += 1;
                    }
                    cnt[0] /= 4;
                    let g = (0..8).max_by_key(|&i| cnt[i]).unwrap();
                    dim(GROUP_COL[g])
                }
                Mode::Hand => {
                    let (mut l, mut d) = (0, 0);
                    for &b in t {
                        if isa::is_chiral(b) {
                            if b & 1 == 0 { l += 1 } else { d += 1 }
                        }
                    }
                    let ee = if l + d == 0 { 0.0 } else { (l - d) as f32 / (l + d) as f32 };
                    if ee >= 0.0 { dim(mix([70, 70, 70], [230, 60, 60], ee)) } else { dim(mix([70, 70, 70], [60, 90, 235], -ee)) }
                }
                Mode::Species => dim(PALETTE[(info.family % 16) as usize]),
                _ => dim(hue(st.h(c, H_DEPTH) as f32 / 48.0, 0.8, 0.9)),
            }
        }
        Mode::ChemEe => {
            if v.chem_k >= p.k {
                return [0, 0, 0];
            }
            let l = st.f(c, p.f_chem(v.chem_k, 0)) as f32;
            let d = st.f(c, p.f_chem(v.chem_k, 1)) as f32;
            let ee = if l + d == 0.0 { 0.0 } else { (l - d) / (l + d) };
            let s = (ee.abs() * 4.0).min(1.0);
            if ee >= 0.0 { [(40.0 + 200.0 * s) as u8, 40, 40] } else { [40, 40, (40.0 + 200.0 * s) as u8] }
        }
        Mode::Pools => {
            if !p.matter_on {
                return [0, 0, 0];
            }
            let f = |cl: usize| ((st.f(c, p.f_pool(cl)) as f32 / nm.mx[cl].max(1.0)).sqrt() * 255.0).min(255.0) as u8;
            [f(0), f(2), f(1)]
        }
        Mode::Signals => {
            if v.sig_j >= p.s {
                return [0, 0, 0];
            }
            let f = |h: usize| ((st.f(c, p.f_sig(v.sig_j, h)) as f32 / nm.mx[h].max(1.0)).sqrt() * 255.0).min(255.0) as u8;
            [f(0), 20, f(1)]
        }
        _ => {
            let x = scalar(st, p, v, c, expo);
            let n = if nm.hi > nm.lo { (x - nm.lo) / (nm.hi - nm.lo) } else { 0.5 };
            let col = ramp(n);
            if occ || v.mode != Mode::Energy { col } else { [8, 8, 8] }
        }
    }
}

pub fn cell_infos(st: &State) -> Vec<CellInfo> {
    use rayon::prelude::*;
    (0..st.n)
        .into_par_iter()
        .map(|c| {
            if !st.occupied(c) {
                return CellInfo::default();
            }
            let t = st.tape(c);
            CellInfo { occ: true, halted: st.flags(c) & FL_HALT != 0, family: genome_hash(t), copier: is_copier(t), bonded: st.bonds(c) != 0 }
        })
        .collect()
}

pub fn render(st: &State, p: &Params, v: &ViewSettings, domains: &[i8], hist1d: &mut Vec<Vec<u8>>) -> Frame {
    let g = &p.geom;
    let expo = if v.mode == Mode::Exposure { primordia_core::phases::exposure(st, p) } else { vec![] };
    let infos = cell_infos(st);
    let mut nm = Norm { lo: f32::MAX, hi: f32::MIN, mx: [0.0; 3], e_hi: 1.0, t_amb: p.t_amb as f32 / 100.0 };
    for c in 0..st.n {
        let x = scalar(st, p, v, c, &expo);
        if v.mode != Mode::Energy || st.occupied(c) {
            nm.lo = nm.lo.min(x);
            nm.hi = nm.hi.max(x);
        }
        if st.occupied(c) {
            nm.e_hi = nm.e_hi.max((st.h(c, H_E) as f32 + 1.0).ln());
        }
        if v.mode == Mode::Pools && p.matter_on {
            for cl in 0..3 {
                nm.mx[cl] = nm.mx[cl].max(st.f(c, p.f_pool(cl)) as f32);
            }
        }
        if v.mode == Mode::Signals && v.sig_j < p.s {
            for h in 0..2 {
                nm.mx[h] = nm.mx[h].max(st.f(c, p.f_sig(v.sig_j, h)) as f32);
            }
        }
    }
    let px = |c: usize| color(st, p, v, c, &nm, &expo, &infos[c]);
    let mut images = vec![];
    let mk = |label: String, w: usize, h: usize, axes: (usize, usize), fixed: [u32; 3], f: &dyn Fn(usize, usize) -> [u8; 3]| -> Image {
        let mut rgba = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let col = f(x, y);
                let i = (y * w + x) * 4;
                rgba[i..i + 3].copy_from_slice(&col);
                rgba[i + 3] = 255;
            }
        }
        Image { label, w, h, rgba, axes, fixed }
    };
    match g.dims {
        0 => {
            let w = (g.n as f64).sqrt().ceil() as usize;
            let h = g.n.div_ceil(w);
            images.push(mk(format!("Well-mixed soup: {} programs", g.n), w, h, (0, 1), [0; 3], &|x, y| {
                let c = x + y * w;
                if c < g.n { px(c) } else { [0, 0, 0] }
            }));
        }
        1 => {
            let row: Vec<u8> = (0..g.n).flat_map(px).collect();
            hist1d.push(row);
            if hist1d.len() > 512 {
                hist1d.remove(0);
            }
            let w = g.n;
            let h = hist1d.len();
            images.push(mk("1D world over time (newest row at the top)".into(), w, h, (0, 1), [0; 3], &|x, y| {
                let r = &hist1d[y];
                [r[3 * x], r[3 * x + 1], r[3 * x + 2]]
            }));
        }
        2 => {
            let (w, h) = (g.size[0] as usize, g.size[1] as usize);
            images.push(mk(v.mode.name().to_string(), w, h, (0, 1), [0; 3], &|x, y| px(x + w * y)));
        }
        _ => {
            let s = g.size;
            let sl = [v.slice[0].min(s[0] - 1), v.slice[1].min(s[1] - 1), v.slice[2].min(s[2] - 1)];
            images.push(mk(format!("Seen from above: layer z = {}", sl[2]), s[0] as usize, s[1] as usize, (0, 1), [0, 0, sl[2]], &|x, y| px(g.index([x as u32, y as u32, sl[2]]))));
            images.push(mk(format!("Side view: slice y = {}", sl[1]), s[0] as usize, s[2] as usize, (0, 2), [0, sl[1], 0], &|x, z| px(g.index([x as u32, sl[1], z as u32]))));
            images.push(mk(format!("Side view: slice x = {}", sl[0]), s[1] as usize, s[2] as usize, (1, 2), [sl[0], 0, 0], &|y, z| px(g.index([sl[0], y as u32, z as u32]))));
        }
    }
    let vents = st.vents.iter().map(|vv| g.coords(vv.pos as usize)).collect();
    let dnb = [16.min(g.size[0] as usize), if g.dims >= 2 { 16.min(g.size[1] as usize) } else { 1 }, if g.dims >= 3 { 16.min(g.size[2] as usize) } else { 1 }];
    let bonds = if g.dims == 2 { (0..st.n).map(|c| st.bonds(c) as u8).collect() } else { vec![] };
    let cells = if g.dims == 2 {
        (0..st.n)
            .map(|c| crate::CellBrief {
                e: st.h(c, H_E),
                len: st.len(c) as u16,
                f: st.f(c, F_F),
                t: st.f(c, F_T),
                light: st.light[c],
                info: infos[c],
            })
            .collect()
    } else {
        vec![]
    };
    Frame { epoch: st.epoch, images, bonds, vents, domains: domains.to_vec(), dnb, size: g.size, dims: g.dims, cells }
}

/// Illustrated colours for every cell (used by the browser viewer), plus a
/// height byte per cell: empty = food level (0..100), occupied = 128 + energy level.
pub fn life_cells(st: &State, p: &Params, infos: &[CellInfo]) -> (Vec<[u8; 3]>, Vec<u8>) {
    use rayon::prelude::*;
    let mut nm = Norm { lo: f32::MAX, hi: f32::MIN, mx: [0.0; 3], e_hi: 1.0, t_amb: p.t_amb as f32 / 100.0 };
    for c in 0..st.n {
        let x = scalar_free(st, c);
        nm.lo = nm.lo.min(x);
        nm.hi = nm.hi.max(x);
        if st.occupied(c) {
            nm.e_hi = nm.e_hi.max((st.h(c, H_E) as f32 + 1.0).ln());
        }
    }
    let cols: Vec<[u8; 3]> = (0..st.n).into_par_iter().map(|c| life_color(st, c, &nm, &infos[c])).collect();
    let h: Vec<u8> = (0..st.n)
        .into_par_iter()
        .map(|c| {
            if st.occupied(c) {
                let e = ((st.h(c, H_E) as f32 + 1.0).ln() / nm.e_hi.max(1.0)).clamp(0.0, 1.0);
                128 + (e * 127.0) as u8
            } else {
                let food = if nm.hi > nm.lo { ((scalar_free(st, c) - nm.lo) / (nm.hi - nm.lo)).clamp(0.0, 1.0) } else { 0.0 };
                (food * 100.0) as u8
            }
        })
        .collect();
    (cols, h)
}
