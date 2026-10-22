//! CPU colouring of the lattice for the viewer.

use crate::{Frame, Image};
use primordia_core::isa;
use primordia_core::params::*;
use primordia_core::state::State;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
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
    pub fn all() -> [Mode; 12] {
        use Mode::*;
        [Group, Hand, Species, Lineage, Energy, Free, Temp, Chem, ChemEe, Pools, Signals, Exposure]
    }
    pub fn name(&self) -> &'static str {
        match self {
            Mode::Group => "dominant opcode group",
            Mode::Hand => "code hand (red L / blue D)",
            Mode::Species => "species (hash to hue)",
            Mode::Lineage => "lineage depth",
            Mode::Energy => "energy E",
            Mode::Free => "free energy F",
            Mode::Temp => "temperature",
            Mode::Chem => "chemical count",
            Mode::ChemEe => "chemical EE",
            Mode::Pools => "monomer pools (L r, A g, D b)",
            Mode::Signals => "signals (L r, D b)",
            Mode::Exposure => "exposure",
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
    pub params: Params,
}

impl ViewSettings {
    pub fn default(p: &Params) -> ViewSettings {
        let s = p.geom.size;
        ViewSettings { mode: Mode::Group, chem_k: 0, chem_h: 0, sig_j: 0, slice: [s[0] / 2, s[1] / 2, s[2] / 2], params: p.clone() }
    }
    /// UI-side copy: only geometry is needed for coordinate mapping.
    pub fn from_out(dims: u32, size: [u32; 3]) -> ViewSettings {
        let mut c = primordia_core::Config::preset("paper-0d").unwrap();
        c.world.dims = dims;
        c.world.size = if dims == 0 { vec![size[0]] } else { size[..dims as usize].to_vec() };
        let p = Params::new(&c);
        ViewSettings { mode: Mode::Group, chem_k: 0, chem_h: 0, sig_j: 0, slice: [size[0] / 2, size[1] / 2, size[2] / 2], params: p }
    }
}

const GROUP_COL: [[u8; 3]; 8] = [[40, 40, 40], [240, 200, 60], [80, 200, 120], [220, 80, 200], [90, 160, 240], [250, 120, 60], [140, 140, 140], [200, 200, 230]];

fn hue(h: f32, s: f32, v: f32) -> [u8; 3] {
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

fn ramp(x: f32) -> [u8; 3] {
    // perceptual-ish dark blue -> teal -> yellow
    let x = x.clamp(0.0, 1.0);
    let r = (255.0 * (1.5 * x - 0.3).clamp(0.0, 1.0)) as u8;
    let g = (255.0 * (1.2 * x).clamp(0.0, 1.0).powf(0.8)) as u8;
    let b = (255.0 * (0.5 + 0.8 * x - 1.2 * x * x).clamp(0.0, 1.0)) as u8;
    [r, g, b]
}

fn genome_hash(t: &[u8]) -> u32 {
    let mut m = u32::MAX;
    for w in t.windows(4) {
        let g = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
        let h = g.wrapping_mul(0x9E3779B1).rotate_left(13).wrapping_mul(0x85EBCA77);
        m = m.min(h);
    }
    m
}

fn scalar(st: &State, p: &Params, v: &ViewSettings, c: usize, expo: &[u32]) -> f32 {
    match v.mode {
        Mode::Energy => (st.h(c, H_E) as f32 + 1.0).ln(),
        Mode::Free => (st.f(c, F_F) as f32 + 1.0).ln(),
        Mode::Temp => st.f(c, F_T) as f32 / 100.0,
        Mode::Chem if v.chem_k < p.k => st.f(c, p.f_chem(v.chem_k, v.chem_h)) as f32,
        Mode::Exposure => expo.get(c).map(|&e| e as f32 / 65536.0).unwrap_or(1.0),
        _ => 0.0,
    }
}

fn color(st: &State, p: &Params, v: &ViewSettings, c: usize, lo: f32, hi: f32, expo: &[u32], mx: [f32; 3]) -> [u8; 3] {
    if st.site[c * SITE + S_WALL] != 0 {
        return if st.site[c * SITE + S_WALL] == WALL_ABSORB { [10, 10, 60] } else { [90, 70, 50] };
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
                    let s = ee.abs();
                    let base = 70.0;
                    if ee >= 0.0 {
                        dim([(base + (255.0 - base) * s) as u8, (base * (1.0 - s)) as u8, (base * (1.0 - s)) as u8])
                    } else {
                        dim([(base * (1.0 - s)) as u8, (base * (1.0 - s) + 60.0 * s) as u8, (base + (255.0 - base) * s) as u8])
                    }
                }
                Mode::Species => dim(hue(genome_hash(t) as f32 / u32::MAX as f32, 0.75, 0.95)),
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
            if ee >= 0.0 { [(40.0 + 215.0 * s) as u8, 40, 40] } else { [40, 40, (40.0 + 215.0 * s) as u8] }
        }
        Mode::Pools => {
            if !p.matter_on {
                return [0, 0, 0];
            }
            let f = |cl: usize| ((st.f(c, p.f_pool(cl)) as f32 / mx[cl].max(1.0)).sqrt() * 255.0).min(255.0) as u8;
            [f(0), f(2), f(1)]
        }
        Mode::Signals => {
            if v.sig_j >= p.s {
                return [0, 0, 0];
            }
            let f = |h: usize| ((st.f(c, p.f_sig(v.sig_j, h)) as f32 / mx[h].max(1.0)).sqrt() * 255.0).min(255.0) as u8;
            [f(0), 20, f(1)]
        }
        _ => {
            let x = scalar(st, p, v, c, expo);
            let n = if hi > lo { (x - lo) / (hi - lo) } else { 0.5 };
            let col = ramp(n);
            if occ || v.mode != Mode::Energy { col } else { [8, 8, 8] }
        }
    }
}

pub fn render(st: &State, p: &Params, v: &ViewSettings, domains: &[i8], hist1d: &mut Vec<Vec<u8>>) -> Frame {
    let g = &p.geom;
    let expo = if v.mode == Mode::Exposure { primordia_core::phases::exposure(st, p) } else { vec![] };
    // normalisation over all cells
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    let mut mx = [0f32; 3];
    for c in 0..st.n {
        let x = scalar(st, p, v, c, &expo);
        if v.mode != Mode::Energy || st.occupied(c) {
            lo = lo.min(x);
            hi = hi.max(x);
        }
        if v.mode == Mode::Pools && p.matter_on {
            for cl in 0..3 {
                mx[cl] = mx[cl].max(st.f(c, p.f_pool(cl)) as f32);
            }
        }
        if v.mode == Mode::Signals && v.sig_j < p.s {
            for h in 0..2 {
                mx[h] = mx[h].max(st.f(c, p.f_sig(v.sig_j, h)) as f32);
            }
        }
    }
    let px = |c: usize| color(st, p, v, c, lo, hi, &expo, mx);
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
    let mut bonds = vec![];
    match g.dims {
        0 => {
            let w = (g.n as f64).sqrt().ceil() as usize;
            let h = g.n.div_ceil(w);
            images.push(mk(format!("0D soup: {} slots (epoch {})", g.n, st.epoch), w, h, (0, 1), [0; 3], &|x, y| {
                let c = x + y * w;
                if c < g.n { px(c) } else { [0, 0, 0] }
            }));
        }
        1 => {
            let row: Vec<u8> = (0..g.n).flat_map(|c| px(c)).collect();
            hist1d.push(row);
            if hist1d.len() > 512 {
                hist1d.remove(0);
            }
            let w = g.n;
            let h = hist1d.len();
            images.push(mk(format!("1D space-time (newest at top), epoch {}", st.epoch), w, h, (0, 1), [0; 3], &|x, y| {
                let r = &hist1d[y];
                [r[3 * x], r[3 * x + 1], r[3 * x + 2]]
            }));
        }
        2 => {
            let (w, h) = (g.size[0] as usize, g.size[1] as usize);
            images.push(mk(format!("{} — epoch {}", v.mode.name(), st.epoch), w, h, (0, 1), [0; 3], &|x, y| px(x + w * y)));
            bonds = (0..st.n).map(|c| st.bonds(c) as u8).collect();
        }
        _ => {
            let s = g.size;
            let sl = [v.slice[0].min(s[0] - 1), v.slice[1].min(s[1] - 1), v.slice[2].min(s[2] - 1)];
            images.push(mk(format!("XY at z={} — {}", sl[2], v.mode.name()), s[0] as usize, s[1] as usize, (0, 1), [0, 0, sl[2]], &|x, y| px(g.index([x as u32, y as u32, sl[2]]))));
            images.push(mk(format!("XZ at y={}", sl[1]), s[0] as usize, s[2] as usize, (0, 2), [0, sl[1], 0], &|x, z| px(g.index([x as u32, sl[1], z as u32]))));
            images.push(mk(format!("YZ at x={}", sl[0]), s[1] as usize, s[2] as usize, (1, 2), [sl[0], 0, 0], &|y, z| px(g.index([sl[0], y as u32, z as u32]))));
        }
    }
    let vents = st.vents.iter().map(|vv| g.coords(vv.pos as usize)).collect();
    let dnb = [16.min(g.size[0] as usize), if g.dims >= 2 { 16.min(g.size[1] as usize) } else { 1 }, if g.dims >= 3 { 16.min(g.size[2] as usize) } else { 1 }];
    Frame { epoch: st.epoch, images, bonds, vents, domains: domains.to_vec(), dnb, size: g.size, dims: g.dims }
}
