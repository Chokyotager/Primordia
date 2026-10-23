//! Cluster persistence: follows bonded bodies and kin clusters (patches of
//! adjacent cells whose programs share >= 12 of their first 16 bytes) from one
//! sample to the next through their surviving member cells, and counts how
//! many sunrises each cluster has lived through.

use primordia_core::params::*;
use primordia_core::{Params, State, phases};
use std::collections::HashMap;

/// Clusters smaller than this are not tracked.
pub const MIN_CLUSTER: u32 = 50;
/// A cluster continues an earlier one when at least this share of the
/// earlier cluster's cells are still alive inside it.
const CARRY: f64 = 0.25;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Body,
    Kin,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Cluster {
    pub id: u64,
    pub kind: Kind,
    pub size: u32,
    pub first_seen: u64,
    pub nights: u32,
    /// Cell coordinates of a member near the cluster's (torus-aware) centre.
    pub center: [u32; 3],
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct KindStats {
    pub clusters: usize,
    pub largest: u32,
    /// Clusters that continue a cluster from the previous sample.
    pub linked: usize,
    pub nights_1: usize,
    pub nights_2: usize,
    pub nights_3: usize,
    pub max_nights: u32,
    pub oldest_age: u64,
    /// Share of living cells inside clusters that have survived >= 1 night.
    pub cells_surviving: f64,
}

struct Prev {
    epoch: u64,
    label: Vec<u32>,
    tracks: HashMap<u32, Cluster>,
}

#[derive(Default)]
pub struct Tracker {
    prev: HashMap<Kind, Prev>,
    birth: Vec<u32>,
    lineage: Vec<u32>,
    next_id: u64,
}

fn find(par: &mut [u32], mut x: u32) -> u32 {
    while par[x as usize] != x {
        par[x as usize] = par[par[x as usize] as usize];
        x = par[x as usize];
    }
    x
}

/// Kin-cluster label per cell (union-find root), NONE for empty cells.
fn kin_labels(st: &State, p: &Params) -> Vec<u32> {
    let n = st.n;
    let l = st.lmax;
    let w = 16.min(l);
    let mut par: Vec<u32> = (0..n as u32).collect();
    let faces = p.geom.faces() as usize;
    for c in 0..n {
        if !st.occupied(c) || st.tapes[c * l] == 0 {
            continue;
        }
        let a = &st.tapes[c * l..c * l + w];
        for d in 0..faces {
            let t = p.nbr[c * 6 + d];
            if t == NONE || !st.occupied(t as usize) {
                continue;
            }
            let t = t as usize;
            let b = &st.tapes[t * l..t * l + w];
            if a.iter().zip(b).filter(|(x, y)| x == y).count() >= 12 {
                let (ra, rb) = (find(&mut par, c as u32), find(&mut par, t as u32));
                if ra != rb {
                    par[ra.max(rb) as usize] = ra.min(rb);
                }
            }
        }
    }
    (0..n).map(|c| if st.occupied(c) { find(&mut par, c as u32) } else { NONE }).collect()
}

/// Keep only labels of clusters with >= MIN_CLUSTER cells.
fn big(label: Vec<u32>) -> (Vec<u32>, HashMap<u32, u32>) {
    let mut size: HashMap<u32, u32> = HashMap::new();
    for &l in &label {
        if l != NONE {
            *size.entry(l).or_default() += 1;
        }
    }
    size.retain(|_, s| *s >= MIN_CLUSTER);
    let label = label.into_iter().map(|l| if size.contains_key(&l) { l } else { NONE }).collect();
    (label, size)
}

/// A member cell near the circular mean of the cluster's coordinates.
fn centers(p: &Params, label: &[u32]) -> HashMap<u32, [u32; 3]> {
    let g = &p.geom;
    let mut acc: HashMap<u32, [f64; 6]> = HashMap::new();
    for (c, &l) in label.iter().enumerate() {
        if l == NONE {
            continue;
        }
        let q = g.coords(c);
        let a = acc.entry(l).or_default();
        for k in 0..3 {
            let th = std::f64::consts::TAU * q[k] as f64 / g.size[k].max(1) as f64;
            a[2 * k] += th.cos();
            a[2 * k + 1] += th.sin();
        }
    }
    let mean: HashMap<u32, [f64; 3]> = acc
        .into_iter()
        .map(|(l, a)| {
            let mut m = [0.0; 3];
            for k in 0..3 {
                m[k] = (a[2 * k + 1].atan2(a[2 * k]).rem_euclid(std::f64::consts::TAU)) / std::f64::consts::TAU * g.size[k].max(1) as f64;
            }
            (l, m)
        })
        .collect();
    let mut best: HashMap<u32, (f64, [u32; 3])> = HashMap::new();
    for (c, &l) in label.iter().enumerate() {
        if l == NONE {
            continue;
        }
        let q = g.coords(c);
        let m = mean[&l];
        let mut d2 = 0.0;
        for k in 0..3 {
            let s = g.size[k].max(1) as f64;
            let mut d = (q[k] as f64 - m[k]).abs();
            d = d.min(s - d);
            d2 += d * d;
        }
        let e = best.entry(l).or_insert((f64::MAX, q));
        if d2 < e.0 {
            *e = (d2, q);
        }
    }
    best.into_iter().map(|(l, (_, q))| (l, q)).collect()
}

impl Tracker {
    /// Sample the state: returns per-kind statistics and all tracked clusters.
    pub fn observe(&mut self, st: &State, p: &Params) -> (Vec<(Kind, KindStats)>, Vec<Cluster>) {
        let e = st.epoch;
        let day = p.day_period;
        let birth: Vec<u32> = (0..st.n).map(|c| st.h(c, H_BIRTH)).collect();
        let lineage: Vec<u32> = (0..st.n).map(|c| st.h(c, H_LINEAGE)).collect();
        let occ = (0..st.n).filter(|&c| st.occupied(c)).count().max(1);
        let mut stats = vec![];
        let mut all = vec![];
        for kind in [Kind::Body, Kind::Kin] {
            let raw = match kind {
                Kind::Body => phases::components(st, p).label,
                Kind::Kin => kin_labels(st, p),
            };
            let (label, size) = big(raw);
            let ctr = centers(p, &label);
            // links through surviving cells (same site, birth epoch and lineage)
            let mut best: HashMap<u32, (u32, u32)> = HashMap::new();
            let mut crossed = false;
            if let Some(pv) = self.prev.get(&kind) {
                crossed = day > 0 && e / day > pv.epoch / day;
                let mut pairs: HashMap<(u32, u32), u32> = HashMap::new();
                for c in 0..st.n {
                    let (a, b) = (pv.label[c], label[c]);
                    if a != NONE && b != NONE && self.birth[c] == birth[c] && self.lineage[c] == lineage[c] {
                        *pairs.entry((a, b)).or_default() += 1;
                    }
                }
                for ((a, b), cnt) in pairs {
                    let sa = pv.tracks.get(&a).map(|t| t.size).unwrap_or(u32::MAX);
                    if cnt as f64 >= CARRY * sa as f64 && cnt > best.get(&b).map(|x| x.0).unwrap_or(0) {
                        best.insert(b, (cnt, a));
                    }
                }
            }
            let mut tracks = HashMap::new();
            let mut ks = KindStats { clusters: size.len(), ..Default::default() };
            let mut surviving_cells = 0u64;
            for (&l, &sz) in &size {
                let prev = best.get(&l).and_then(|(_, a)| self.prev.get(&kind).and_then(|pv| pv.tracks.get(a)));
                let t = match prev {
                    Some(t0) => {
                        ks.linked += 1;
                        Cluster { id: t0.id, kind, size: sz, first_seen: t0.first_seen, nights: t0.nights + crossed as u32, center: ctr[&l] }
                    }
                    None => {
                        self.next_id += 1;
                        Cluster { id: self.next_id, kind, size: sz, first_seen: e, nights: 0, center: ctr[&l] }
                    }
                };
                ks.largest = ks.largest.max(sz);
                ks.nights_1 += (t.nights >= 1) as usize;
                ks.nights_2 += (t.nights >= 2) as usize;
                ks.nights_3 += (t.nights >= 3) as usize;
                ks.max_nights = ks.max_nights.max(t.nights);
                ks.oldest_age = ks.oldest_age.max(e - t.first_seen);
                if t.nights >= 1 {
                    surviving_cells += sz as u64;
                }
                all.push(t.clone());
                tracks.insert(l, t);
            }
            ks.cells_surviving = surviving_cells as f64 / occ as f64;
            stats.push((kind, ks));
            self.prev.insert(kind, Prev { epoch: e, label, tracks });
        }
        self.birth = birth;
        self.lineage = lineage;
        all.sort_by(|a, b| (b.nights, b.size).cmp(&(a.nights, a.size)));
        (stats, all)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use primordia_core::Config;

    /// A 10 x 6 block of identical cells is one kin cluster; it keeps its
    /// identity and gains a night across a sunrise, and becomes a new cluster
    /// when fewer than 25% of its cells survive.
    #[test]
    fn kin_cluster_survives_a_night() {
        let mut cfg = Config::preset("primordia-2d-sun").unwrap();
        cfg.world.size = vec![30, 30];
        cfg.validate().unwrap();
        let p = Params::new(&cfg);
        let mut st = State::init(&cfg, &p);
        for c in 0..st.n {
            st.hdr[c * HDR + H_FLAGS] = 0;
        }
        let prog: Vec<u8> = (0..16).map(|i| 0x40 + i as u8).collect();
        let block: Vec<usize> = (0..6).flat_map(|y| (0..10).map(move |x| x + 30 * (y + 5))).collect();
        for &c in &block {
            st.hdr[c * HDR + H_FLAGS] = FL_OCC;
            st.hdr[c * HDR + H_LEN] = 16;
            st.hdr[c * HDR + H_BIRTH] = 100;
            st.hdr[c * HDR + H_LINEAGE] = 7;
            st.tapes[c * st.lmax..c * st.lmax + 16].copy_from_slice(&prog);
        }
        let kin = |s: &[(Kind, KindStats)]| s.iter().find(|x| x.0 == Kind::Kin).unwrap().1.clone();
        let mut tr = Tracker::default();
        st.epoch = 1500;
        let (s0, c0) = tr.observe(&st, &p);
        assert_eq!(kin(&s0).clusters, 1);
        assert_eq!(kin(&s0).largest, 60);
        let id = c0.iter().find(|c| c.kind == Kind::Kin).unwrap().id;
        st.epoch = 2250; // past the sunrise at 2000
        let (s1, c1) = tr.observe(&st, &p);
        let k1 = c1.iter().find(|c| c.kind == Kind::Kin).unwrap();
        assert_eq!((k1.id, k1.nights, kin(&s1).linked), (id, 1, 1));
        assert!((kin(&s1).cells_surviving - 1.0).abs() < 1e-9);
        // 50 of the 60 cells are replaced by newborns: the 10 survivors are < 25%
        for &c in &block[10..] {
            st.hdr[c * HDR + H_BIRTH] = 2300;
        }
        st.epoch = 2500;
        let (s2, c2) = tr.observe(&st, &p);
        let k2 = c2.iter().find(|c| c.kind == Kind::Kin).unwrap();
        assert_ne!(k2.id, id);
        assert_eq!((k2.nights, kin(&s2).linked), (0, 0));
    }
}
