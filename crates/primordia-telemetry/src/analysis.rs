//! Derived analyses: high-order entropy, replicator assay, species census
//! (MinHash/LSH), chirality domains.

use primordia_core::config::{Config, PairOrder};
use primordia_core::exec::{exec_slice, flag_snapshot};
use primordia_core::isa;
use primordia_core::params::*;
use primordia_core::state::State;
use primordia_core::Params;
use std::collections::HashMap;
use std::io::Write;

// ------------------------------------------------------------ entropy

/// (Shannon entropy of bytes H0, brotli bits/byte, high-order entropy = H0 - bpb)
/// over the concatenated tapes of up to `max` occupied cells (evenly sampled).
pub fn high_order_entropy(st: &State, max: usize) -> (f64, f64, f64) {
    let occ: Vec<usize> = (0..st.n).filter(|&c| st.occupied(c) && st.len(c) > 0).collect();
    if occ.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let step = (occ.len() as f64 / max as f64).max(1.0);
    let mut data = Vec::new();
    let mut i = 0.0;
    while (i as usize) < occ.len() {
        data.extend_from_slice(st.tape(occ[i as usize]));
        i += step;
    }
    let mut hist = [0u64; 256];
    for &b in &data {
        hist[b as usize] += 1;
    }
    let n = data.len() as f64;
    let h0: f64 = hist.iter().filter(|&&c| c > 0).map(|&c| -(c as f64 / n) * (c as f64 / n).log2()).sum();
    let mut out = Vec::new();
    {
        let mut w = brotli::CompressorWriter::new(&mut out, 4096, 2, 22);
        w.write_all(&data).unwrap();
    }
    let bpb = out.len() as f64 * 8.0 / n;
    (h0, bpb, h0 - bpb)
}

// ------------------------------------------------------------ replicator assay

#[derive(Clone, Debug)]
pub struct AssayResult {
    pub genome: Vec<u8>,
    pub count: usize,
    pub identity: f64,
    pub replicator: bool,
    pub minimal: Vec<u8>,
}

fn sandbox_params(cfg: &Config, len: usize) -> Params {
    let mut c = Config::preset("paper-0d").unwrap();
    c.world.size = vec![2];
    c.individual.l_max = (cfg.individual.l_max).max(((len + 3) / 4 * 4) as u32);
    c.individual.l_init = 0;
    c.individual.step_budget = cfg.individual.step_budget;
    c.individual.pair_order = PairOrder::SelfFirst;
    c.mutation.enabled = false;
    c.seed = 1;
    Params::new(&c)
}

/// Run genome + empty partner once with all physics off; return the fraction
/// of partner bytes equal to the genome.
pub fn sandbox_identity(p: &Params, genome: &[u8]) -> f64 {
    if genome.is_empty() {
        return 0.0;
    }
    let mut st = State::empty(p);
    let l = st.lmax;
    let n = genome.len().min(l);
    st.tapes[..n].copy_from_slice(&genome[..n]);
    for (c, len) in [(0usize, n), (1usize, n)] {
        st.hdr[c * HDR + H_LEN] = len as u32;
        st.hdr[c * HDR + H_FLAGS] = FL_OCC;
    }
    let snap = flag_snapshot(&st);
    exec_slice(&mut st, p, &snap, 0, Some(1), None);
    // identity over the genome's non-empty sites, so that mostly-empty tapes
    // do not "match" the empty partner
    let plen = st.len(1);
    let nz: Vec<usize> = (0..n).filter(|&i| genome[i] != 0).collect();
    if nz.len() < 2 {
        return 0.0;
    }
    let same = nz.iter().filter(|&&i| i < plen && st.tapes[l + i] == genome[i]).count();
    same as f64 / nz.len() as f64
}

pub fn assay_genome(cfg: &Config, genome: &[u8], count: usize) -> AssayResult {
    let p = sandbox_params(cfg, genome.len());
    let identity = sandbox_identity(&p, genome);
    let replicator = identity >= 0.9;
    let mut minimal = vec![];
    if replicator {
        let mut g = genome.to_vec();
        for i in 0..g.len() {
            if g[i] == 0 {
                continue;
            }
            let old = g[i];
            g[i] = 0;
            // the ablated genome must still copy the original's essential bytes
            if sandbox_identity(&p, &g) < 0.9 {
                g[i] = old;
            }
        }
        let first = g.iter().position(|&b| b != 0).unwrap_or(0);
        let last = g.iter().rposition(|&b| b != 0).unwrap_or(0);
        minimal = g[first..=last].to_vec();
    }
    AssayResult { genome: genome.to_vec(), count, identity, replicator, minimal }
}

pub fn top_genomes(st: &State, k: usize) -> Vec<(Vec<u8>, usize)> {
    let mut counts: HashMap<&[u8], usize> = HashMap::new();
    for c in 0..st.n {
        if st.occupied(c) && st.len(c) > 0 {
            *counts.entry(st.tape(c)).or_insert(0) += 1;
        }
    }
    let mut v: Vec<(Vec<u8>, usize)> = counts.into_iter().map(|(g, n)| (g.to_vec(), n)).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v.truncate(k);
    v
}

// ------------------------------------------------------------ species

pub const NHASH: usize = 32;
const BANDS: usize = 8;
const ROWS: usize = 4;

fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

pub fn minhash(t: &[u8]) -> [u32; NHASH] {
    let mut sig = [u32::MAX; NHASH];
    if t.len() < 4 {
        let g = t.iter().fold(0u64, |a, &b| a << 8 | b as u64);
        for (i, s) in sig.iter_mut().enumerate() {
            *s = mix(g ^ ((i as u64) << 40)) as u32;
        }
        return sig;
    }
    for w in t.windows(4) {
        let g = u32::from_le_bytes([w[0], w[1], w[2], w[3]]) as u64;
        for (i, s) in sig.iter_mut().enumerate() {
            let h = mix(g ^ ((i as u64 + 1) << 32)) as u32;
            if h < *s {
                *s = h;
            }
        }
    }
    sig
}

pub fn sig_jaccard(a: &[u32; NHASH], b: &[u32; NHASH]) -> f64 {
    a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / NHASH as f64
}

#[derive(Clone, Debug)]
pub struct Species {
    pub id: u64,
    pub sig: [u32; NHASH],
    pub representative: Vec<u8>,
    pub first_seen: u64,
    pub last_seen: u64,
}

/// Species seen in recent censuses, with a banded MinHash index so a
/// lookup costs a few hash probes instead of a scan of every species ever seen.
#[derive(Default)]
pub struct Registry {
    species: Vec<Species>,
    index: HashMap<(u8, [u32; ROWS]), Vec<usize>>,
    /// number of species ever registered (also the next id)
    pub total: u64,
    censuses: u64,
}

impl Registry {
    fn band(sig: &[u32; NHASH], b: usize) -> (u8, [u32; ROWS]) {
        let mut k = [0u32; ROWS];
        k.copy_from_slice(&sig[b * ROWS..(b + 1) * ROWS]);
        (b as u8, k)
    }
    /// Best match with estimated Jaccard >= 0.8 among species sharing a band.
    fn find(&self, sig: &[u32; NHASH]) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for b in 0..BANDS {
            if let Some(v) = self.index.get(&Self::band(sig, b)) {
                for &i in v {
                    let j = sig_jaccard(&self.species[i].sig, sig);
                    if j >= 0.8 && best.map(|x| j > x.1).unwrap_or(true) {
                        best = Some((i, j));
                    }
                }
            }
        }
        best.map(|x| x.0)
    }
    fn insert(&mut self, s: Species) -> usize {
        let i = self.species.len();
        for b in 0..BANDS {
            self.index.entry(Self::band(&s.sig, b)).or_default().push(i);
        }
        self.species.push(s);
        i
    }
    /// Forget species not seen for `keep` censuses (bounds memory and time).
    fn prune(&mut self, min_last_seen: u64) {
        if self.species.iter().all(|s| s.last_seen >= min_last_seen) {
            return;
        }
        let old = std::mem::take(&mut self.species);
        self.index.clear();
        for s in old.into_iter().filter(|s| s.last_seen >= min_last_seen) {
            self.insert(s);
        }
    }
    pub fn live(&self) -> usize {
        self.species.len()
    }
}

#[derive(Clone, Debug)]
pub struct Cluster {
    pub species: u64,
    pub first_seen: u64,
    pub representative: Vec<u8>,
    pub abundance: usize,
    pub centroid: [f64; 3],
    pub extent: f64,
    pub mean_len: f64,
    pub hand: i32,
    pub new: bool,
}

/// Cluster genomes (Jaccard >= 0.8 over byte 4-grams) and match the clusters
/// against the species registry.
pub fn census(st: &State, p: &Params, registry: &mut Registry, epoch: u64, max_cells: usize) -> Vec<Cluster> {
    registry.censuses += 1;
    let occ: Vec<usize> = (0..st.n).filter(|&c| st.occupied(c) && st.len(c) > 0).collect();
    let step = (occ.len() as f64 / max_cells as f64).max(1.0);
    let mut cells = vec![];
    let mut i = 0.0;
    while (i as usize) < occ.len() {
        cells.push(occ[i as usize]);
        i += step;
    }
    // unique genomes first
    let mut uniq: HashMap<&[u8], Vec<usize>> = HashMap::new();
    for &c in &cells {
        uniq.entry(st.tape(c)).or_default().push(c);
    }
    let mut gens: Vec<(&[u8], Vec<usize>)> = uniq.into_iter().collect();
    gens.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(b.0)));
    let sigs: Vec<[u32; NHASH]> = gens.iter().map(|(g, _)| minhash(g)).collect();
    let n = gens.len();
    let mut par: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for b in 0..BANDS {
        let mut buckets: HashMap<[u32; ROWS], usize> = HashMap::new();
        for (j, s) in sigs.iter().enumerate() {
            let mut key = [0u32; ROWS];
            key.copy_from_slice(&s[b * ROWS..(b + 1) * ROWS]);
            match buckets.get(&key) {
                Some(&k) => {
                    if sig_jaccard(&sigs[k], s) >= 0.8 {
                        let (x, y) = (find(&mut par, k), find(&mut par, j));
                        if x != y {
                            par[x.max(y)] = x.min(y);
                        }
                    }
                }
                None => {
                    buckets.insert(key, j);
                }
            }
        }
    }
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for j in 0..n {
        let r = find(&mut par, j);
        groups.entry(r).or_default().push(j);
    }
    let g = &p.geom;
    let mut out = vec![];
    let scale = step;
    for (root, members) in groups {
        let cellsm: Vec<usize> = members.iter().flat_map(|&j| gens[j].1.iter().copied()).collect();
        let abundance = (cellsm.len() as f64 * scale).round() as usize;
        let mut cen = [0.0; 3];
        let mut len = 0.0;
        let (mut nl, mut nd) = (0i64, 0i64);
        for &c in &cellsm {
            let q = g.coords(c);
            for a in 0..3 {
                cen[a] += q[a] as f64;
            }
            let t = st.tape(c);
            len += t.len() as f64;
            for &b in t {
                if isa::is_chiral(b) {
                    if b & 1 == 0 { nl += 1 } else { nd += 1 }
                }
            }
        }
        let k = cellsm.len() as f64;
        for a in 0..3 {
            cen[a] /= k;
        }
        let ext = (cellsm
            .iter()
            .map(|&c| {
                let q = g.coords(c);
                (0..3).map(|a| (q[a] as f64 - cen[a]).powi(2)).sum::<f64>()
            })
            .sum::<f64>()
            / k)
            .sqrt();
        let rep_sig = sigs[root];
        let (idx, new) = match registry.find(&rep_sig) {
            Some(i) => {
                registry.species[i].last_seen = epoch;
                (i, false)
            }
            None => {
                let id = registry.total;
                registry.total += 1;
                (registry.insert(Species { id, sig: rep_sig, representative: gens[root].0.to_vec(), first_seen: epoch, last_seen: epoch }), true)
            }
        };
        let sp = &registry.species[idx];
        out.push(Cluster {
            species: sp.id,
            first_seen: sp.first_seen,
            representative: sp.representative.clone(),
            abundance,
            centroid: cen,
            extent: ext,
            mean_len: len / k,
            hand: (nl - nd).signum() as i32,
            new,
        });
    }
    out.sort_by(|a, b| b.abundance.cmp(&a.abundance));
    // keep species seen in the last 5 censuses (indices are rebuilt, ids stay)
    if registry.censuses % 5 == 0 {
        let horizon = epoch.saturating_sub(5 * (epoch / registry.censuses.max(1)).max(1));
        registry.prune(horizon);
    }
    out
}

// ------------------------------------------------------------ chirality

pub fn ee(l: i64, d: i64) -> f64 {
    if l + d == 0 { 0.0 } else { (l - d) as f64 / (l + d) as f64 }
}

pub struct Chirality {
    pub code: (i64, i64),
    pub chem: Vec<(i64, i64)>,
    pub mono: (i64, i64),
    pub sig: (i64, i64),
    /// per-block code (L, D) counts, 16 blocks per axis
    pub blocks: Vec<(i64, i64)>,
    pub nb: [usize; 3],
}

pub fn chirality(st: &State, p: &Params) -> Chirality {
    let g = &p.geom;
    let nb = [if g.dims >= 1 { 16.min(g.size[0] as usize) } else { 1 }, if g.dims >= 2 { 16.min(g.size[1] as usize) } else { 1 }, if g.dims >= 3 { 16.min(g.size[2] as usize) } else { 1 }];
    let mut blocks = vec![(0i64, 0i64); nb[0] * nb[1] * nb[2]];
    let mut code = (0, 0);
    let mut chem = vec![(0i64, 0i64); p.k];
    let mut mono = (0, 0);
    let mut sig = (0, 0);
    for c in 0..st.n {
        let q = g.coords(c);
        let bi = |a: usize| (q[a] as usize * nb[a]) / g.size[a] as usize;
        let b = bi(0) + nb[0] * (bi(1) + nb[1] * bi(2));
        if st.occupied(c) {
            for &x in st.tape(c) {
                if isa::is_chiral(x) {
                    if x & 1 == 0 {
                        code.0 += 1;
                        blocks[b].0 += 1;
                    } else {
                        code.1 += 1;
                        blocks[b].1 += 1;
                    }
                }
            }
        }
        for k in 0..p.k {
            chem[k].0 += st.f(c, p.f_chem(k, 0)) as i64;
            chem[k].1 += st.f(c, p.f_chem(k, 1)) as i64;
        }
        if p.matter_on {
            mono.0 += st.f(c, p.f_pool(0)) as i64;
            mono.1 += st.f(c, p.f_pool(1)) as i64;
        }
        for j in 0..p.s {
            sig.0 += st.f(c, p.f_sig(j, 0)) as i64;
            sig.1 += st.f(c, p.f_sig(j, 1)) as i64;
        }
    }
    Chirality { code, chem, mono, sig, blocks, nb }
}

/// Domain statistics on the smoothed per-block code EE map, thresholded at
/// |EE| > 0.3: (domain count, mean domain size in blocks, interface length).
pub fn domains(ch: &Chirality) -> (usize, f64, usize, Vec<i8>) {
    let nb = ch.nb;
    let n = nb[0] * nb[1] * nb[2];
    let idx = |x: usize, y: usize, z: usize| x + nb[0] * (y + nb[1] * z);
    let mut sm = vec![0.0f64; n];
    for z in 0..nb[2] {
        for y in 0..nb[1] {
            for x in 0..nb[0] {
                let (mut l, mut d) = (0i64, 0i64);
                for dz in -1i64..=1 {
                    for dy in -1i64..=1 {
                        for dx in -1i64..=1 {
                            let w = |v: usize, dv: i64, m: usize| ((v as i64 + dv).rem_euclid(m as i64)) as usize;
                            let (xx, yy, zz) = (w(x, dx, nb[0]), w(y, dy, nb[1]), w(z, dz, nb[2]));
                            let b = ch.blocks[idx(xx, yy, zz)];
                            l += b.0;
                            d += b.1;
                        }
                    }
                }
                sm[idx(x, y, z)] = ee(l, d);
            }
        }
    }
    let sign: Vec<i8> = sm.iter().map(|&e| if e > 0.3 { 1 } else if e < -0.3 { -1 } else { 0 }).collect();
    let mut lab = vec![usize::MAX; n];
    let mut sizes = vec![];
    let nbrs = |i: usize| -> Vec<usize> {
        let x = i % nb[0];
        let y = (i / nb[0]) % nb[1];
        let z = i / (nb[0] * nb[1]);
        let mut v = vec![];
        for (a, m) in nb.iter().enumerate() {
            if *m < 2 {
                continue;
            }
            for s in [-1i64, 1] {
                let mut q = [x, y, z];
                q[a] = ((q[a] as i64 + s).rem_euclid(*m as i64)) as usize;
                v.push(idx(q[0], q[1], q[2]));
            }
        }
        v
    };
    for i in 0..n {
        if sign[i] == 0 || lab[i] != usize::MAX {
            continue;
        }
        let id = sizes.len();
        let mut stack = vec![i];
        lab[i] = id;
        let mut sz = 0;
        while let Some(j) = stack.pop() {
            sz += 1;
            for k in nbrs(j) {
                if sign[k] == sign[i] && lab[k] == usize::MAX {
                    lab[k] = id;
                    stack.push(k);
                }
            }
        }
        sizes.push(sz);
    }
    let mut interface = 0;
    for i in 0..n {
        for k in nbrs(i) {
            if k > i && sign[i] * sign[k] == -1 {
                interface += 1;
            }
        }
    }
    let mean = if sizes.is_empty() { 0.0 } else { sizes.iter().sum::<usize>() as f64 / sizes.len() as f64 };
    (sizes.len(), mean, interface, sign)
}
