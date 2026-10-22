//! Telemetry (§13): ledgers, population, complexity, chirality, bodies,
//! ecology, species and lineage, streamed to Parquet in the run directory.

pub mod analysis;
pub mod table;

use analysis::*;
use anyhow::Result;
use primordia_core::isa;
use primordia_core::params::*;
use primordia_core::phases;
use primordia_core::Sim;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use table::{Table, Ty, V};

/// Headline numbers, also used for the batch summary and the viewer.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Summary {
    pub epoch: u64,
    pub occupied: u64,
    pub halted: u64,
    pub mean_len: f64,
    pub hoe: f64,
    pub ee_code: f64,
    pub ee_chem: f64,
    pub ee_mono: f64,
    pub first_replicator: Option<u64>,
    pub replicators: usize,
    pub best_replicator: Option<String>,
    pub t_b: Option<u64>,
    pub winning_hand: Option<String>,
    pub species: usize,
    pub species_total: usize,
    pub bodies: usize,
    pub max_body: usize,
    pub organisms: usize,
    pub ledger_violations: u64,
    pub energy_residual: i64,
    pub matter_residual: [i64; 3],
    pub atom_residual: i64,
}

pub struct Recorder {
    pub dir: Option<PathBuf>,
    cfg: primordia_core::config::TelemetryCfg,
    metrics: Table,
    chir: Table,
    species_t: Table,
    lineage: Table,
    opcodes: Table,
    ecology: Table,
    assay: Table,
    registry: Vec<Species>,
    tb_start: Option<(u64, i32)>,
    last_interface: Option<(u64, usize)>,
    pub summary: Summary,
    /// Recent time series for live plots (metric name -> (epoch, value)).
    pub series: BTreeMap<String, Vec<[f64; 2]>>,
    pub last_domains: Vec<i8>,
    pub last_assay: Vec<AssayResult>,
    pub max_series: usize,
}

fn cols(c: &[(&'static str, Ty)]) -> Vec<(&'static str, Ty)> {
    c.to_vec()
}

impl Recorder {
    pub fn new(dir: Option<&Path>, sim: &Sim) -> Result<Recorder> {
        if let Some(d) = dir {
            std::fs::create_dir_all(d)?;
        }
        use Ty::*;
        let mut chem_cols: Vec<(&'static str, Ty)> = vec![("epoch", I), ("ee_code", F), ("ee_mono", F), ("ee_signal", F)];
        let names: Vec<&'static str> = (0..sim.p.k).map(|k| &*Box::leak(format!("ee_chem_{}", sim.cfg.chemistry.species[k]).into_boxed_str())).collect();
        for n in names {
            chem_cols.push((n, F));
        }
        chem_cols.extend([("domains", I), ("mean_domain", F), ("interface", I), ("interface_velocity", F), ("code_l", I), ("code_d", I)]);
        Ok(Recorder {
            dir: dir.map(|d| d.to_path_buf()),
            cfg: sim.cfg.telemetry.clone(),
            metrics: Table::new(
                dir,
                "metrics",
                &cols(&[
                    ("epoch", I),
                    ("occupied", I),
                    ("halted", I),
                    ("sleeping", I),
                    ("mean_len", F),
                    ("births", I),
                    ("deaths", I),
                    ("energy_total", I),
                    ("mean_f", F),
                    ("mean_t_kelvin", F),
                    ("mean_e", F),
                    ("energy_residual", I),
                    ("matter_residual_l", I),
                    ("matter_residual_d", I),
                    ("matter_residual_a", I),
                    ("atom_residual", I),
                    ("ledger_violations", I),
                    ("body_moves", I),
                    ("bodies", I),
                    ("mean_body", F),
                    ("max_body", I),
                    ("mean_exposure", F),
                    ("organisms", I),
                    ("h0", F),
                    ("bpb", F),
                    ("hoe", F),
                    ("vents", I),
                    ("species", I),
                ]),
            )?,
            chir: Table::new(dir, "chirality", &chem_cols)?,
            species_t: Table::new(
                dir,
                "species",
                &cols(&[("epoch", I), ("species", I), ("abundance", I), ("cx", F), ("cy", F), ("cz", F), ("extent", F), ("mean_len", F), ("hand", I), ("first_seen", I), ("representative", S)]),
            )?,
            lineage: Table::new(dir, "lineage", &cols(&[("epoch", I), ("parent", I), ("child", I), ("position", I), ("depth", I), ("genome_hash", I)]))?,
            opcodes: Table::new(dir, "opcodes", &cols(&[("epoch", I), ("opcode", I), ("count", I)]))?,
            ecology: Table::new(dir, "ecology", &cols(&[("epoch", I), ("block", I), ("mean_f", F), ("mean_t_kelvin", F), ("mean_chem", F), ("occupancy", F)]))?,
            assay: Table::new(dir, "assay", &cols(&[("epoch", I), ("rank", I), ("count", I), ("identity", F), ("replicator", I), ("genome", S), ("minimal", S)]))?,
            registry: vec![],
            tb_start: None,
            last_interface: None,
            summary: Summary::default(),
            series: BTreeMap::new(),
            last_domains: vec![],
            last_assay: vec![],
            max_series: 20_000,
        })
    }

    fn series(&mut self, name: &str, e: u64, v: f64) {
        let s = self.series.entry(name.to_string()).or_default();
        s.push([e as f64, v]);
        if s.len() > self.max_series {
            let keep: Vec<[f64; 2]> = s.iter().step_by(2).copied().collect();
            *s = keep;
        }
    }

    /// Record everything due after the epoch that just finished.
    pub fn record(&mut self, sim: &Sim) -> Result<()> {
        let st = &sim.st;
        let p = &sim.p;
        let e = st.epoch; // epochs completed
        let due = |every: u64| every > 0 && e % every == 0;
        // lineage: every birth this epoch
        for c in 0..st.n {
            let h = &st.hdr[c * HDR..(c + 1) * HDR];
            if h[H_FLAGS] & FL_NEWBORN != 0 {
                let gh = st.tape(c).iter().fold(0xcbf29ce484222325u64, |a, &b| (a ^ b as u64).wrapping_mul(0x100000001b3));
                self.lineage.push(vec![V::I(e as i64 - 1), V::I(h[H_PARENT] as i64), V::I(h[H_LINEAGE] as i64), V::I(c as i64), V::I(h[H_DEPTH] as i64), V::I(gh as i64)])?;
            }
        }
        let mut s = std::mem::take(&mut self.summary);
        s.epoch = e;
        s.ledger_violations = st.ledger.violations;
        s.energy_residual = st.ledger.energy_residual;
        s.matter_residual = st.ledger.matter_residual;
        s.atom_residual = st.ledger.atom_residual;
        if due(self.cfg.metrics_every.max(1)) {
            let mut occ = 0u64;
            let mut halted = 0u64;
            let mut sleeping = 0u64;
            let mut len = 0u64;
            let mut esum = 0u64;
            let (mut fsum, mut tsum) = (0u64, 0u64);
            for c in 0..st.n {
                let h = &st.hdr[c * HDR..(c + 1) * HDR];
                fsum += st.f(c, F_F) as u64;
                tsum += st.f(c, F_T) as u64;
                if h[H_FLAGS] & FL_OCC != 0 {
                    occ += 1;
                    len += h[H_LEN] as u64;
                    esum += h[H_E] as u64;
                    if h[H_FLAGS] & FL_HALT != 0 {
                        halted += 1;
                    }
                    if h[H_SLEEP] > 0 {
                        sleeping += 1;
                    }
                }
            }
            s.occupied = occ;
            s.halted = halted;
            s.mean_len = if occ > 0 { len as f64 / occ as f64 } else { 0.0 };
            let n = st.n as f64;
            // bodies (every opcode_every to bound cost)
            let (mut nb, mut mean_b, mut max_b, mut mexp, mut orgs) = (s.bodies as i64, 0.0, s.max_body as i64, 0.0, s.organisms as i64);
            if due(self.cfg.opcode_every.max(1)) {
                let bodies = phases::components(st, p);
                nb = bodies.count as i64;
                max_b = bodies.sizes.iter().map(|x| x.1).max().unwrap_or(0) as i64;
                let big: Vec<_> = bodies.sizes.iter().filter(|x| x.1 >= 2).collect();
                mean_b = if big.is_empty() { 0.0 } else { big.iter().map(|x| x.1 as f64).sum::<f64>() / big.len() as f64 };
                let ex = phases::exposure(st, p);
                mexp = if occ > 0 { (0..st.n).filter(|&c| st.occupied(c)).map(|c| ex[c] as f64 / 65536.0).sum::<f64>() / occ as f64 } else { 0.0 };
                orgs = organisms(st, &bodies) as i64;
                s.bodies = nb as usize;
                s.max_body = max_b as usize;
                s.organisms = orgs as usize;
                // opcode histogram
                let mut hist = [0u64; 256];
                for c in 0..st.n {
                    if st.occupied(c) {
                        for &b in st.tape(c) {
                            hist[b as usize] += 1;
                        }
                    }
                }
                for (b, &cnt) in hist.iter().enumerate() {
                    if cnt > 0 {
                        self.opcodes.push(vec![V::I(e as i64), V::I(b as i64), V::I(cnt as i64)])?;
                    }
                }
            }
            let (mut h0, mut bpb, mut hoe) = (f64::NAN, f64::NAN, f64::NAN);
            if due(self.cfg.entropy_every.max(1)) {
                (h0, bpb, hoe) = high_order_entropy(st, self.cfg.entropy_samples as usize);
                s.hoe = hoe;
                self.series("high-order entropy", e, hoe);
            }
            self.metrics.push(vec![
                V::I(e as i64),
                V::I(occ as i64),
                V::I(halted as i64),
                V::I(sleeping as i64),
                V::F(s.mean_len),
                V::I(st.tally.births as i64),
                V::I(st.tally.deaths as i64),
                V::I(st.ledger.energy_prev),
                V::F(fsum as f64 / n),
                V::F(tsum as f64 / n / 100.0),
                V::F(if occ > 0 { esum as f64 / occ as f64 } else { 0.0 }),
                V::I(st.ledger.energy_residual),
                V::I(st.ledger.matter_residual[0]),
                V::I(st.ledger.matter_residual[1]),
                V::I(st.ledger.matter_residual[2]),
                V::I(st.ledger.atom_residual),
                V::I(st.ledger.violations as i64),
                V::I(st.tally.body_moves as i64),
                V::I(nb),
                V::F(mean_b),
                V::I(max_b),
                V::F(mexp),
                V::I(orgs),
                V::F(h0),
                V::F(bpb),
                V::F(hoe),
                V::I(st.vents.len() as i64),
                V::I(s.species as i64),
            ])?;
            self.series("population", e, occ as f64);
            self.series("mean tape length", e, s.mean_len);
            self.series("ledger |residual|", e, (st.ledger.energy_residual.abs() + st.ledger.matter_residual.iter().map(|x| x.abs()).sum::<i64>() + st.ledger.atom_residual.abs()) as f64);
            self.series("births", e, st.tally.births as f64);
            self.series("mean T (K)", e, tsum as f64 / n / 100.0);
            self.series("mean F", e, fsum as f64 / n);
            self.series("mean E", e, if occ > 0 { esum as f64 / occ as f64 } else { 0.0 });
        }
        if due(self.cfg.chirality_every.max(1)) {
            let ch = chirality(st, p);
            let (nd, md, iface, sign) = domains(&ch);
            let vel = match self.last_interface {
                Some((e0, i0)) if e > e0 => (iface as f64 - i0 as f64).abs() / (e - e0) as f64,
                _ => 0.0,
            };
            self.last_interface = Some((e, iface));
            self.last_domains = sign;
            s.ee_code = ee(ch.code.0, ch.code.1);
            let (cl, cd) = ch.chem.iter().fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
            s.ee_chem = ee(cl, cd);
            s.ee_mono = ee(ch.mono.0, ch.mono.1);
            let mut row = vec![V::I(e as i64), V::F(s.ee_code), V::F(s.ee_mono), V::F(ee(ch.sig.0, ch.sig.1))];
            for c in &ch.chem {
                row.push(V::F(ee(c.0, c.1)));
            }
            row.extend([V::I(nd as i64), V::F(md), V::I(iface as i64), V::F(vel), V::I(ch.code.0), V::I(ch.code.1)]);
            self.chir.push(row)?;
            self.series("EE code", e, s.ee_code);
            self.series("EE chemicals", e, s.ee_chem);
            self.series("EE monomers", e, s.ee_mono);
            // symmetry-breaking time
            let hand = if s.ee_code > 0.5 { 1 } else if s.ee_code < -0.5 { -1 } else { 0 };
            match (self.tb_start, hand) {
                (_, 0) => self.tb_start = None,
                (Some((_, h)), x) if h != x => self.tb_start = Some((e, x)),
                (None, x) => self.tb_start = Some((e, x)),
                _ => {}
            }
            if let (Some((t0, h)), None) = (self.tb_start, s.t_b) {
                if e - t0 >= 1000 {
                    s.t_b = Some(t0);
                    s.winning_hand = Some(if h > 0 { "L".into() } else { "D".into() });
                }
            }
        }
        if due(self.cfg.ecology_every.max(1)) && p.geom.dims > 0 {
            let g = &p.geom;
            let nbk = [16.min(g.size[0]) as usize, if g.dims >= 2 { 16.min(g.size[1]) as usize } else { 1 }, if g.dims >= 3 { 16.min(g.size[2]) as usize } else { 1 }];
            let nblk = nbk[0] * nbk[1] * nbk[2];
            let mut acc = vec![[0f64; 5]; nblk];
            for c in 0..st.n {
                let q = g.coords(c);
                let bi = |a: usize| (q[a] as usize * nbk[a]) / g.size[a] as usize;
                let b = bi(0) + nbk[0] * (bi(1) + nbk[1] * bi(2));
                acc[b][0] += st.f(c, F_F) as f64;
                acc[b][1] += st.f(c, F_T) as f64 / 100.0;
                acc[b][2] += (0..p.k).map(|k| (st.f(c, p.f_chem(k, 0)) + st.f(c, p.f_chem(k, 1))) as f64).sum::<f64>();
                acc[b][3] += st.occupied(c) as u32 as f64;
                acc[b][4] += 1.0;
            }
            for (b, a) in acc.iter().enumerate() {
                self.ecology.push(vec![V::I(e as i64), V::I(b as i64), V::F(a[0] / a[4]), V::F(a[1] / a[4]), V::F(a[2] / a[4]), V::F(a[3] / a[4])])?;
            }
        }
        if due(self.cfg.assay_every.max(1)) {
            let top = top_genomes(st, 100);
            let cfg = sim.cfg.clone();
            use rayon::prelude::*;
            let res: Vec<AssayResult> = top.par_iter().map(|(g, n)| assay_genome(&cfg, g, *n)).collect();
            s.replicators = res.iter().filter(|r| r.replicator).count();
            if s.replicators > 0 && s.first_replicator.is_none() {
                s.first_replicator = Some(e);
            }
            s.best_replicator = res.iter().filter(|r| r.replicator).min_by_key(|r| r.minimal.len()).map(|r| isa::to_hex(&r.minimal));
            for (i, r) in res.iter().enumerate() {
                self.assay.push(vec![
                    V::I(e as i64),
                    V::I(i as i64),
                    V::I(r.count as i64),
                    V::F(r.identity),
                    V::I(r.replicator as i64),
                    V::S(isa::to_hex(&r.genome)),
                    V::S(isa::to_hex(&r.minimal)),
                ])?;
            }
            self.series("replicators (top 100)", e, s.replicators as f64);
            self.last_assay = res;
        }
        if due(self.cfg.census_every.max(1)) {
            let cl = census(st, p, &mut self.registry, e, 20_000);
            s.species = cl.len();
            s.species_total = self.registry.len();
            for c in &cl {
                let sp = &self.registry[c.species as usize];
                self.species_t.push(vec![
                    V::I(e as i64),
                    V::I(c.species as i64),
                    V::I(c.abundance as i64),
                    V::F(c.centroid[0]),
                    V::F(c.centroid[1]),
                    V::F(c.centroid[2]),
                    V::F(c.extent),
                    V::F(c.mean_len),
                    V::I(c.hand as i64),
                    V::I(sp.first_seen as i64),
                    V::S(isa::to_hex(&sp.representative)),
                ])?;
            }
            self.series("species", e, s.species as f64);
        }
        self.summary = s;
        Ok(())
    }

    pub fn finish(&mut self) -> Result<Summary> {
        for t in [&mut self.metrics, &mut self.chir, &mut self.species_t, &mut self.lineage, &mut self.opcodes, &mut self.ecology, &mut self.assay] {
            t.close()?;
        }
        if let Some(d) = &self.dir {
            std::fs::write(d.join("summary.json"), serde_json::to_string_pretty(&self.summary)?)?;
        }
        Ok(self.summary.clone())
    }
}

/// Operational organism candidates (§13.2, simplified): bonded bodies of
/// size >= 2 whose members all descend from one parent lineage or from each
/// other.
pub fn organisms(st: &primordia_core::State, bodies: &phases::Bodies) -> usize {
    let mut members: HashMap<u32, Vec<usize>> = HashMap::new();
    for c in 0..st.n {
        if bodies.label[c] != NONE {
            members.entry(bodies.label[c]).or_default().push(c);
        }
    }
    members
        .values()
        .filter(|m| m.len() >= 2)
        .filter(|m| {
            let lin: HashSet<u32> = m.iter().map(|&c| st.h(c, H_LINEAGE)).collect();
            let parents: HashSet<u32> = m.iter().map(|&c| st.h(c, H_PARENT)).filter(|p| !lin.contains(p)).collect();
            parents.len() <= 1 && m.iter().any(|&c| st.h(c, H_PARENT) != 0)
        })
        .count()
}
