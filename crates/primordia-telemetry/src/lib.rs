//! Telemetry (§13): ledgers, population, complexity, chirality, bodies,
//! ecology, species and lineage, streamed to Parquet in the run directory.

pub mod analysis;
pub mod clusters;
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
    /// Shares of living cells (0..1): asleep, dormant (zero energy), carrying
    /// SLEEP in their program, with at least one bond, older than one day.
    pub asleep: f64,
    pub dormant: f64,
    pub sleep_carriers: f64,
    pub bonded: f64,
    pub older_than_day: f64,
    pub mean_age: f64,
    /// Population at the last sunrise / the sunset before it.
    pub night_survival: f64,
    pub births_last_day: u64,
    /// Longest run of nights survived by any tracked body / kin cluster.
    pub body_nights: u32,
    pub kin_nights: u32,
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
    registry: Registry,
    tb_start: Option<(u64, i32)>,
    last_interface: Option<(u64, usize)>,
    pub summary: Summary,
    /// Recent time series for live plots (metric name -> (epoch, value)).
    pub series: BTreeMap<String, Vec<[f64; 2]>>,
    /// The same metrics as mean / min / max buckets over the whole run
    /// (merged pairwise when full, so day/night swings never alias).
    pub bands: BTreeMap<String, Vec<Band>>,
    pub last_domains: Vec<i8>,
    pub last_assay: Vec<AssayResult>,
    pub max_series: usize,
    life: Table,
    days: Table,
    clusters_t: Table,
    tracker: clusters::Tracker,
    /// Longest-lived clusters at the last cluster sample (both kinds).
    pub last_clusters: Vec<clusters::Cluster>,
    pop_sunset: Option<u64>,
    pop_sunrise: Option<u64>,
    births_day: u64,
}

/// Behaviour instructions whose frequency is logged as a live series.
pub const TRACKED_OPS: [u8; 23] = [
    0x12, 0x61, 0x62, 0x6e, 0x64, 0x63, 0x60, 0x69, 0x66, 0x67, 0x68, 0x1d, 0x1e, 0x6b, 0x65, 0x2c, 0x2d, 0x30, 0x31, 0x32, 0x33, 0x6f, 0x0d,
];

/// A bucket of consecutive samples of one metric.
#[derive(Clone, Copy, Debug)]
pub struct Band {
    pub e0: f64,
    pub e1: f64,
    pub n: f64,
    pub sum: f64,
    pub lo: f64,
    pub hi: f64,
}

impl Band {
    pub fn merge(bs: &[Band]) -> Band {
        let mut m = bs[0];
        for b in &bs[1..] {
            m = Band { e0: m.e0.min(b.e0), e1: m.e1.max(b.e1), n: m.n + b.n, sum: m.sum + b.sum, lo: m.lo.min(b.lo), hi: m.hi.max(b.hi) };
        }
        m
    }
    pub fn mid(&self) -> f64 {
        (self.e0 + self.e1) / 2.0
    }
    pub fn mean(&self) -> f64 {
        self.sum / self.n
    }
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
            registry: Registry::default(),
            tb_start: None,
            last_interface: None,
            summary: Summary::default(),
            series: BTreeMap::new(),
            bands: BTreeMap::new(),
            last_domains: vec![],
            last_assay: vec![],
            max_series: 20_000,
            life: Table::new(
                dir,
                "life",
                &cols(&[
                    ("epoch", I),
                    ("sunlight", F),
                    ("asleep", F),
                    ("dormant", F),
                    ("sleep_carriers", F),
                    ("bonded", F),
                    ("mean_age", F),
                    ("older_1d", F),
                    ("older_2d", F),
                ]),
            )?,
            days: Table::new(
                dir,
                "days",
                &cols(&[
                    ("epoch", I),
                    ("day", I),
                    ("pop_sunset", I),
                    ("pop_sunrise", I),
                    ("night_survival", F),
                    ("births", I),
                    ("deaths", I),
                    ("asleep_dawn", F),
                    ("dormant_dawn", F),
                    ("largest_body_dawn", I),
                ]),
            )?,
            clusters_t: Table::new(
                dir,
                "clusters",
                &cols(&[
                    ("epoch", I),
                    ("kind", S),
                    ("clusters", I),
                    ("largest", I),
                    ("linked", I),
                    ("nights_1", I),
                    ("nights_2", I),
                    ("nights_3", I),
                    ("max_nights", I),
                    ("oldest_age", I),
                    ("cells_surviving", F),
                ]),
            )?,
            tracker: clusters::Tracker::default(),
            last_clusters: vec![],
            pop_sunset: None,
            pop_sunrise: None,
            births_day: 0,
        })
    }

    fn series(&mut self, name: &str, e: u64, v: f64) {
        let s = self.series.entry(name.to_string()).or_default();
        let b = self.bands.entry(name.to_string()).or_default();
        if v.is_finite() {
            b.push(Band { e0: e as f64, e1: e as f64, n: 1.0, sum: v, lo: v, hi: v });
        }
        if b.len() > self.max_series {
            *b = b.chunks(2).map(Band::merge).collect();
            *s = b.iter().map(|b| [b.mid(), b.mean()]).collect();
        }
        s.push([e as f64, v]);
        if s.len() > 2 * self.max_series {
            *s = s.iter().step_by(2).copied().collect();
        }
    }

    /// Record everything due after the epoch that just finished.
    /// True when `record` will read cell state after this epoch (so a
    /// resident simulation must provide a snapshot).
    pub fn needs_state(&self, epoch: u64) -> bool {
        needs_state(&self.cfg, epoch)
    }

    /// Record everything due after the epoch that just finished (blocking:
    /// syncs a resident simulation when cell state is needed).
    pub fn record(&mut self, sim: &mut Sim) -> Result<()> {
        let b = sim.take_births();
        self.record_births(&b)?;
        if self.needs_state(sim.st.epoch) {
            sim.sync();
            let cfg = sim.cfg.clone();
            self.record_state(&sim.st, &sim.p, &cfg)
        } else {
            self.record_ledger(&sim.st);
            Ok(())
        }
    }

    pub fn record_births(&mut self, births: &[primordia_core::world::Birth]) -> Result<()> {
        self.births_day += births.len() as u64;
        for b in births {
            self.lineage.push(vec![V::I(b[0] as i64), V::I(b[1] as i64), V::I(b[2] as i64), V::I(b[3] as i64), V::I(b[4] as i64), V::I(b[5] as i64)])?;
        }
        Ok(())
    }

    /// Update only the ledger part of the summary.
    pub fn record_ledger(&mut self, st: &primordia_core::State) {
        self.summary.epoch = st.epoch;
        self.summary.ledger_violations = st.ledger.violations;
        self.summary.energy_residual = st.ledger.energy_residual;
        self.summary.matter_residual = st.ledger.matter_residual;
        self.summary.atom_residual = st.ledger.atom_residual;
    }

    /// Record from a state snapshot taken after epoch `st.epoch - 1`.
    pub fn record_state(&mut self, st: &primordia_core::State, p: &primordia_core::Params, cfg: &primordia_core::Config) -> Result<()> {
        let e = st.epoch; // epochs completed
        let due = |every: u64| every > 0 && e % every == 0;
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
            let (mut carriers, mut bonded, mut age_sum, mut old1, mut old2) = (0u64, 0u64, 0u64, 0u64, 0u64);
            let day = p.day_period.max(1) as u32;
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
                    if st.tape(c).contains(&isa::op::SLEEP) {
                        carriers += 1;
                    }
                    if h[H_FLAGS] & BOND_MASK != 0 {
                        bonded += 1;
                    }
                    let age = (e as u32).wrapping_sub(h[H_BIRTH]);
                    age_sum += age as u64;
                    old1 += (age > day) as u64;
                    old2 += (age > 2 * day) as u64;
                }
            }
            let share = |x: u64| if occ > 0 { x as f64 / occ as f64 } else { 0.0 };
            let sunlight = if p.sun_on && p.sun_i0_q16 > 0 {
                let rows = phases::row_lights(p, e);
                rows.iter().map(|&r| r as f64).sum::<f64>() / rows.len().max(1) as f64 / p.sun_i0_q16 as f64
            } else {
                0.0
            };
            s.asleep = share(sleeping);
            s.dormant = share(halted);
            s.sleep_carriers = share(carriers);
            s.bonded = share(bonded);
            s.older_than_day = share(old1);
            s.mean_age = if occ > 0 { age_sum as f64 / occ as f64 } else { 0.0 };
            self.life.push(vec![
                V::I(e as i64),
                V::F(sunlight),
                V::F(s.asleep),
                V::F(s.dormant),
                V::F(s.sleep_carriers),
                V::F(s.bonded),
                V::F(s.mean_age),
                V::F(s.older_than_day),
                V::F(share(old2)),
            ])?;
            self.series("sunlight", e, sunlight);
            self.series("asleep %", e, 100.0 * s.asleep);
            self.series("dormant %", e, 100.0 * s.dormant);
            self.series("SLEEP carriers %", e, 100.0 * s.sleep_carriers);
            self.series("bonded cells %", e, 100.0 * s.bonded);
            self.series("mean age", e, s.mean_age);
            self.series("older than 1 day %", e, 100.0 * s.older_than_day);
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
                // frequency relative to a uniformly random byte (1 = chance)
                let total = hist.iter().sum::<u64>().max(1) as f64;
                for b in TRACKED_OPS {
                    self.series(&format!("op {}", isa::mnemonic(b)), e, hist[b as usize] as f64 / total * 256.0);
                }
                self.series("bodies", e, nb as f64);
                self.series("largest body", e, max_b as f64);
                self.series("mean body size", e, mean_b);
                self.series("organisms", e, orgs as f64);
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
            // day/night bookkeeping: sunset at half period, sunrise at 0
            let dp = p.day_period;
            if dp > 0 && e % dp == dp / 2 {
                self.pop_sunset = Some(occ);
                self.series("population at sunset", e, occ as f64);
            }
            if dp > 0 && e % dp == 0 && e > 0 {
                let surv = self.pop_sunset.map(|ps| if ps > 0 { occ as f64 / ps as f64 } else { 0.0 });
                let births = std::mem::take(&mut self.births_day);
                let deaths = self.pop_sunrise.map(|p0| (births as i64 - (occ as i64 - p0 as i64)).max(0));
                if let Some(v) = surv {
                    s.night_survival = v;
                    self.series("night survival %", e, 100.0 * v);
                }
                s.births_last_day = births;
                self.series("population at sunrise", e, occ as f64);
                self.series("births per day", e, births as f64);
                if let Some(d) = deaths {
                    self.series("deaths per day", e, d as f64);
                }
                self.series("asleep at dawn %", e, 100.0 * s.asleep);
                self.series("dormant at dawn %", e, 100.0 * s.dormant);
                self.series("largest body at dawn", e, max_b as f64);
                self.days.push(vec![
                    V::I(e as i64),
                    V::I((e / dp) as i64),
                    V::I(self.pop_sunset.map(|x| x as i64).unwrap_or(-1)),
                    V::I(occ as i64),
                    V::F(surv.unwrap_or(f64::NAN)),
                    V::I(births as i64),
                    V::I(deaths.unwrap_or(-1)),
                    V::F(s.asleep),
                    V::F(s.dormant),
                    V::I(max_b),
                ])?;
                self.pop_sunrise = Some(occ);
            }
        }
        if due(self.cfg.cluster_every) {
            let (stats, all) = self.tracker.observe(st, p);
            for (kind, k) in &stats {
                let name = match kind {
                    clusters::Kind::Body => "bodies",
                    clusters::Kind::Kin => "kin clusters",
                };
                self.clusters_t.push(vec![
                    V::I(e as i64),
                    V::S(name.into()),
                    V::I(k.clusters as i64),
                    V::I(k.largest as i64),
                    V::I(k.linked as i64),
                    V::I(k.nights_1 as i64),
                    V::I(k.nights_2 as i64),
                    V::I(k.nights_3 as i64),
                    V::I(k.max_nights as i64),
                    V::I(k.oldest_age as i64),
                    V::F(k.cells_surviving),
                ])?;
                self.series(&format!("{name} (>=50 cells)"), e, k.clusters as f64);
                self.series(&format!("{name} surviving >=1 night"), e, k.nights_1 as f64);
                self.series(&format!("{name} surviving >=3 nights"), e, k.nights_3 as f64);
                self.series(&format!("{name}: most nights survived"), e, k.max_nights as f64);
                self.series(&format!("cells in night-surviving {name} %"), e, 100.0 * k.cells_surviving);
                match kind {
                    clusters::Kind::Body => s.body_nights = k.max_nights,
                    clusters::Kind::Kin => s.kin_nights = k.max_nights,
                }
            }
            self.last_clusters = all.into_iter().take(12).collect();
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
            let cfg = cfg.clone();
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
            s.species_total = self.registry.total as usize;
            for c in &cl {
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
                    V::I(c.first_seen as i64),
                    V::S(isa::to_hex(&c.representative)),
                ])?;
            }
            self.series("species", e, s.species as f64);
        }
        self.summary = s;
        if due(cfg.telemetry.snapshot_every) {
            self.save_history()?;
        }
        Ok(())
    }

    /// Write the plot history to `<run>/history.json` (beside the snapshots),
    /// so a run resumed from this directory keeps its graphs.
    pub fn save_history(&self) -> Result<()> {
        let Some(d) = &self.dir else { return Ok(()) };
        let h: BTreeMap<&String, Vec<[f64; 6]>> = self.bands.iter().map(|(k, v)| (k, v.iter().map(|b| [b.e0, b.e1, b.n, b.sum, b.lo, b.hi]).collect())).collect();
        let tmp = d.join("history.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(&h)?)?;
        std::fs::rename(tmp, d.join("history.json"))?;
        Ok(())
    }

    /// Seed the plot history from a parent run's `history.json`, keeping
    /// only samples at or before `upto` (the epoch being resumed from).
    pub fn load_history(&mut self, path: &Path, upto: u64) -> Result<()> {
        let h: BTreeMap<String, Vec<[f64; 6]>> = serde_json::from_slice(&std::fs::read(path)?)?;
        for (k, v) in h {
            let mut b: Vec<Band> = v.into_iter().filter(|x| x[1] <= upto as f64).map(|x| Band { e0: x[0], e1: x[1], n: x[2], sum: x[3], lo: x[4], hi: x[5] }).collect();
            while b.len() > self.max_series {
                b = b.chunks(2).map(Band::merge).collect();
            }
            self.series.insert(k.clone(), b.iter().map(|b| [b.mid(), b.mean()]).collect());
            self.bands.insert(k, b);
        }
        Ok(())
    }

    pub fn finish(&mut self) -> Result<Summary> {
        self.save_history()?;
        for t in [&mut self.metrics, &mut self.chir, &mut self.species_t, &mut self.lineage, &mut self.opcodes, &mut self.ecology, &mut self.assay, &mut self.life, &mut self.days, &mut self.clusters_t] {
            t.close()?;
        }
        if let Some(d) = &self.dir {
            std::fs::write(d.join("summary.json"), serde_json::to_string_pretty(&self.summary)?)?;
        }
        Ok(self.summary.clone())
    }
}

/// True when the recorder needs cell state after `epoch` epochs.
pub fn needs_state(c: &primordia_core::config::TelemetryCfg, epoch: u64) -> bool {
    [c.metrics_every, c.opcode_every, c.chirality_every, c.entropy_every, c.assay_every, c.census_every, c.ecology_every, c.cluster_every]
        .iter()
        .any(|&k| k > 0 && epoch % k == 0)
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
