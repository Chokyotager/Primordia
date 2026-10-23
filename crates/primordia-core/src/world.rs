//! The epoch pipeline (§11.1) with a pluggable execution backend.

use crate::config::Config;
use crate::events::{self, Event};
use crate::exec;
use crate::params::Params;
use crate::phases;
use crate::state::{State, Tally};
use anyhow::Result;

/// Phase 6 implementation (CPU golden model or GPU).
pub trait ExecBackend: Send {
    fn exec(&mut self, st: &mut State, p: &Params);
    fn name(&self) -> String;
    /// Called when Params change (live config edits).
    fn reconfigure(&mut self, _p: &Params) {}
    /// Accumulated (upload, kernels, download) seconds, if the backend tracks it.
    fn timing(&self) -> Option<[f64; 3]> {
        None
    }
    /// True when the backend keeps the whole state resident and runs every
    /// phase itself. Epochs are then submitted asynchronously
    /// (`submit_resident`), their tallies arrive later (`poll_resident`) and
    /// the `State` arrays are only valid after `download`.
    fn resident(&self) -> bool {
        false
    }
    fn submit_resident(&mut self, _epoch: u64, _mirrored: bool, _p: &Params, _vents: &[u32], _map: Option<&[u32]>) {
        unimplemented!("backend is not resident")
    }
    /// Results of finished epochs, oldest first. `wait`: block until every
    /// submitted epoch has finished.
    fn poll_resident(&mut self, _wait: bool) -> Vec<EpochResult> {
        vec![]
    }
    /// Queue a copy of the full state as of the last submitted epoch.
    fn request_copy(&mut self, _epoch: u64) {}
    /// Finished state copies, oldest first.
    fn poll_copies(&mut self, _wait: bool) -> Vec<StateCopy> {
        vec![]
    }
    fn download(&mut self, _st: &mut State) {}
    fn upload(&mut self, _st: &State) {}
}

/// Results of one resident epoch: the 64-bit tally slots (see
/// `wgsl::ts`) and the births of this epoch.
pub struct EpochResult {
    pub epoch: u64,
    pub tally: Vec<u64>,
    pub births: Vec<Birth>,
}

/// A copy of the GPU state buffers taken after epoch `epoch - 1` finished.
pub struct StateCopy {
    pub epoch: u64,
    pub tapes: Vec<u8>,
    pub hdr: Vec<u32>,
    pub fields: Vec<u32>,
    pub site: Vec<u32>,
    pub light: Vec<u32>,
}

/// One birth: epoch, parent lineage, child lineage, cell, depth, FNV-1a-32 of the tape.
pub type Birth = [u32; 6];

pub fn fnv32(t: &[u8]) -> u32 {
    let mut h = 0x811c_9dc5u32;
    for &b in t {
        h = (h ^ b as u32).wrapping_mul(0x0100_0193);
    }
    h
}

pub struct CpuExec;
impl ExecBackend for CpuExec {
    fn exec(&mut self, st: &mut State, p: &Params) {
        exec::exec_phase_cpu(st, p);
    }
    fn name(&self) -> String {
        "cpu".into()
    }
}

pub fn config_hash(cfg: &Config) -> u64 {
    let s = cfg.to_toml();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub struct Sim {
    pub cfg: Config,
    pub p: Params,
    pub st: State,
    /// Full event log (applied and pending), sorted by epoch.
    pub events: Vec<Event>,
    pub backend: Box<dyn ExecBackend>,
    pub map_image: Option<Vec<u32>>,
    /// Errors from events that were rejected (kept for the viewer / log).
    pub rejected: Vec<String>,
    /// Births not yet taken by `take_births` (resident: of finished epochs).
    pub births: Vec<Birth>,
    /// Resident backends: CPU arrays are older than the GPU state.
    pub cpu_stale: bool,
    /// Resident backends: CPU arrays were modified and must be uploaded.
    pub gpu_stale: bool,
    /// Resident backends: CPU-side ledger contributions of submitted epochs
    /// whose GPU results have not arrived yet.
    pending: std::collections::VecDeque<(u64, Tally)>,
    /// Vents at each requested snapshot epoch.
    snap_vents: std::collections::BTreeMap<u64, (Vec<crate::state::Vent>, u32)>,
    /// Finished snapshots not yet taken.
    snaps: Vec<State>,
    /// Births / deaths / body moves of finished epochs (for snapshots).
    epoch_stats: std::collections::BTreeMap<u64, (u64, u64, u64)>,
    /// Number of epochs whose labelling did not converge on the GPU (must stay 0).
    pub faults: u64,
}

impl Sim {
    pub fn new(cfg: Config, backend: Box<dyn ExecBackend>) -> Result<Sim> {
        let p = Params::new(&cfg);
        let st = State::init(&cfg, &p);
        Self::from_state(cfg, p, st, backend)
    }
    pub fn from_state(cfg: Config, p: Params, st: State, mut backend: Box<dyn ExecBackend>) -> Result<Sim> {
        backend.reconfigure(&p);
        let map_image = load_map_image(&cfg, &p)?;
        let resident = backend.resident();
        Ok(Sim {
            cfg,
            p,
            st,
            events: vec![],
            backend,
            map_image,
            rejected: vec![],
            births: vec![],
            cpu_stale: false,
            gpu_stale: resident,
            pending: Default::default(),
            snap_vents: Default::default(),
            snaps: vec![],
            epoch_stats: Default::default(),
            faults: 0,
        })
    }
    pub fn epoch(&self) -> u64 {
        self.st.epoch
    }

    pub fn push_event(&mut self, ev: Event) {
        let pos = self.events.partition_point(|e| e.epoch <= ev.epoch);
        self.events.insert(pos, ev);
    }

    /// Make the CPU copy of the state current (resident backends only):
    /// waits for every submitted epoch and downloads the state.
    pub fn sync(&mut self) {
        if self.cpu_stale {
            self.collect(true);
            self.backend.download(&mut self.st);
            self.cpu_stale = false;
        }
    }
    /// Births since the last call (resident: of finished epochs).
    pub fn take_births(&mut self) -> Vec<Birth> {
        std::mem::take(&mut self.births)
    }
    /// Ask for a copy of the state as of now (after the last stepped epoch).
    /// Resident backends copy on the GPU asynchronously; others copy at once.
    pub fn request_snapshot(&mut self) {
        let e = self.st.epoch;
        if self.backend.resident() && self.cpu_stale {
            self.snap_vents.insert(e, (self.st.vents.clone(), self.st.next_vent));
            self.backend.request_copy(e);
        } else {
            let mut s = self.st.clone();
            let (b, d, m) = (self.st.tally.births, self.st.tally.deaths, self.st.tally.body_moves);
            s.tally = Default::default();
            (s.tally.births, s.tally.deaths, s.tally.body_moves) = (b, d, m);
            self.snaps.push(s);
        }
    }
    /// Finished snapshots, oldest first. `wait`: block for all requested ones.
    pub fn take_snapshots(&mut self, wait: bool) -> Vec<State> {
        if self.backend.resident() {
            // results first, so the ledger is current up to the snapshot epochs
            self.collect(wait);
            for c in self.backend.poll_copies(wait) {
                let mut s = State {
                    n: self.st.n,
                    lmax: self.st.lmax,
                    nf: self.st.nf,
                    epoch: c.epoch,
                    mirrored: self.st.mirrored,
                    tapes: c.tapes,
                    hdr: c.hdr,
                    fields: c.fields,
                    site: c.site,
                    vents: vec![],
                    next_vent: 0,
                    tally: Default::default(),
                    ledger: self.st.ledger.clone(),
                    light: c.light,
                };
                if let Some((v, nv)) = self.snap_vents.remove(&c.epoch) {
                    s.vents = v;
                    s.next_vent = nv;
                }
                if let Some(&(b, d, m)) = c.epoch.checked_sub(1).and_then(|e| self.epoch_stats.get(&e)) {
                    (s.tally.births, s.tally.deaths, s.tally.body_moves) = (b, d, m);
                }
                self.snaps.push(s);
            }
        }
        std::mem::take(&mut self.snaps)
    }

    /// Merge the results of finished resident epochs into the ledger.
    fn collect(&mut self, wait: bool) {
        use crate::wgsl::ts;
        if !self.backend.resident() {
            return;
        }
        for r in self.backend.poll_resident(wait) {
            let (e, cpu) = self.pending.pop_front().expect("result without a submitted epoch");
            assert_eq!(e, r.epoch, "resident results out of order");
            let t = |s: usize| r.tally[s] as i64;
            let p = &self.p;
            let tl = &mut self.st.tally;
            tl.energy_source += cpu.energy_source + t(ts::SUN);
            tl.atom_source += cpu.atom_source;
            tl.dissipation += t(ts::DISS);
            tl.energy_sink += t(ts::E_SINK);
            tl.spending += t(ts::UPKEEP) + t(ts::SPENT);
            tl.reaction_net += t(ts::RXN_GAIN) - t(ts::RXN_USED);
            tl.atom_sink += t(ts::ATOM_SINK);
            for cl in 0..3 {
                tl.matter_sink[cl] += t(ts::M_SINK + cl);
            }
            for h in 0..2 {
                tl.matter_source[h] += t(ts::RAC_SRC + h);
                tl.matter_sink[h] += t(ts::RAC_SINK + h);
            }
            tl.births = r.tally[ts::BIRTHS];
            tl.deaths = r.tally[ts::DEATHS];
            tl.body_moves = r.tally[ts::BODY_MOVES];
            self.epoch_stats.insert(e, (tl.births, tl.deaths, tl.body_moves));
            while self.epoch_stats.len() > 64 {
                let k = *self.epoch_stats.keys().next().unwrap();
                self.epoch_stats.remove(&k);
            }
            if r.tally[ts::FAULT] != 0 {
                self.faults += 1;
                self.rejected.push(format!("epoch {e}: GPU body labelling did not converge ({} cells); body moves skipped", r.tally[ts::FAULT]));
            }
            if r.tally[ts::BIRTH_COUNT] as usize > r.births.len() {
                self.rejected.push(format!("epoch {e}: {} births, only {} logged", r.tally[ts::BIRTH_COUNT], r.births.len()));
            }
            self.births.extend(r.births);
            let le = self.cfg.telemetry.ledger_every.max(1);
            if (e + 1) % le == 0 {
                let m = if p.matter_on { [t(ts::L_MATTER), t(ts::L_MATTER + 1), t(ts::L_MATTER + 2)] } else { [0; 3] };
                let a = if p.chem_on { t(ts::L_ATOMS) } else { 0 };
                phases::ledger_apply(&mut self.st, t(ts::L_ENERGY), m, a);
                self.reset_tally();
            }
        }
    }

    /// Declare that the CPU copy was modified (it is uploaded before the next epoch).
    pub fn touch(&mut self) {
        if self.backend.resident() {
            self.gpu_stale = true;
        }
    }

    fn apply_events(&mut self) {
        let e = self.st.epoch;
        let due: Vec<Event> = self.events.iter().filter(|x| x.epoch == e).cloned().collect();
        if !due.is_empty() {
            self.sync();
            self.touch();
            self.st.tally.births = 0;
            self.st.tally.deaths = 0;
            self.st.tally.body_moves = 0;
        }
        for ev in due {
            if ev.kind == "set" {
                if let Err(err) = self.apply_set(&ev) {
                    self.rejected.push(format!("epoch {e}: {err:#}"));
                }
            } else if let Err(err) = events::apply(&mut self.st, &self.p, &ev) {
                self.rejected.push(format!("epoch {e}: {} rejected: {err:#}", ev.kind));
            }
        }
    }

    /// Phases 1-5 on the CPU golden model.
    pub fn begin_epoch(&mut self) {
        self.sync();
        self.st.tally.births = 0;
        self.st.tally.deaths = 0;
        self.st.tally.body_moves = 0;
        self.apply_events();
        let p = &self.p;
        // 2-5
        phases::sources(&mut self.st, p);
        phases::transport(&mut self.st, p);
        phases::chemistry(&mut self.st, p);
        phases::absorb_upkeep(&mut self.st, p);
    }

    /// Phases 7-9.
    pub fn end_epoch(&mut self) {
        let p = &self.p;
        phases::collect_site_counters(&mut self.st);
        let before = self.births.len();
        for c in 0..self.st.n {
            let h = &self.st.hdr[c * crate::params::HDR..(c + 1) * crate::params::HDR];
            if h[crate::params::H_FLAGS] & crate::params::FL_NEWBORN != 0 {
                use crate::params::*;
                self.births.push([self.st.epoch as u32, h[H_PARENT], h[H_LINEAGE], c as u32, h[H_DEPTH], fnv32(self.st.tape(c))]);
            }
        }
        self.st.tally.births = (self.births.len() - before) as u64;
        phases::body_move(&mut self.st, p);
        let map = phases::mutation_map(&self.st, p, self.map_image.as_ref());
        phases::mutate_decay(&mut self.st, p, map.as_ref());
        let le = self.cfg.telemetry.ledger_every.max(1);
        if (self.st.epoch + 1) % le == 0 {
            phases::ledger_check(&mut self.st, p);
            self.reset_tally();
        }
        self.st.epoch += 1;
    }

    fn reset_tally(&mut self) {
        let keep = (self.st.tally.births, self.st.tally.deaths, self.st.tally.body_moves);
        self.st.tally = Default::default();
        (self.st.tally.births, self.st.tally.deaths, self.st.tally.body_moves) = keep;
    }

    pub fn step(&mut self) {
        if self.backend.resident() {
            self.step_resident();
            return;
        }
        self.begin_epoch();
        self.backend.exec(&mut self.st, &self.p);
        self.end_epoch();
    }

    /// One epoch with the whole state on the GPU. Only submits the work:
    /// results are merged as they arrive (see `collect`).
    fn step_resident(&mut self) {
        self.apply_events();
        if self.gpu_stale {
            self.collect(true);
            self.backend.upload(&self.st);
            self.gpu_stale = false;
        }
        let p = &self.p;
        // vents: inject at the current positions (GPU), then age / drift / spawn (CPU)
        let vents: Vec<u32> = if p.vents_on { self.st.vents.iter().map(|v| v.pos).collect() } else { vec![] };
        let nv = vents.len() as i64;
        // CPU-side ledger contributions of this epoch (vents; interventions were
        // added to st.tally directly after a full sync)
        let mut cpu = Tally::default();
        if p.vents_on {
            if !p.chem_only {
                cpu.energy_source += nv * p.vent_cfg.rate as i64;
            }
            if let Some(x4) = p.x4_species {
                cpu.atom_source += nv * 2 * p.vent_cfg.x4_rate as i64 * p.mass[x4] as i64;
            }
        }
        phases::vent_update(&mut self.st, p);
        let map = phases::mutation_map(&self.st, p, self.map_image.as_ref());
        self.backend.submit_resident(self.st.epoch, self.st.mirrored, p, &vents, map.as_deref());
        self.pending.push_back((self.st.epoch, cpu));
        self.st.epoch += 1;
        self.cpu_stale = true;
        self.collect(false);
    }

    /// Step with the execution phase split into its sub-steps on the CPU
    /// golden model; `on_substep` sees the state after each one.
    pub fn step_substeps(&mut self, mut on_substep: impl FnMut(&State, u32)) {
        self.touch();
        self.begin_epoch();
        let p = &self.p;
        if p.geom.dims == 0 {
            exec::exec_phase_cpu(&mut self.st, p);
            on_substep(&self.st, 0);
        } else {
            for (i, col) in exec::color_order(p, self.st.epoch, self.st.mirrored).into_iter().enumerate() {
                let snap = exec::flag_snapshot(&self.st);
                for k in 0..p.geom.cells_per_color() {
                    let c = p.geom.color_cell(col, k);
                    exec::exec_slice(&mut self.st, p, &snap, c, None, None);
                }
                on_substep(&self.st, i as u32);
            }
        }
        self.end_epoch();
    }

    fn apply_set(&mut self, ev: &Event) -> Result<()> {
        let key = ev.params.get("key").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("set needs key"))?;
        let val = ev.params.get("value").cloned().unwrap_or(serde_json::Value::Null);
        let tv: toml::Value = json_to_toml(&val)?;
        let new = self.cfg.with_override(key, tv)?;
        let np = Params::new(&new);
        anyhow::ensure!(np.nf == self.p.nf && np.lmax == self.p.lmax && np.geom.n == self.p.geom.n, "edit would change the state layout");
        self.cfg = new;
        self.p = np;
        self.backend.reconfigure(&self.p);
        self.touch();
        Ok(())
    }
}

pub fn json_to_toml(v: &serde_json::Value) -> Result<toml::Value> {
    Ok(match v {
        serde_json::Value::Bool(b) => toml::Value::Boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                toml::Value::Integer(i)
            } else {
                toml::Value::Float(n.as_f64().unwrap())
            }
        }
        serde_json::Value::String(s) => toml::Value::String(s.clone()),
        serde_json::Value::Array(a) => toml::Value::Array(a.iter().map(json_to_toml).collect::<Result<_>>()?),
        serde_json::Value::Object(o) => {
            let mut t = toml::map::Map::new();
            for (k, v) in o {
                t.insert(k.clone(), json_to_toml(v)?);
            }
            toml::Value::Table(t)
        }
        serde_json::Value::Null => anyhow::bail!("null value"),
    })
}

fn load_map_image(cfg: &Config, p: &Params) -> Result<Option<Vec<u32>>> {
    if p.map_kind != 2 {
        return Ok(None);
    }
    let path = cfg.mutation.map_image.as_ref().ok_or_else(|| anyhow::anyhow!("mutation.map = image needs map_image"))?;
    let img = image::open(path)?.to_luma8();
    let g = &p.geom;
    anyhow::ensure!(img.width() == g.size[0] && img.height() == g.size[1], "map image must be lattice-sized");
    let mut m = vec![0u32; g.n];
    for c in 0..g.n {
        let q = g.coords(c);
        let v = img.get_pixel(q[0], q[1])[0] as f64 / 255.0;
        m[c] = crate::fixed::to_q16(1.0 + cfg.mutation.map_a * v);
    }
    Ok(Some(m))
}
