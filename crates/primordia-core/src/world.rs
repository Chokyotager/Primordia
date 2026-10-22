//! The epoch pipeline (§11.1) with a pluggable execution backend.

use crate::config::Config;
use crate::events::{self, Event};
use crate::exec;
use crate::params::Params;
use crate::phases;
use crate::state::State;
use anyhow::Result;

/// Phase 6 implementation (CPU golden model or GPU).
pub trait ExecBackend: Send {
    fn exec(&mut self, st: &mut State, p: &Params);
    fn name(&self) -> String;
    /// Called when Params change (live config edits).
    fn reconfigure(&mut self, _p: &Params) {}
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
        Ok(Sim { cfg, p, st, events: vec![], backend, map_image, rejected: vec![] })
    }
    pub fn epoch(&self) -> u64 {
        self.st.epoch
    }

    pub fn push_event(&mut self, ev: Event) {
        let pos = self.events.partition_point(|e| e.epoch <= ev.epoch);
        self.events.insert(pos, ev);
    }

    /// Phases 1-5.
    pub fn begin_epoch(&mut self) {
        let e = self.st.epoch;
        self.st.tally.births = 0;
        self.st.tally.deaths = 0;
        self.st.tally.body_moves = 0;
        // 1. interventions
        let due: Vec<Event> = self.events.iter().filter(|x| x.epoch == e).cloned().collect();
        for ev in due {
            if ev.kind == "set" {
                if let Err(err) = self.apply_set(&ev) {
                    self.rejected.push(format!("epoch {e}: {err:#}"));
                }
            } else if let Err(err) = events::apply(&mut self.st, &self.p, &ev) {
                self.rejected.push(format!("epoch {e}: {} rejected: {err:#}", ev.kind));
            }
        }
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
        self.st.tally.births = self.st.hdr.chunks(crate::params::HDR).filter(|h| h[crate::params::H_FLAGS] & crate::params::FL_NEWBORN != 0).count() as u64;
        phases::body_move(&mut self.st, p);
        let map = phases::mutation_map(&self.st, p, self.map_image.as_ref());
        phases::mutate_decay(&mut self.st, p, map.as_ref());
        let le = self.cfg.telemetry.ledger_every.max(1);
        if (self.st.epoch + 1) % le == 0 {
            phases::ledger_check(&mut self.st, p);
            let keep = (self.st.tally.births, self.st.tally.deaths, self.st.tally.body_moves);
            self.st.tally = Default::default();
            (self.st.tally.births, self.st.tally.deaths, self.st.tally.body_moves) = keep;
        }
        self.st.epoch += 1;
    }

    pub fn step(&mut self) {
        self.begin_epoch();
        self.backend.exec(&mut self.st, &self.p);
        self.end_epoch();
    }

    /// Step with the execution phase split into its sub-steps on the CPU
    /// golden model; `on_substep` sees the state after each one.
    pub fn step_substeps(&mut self, mut on_substep: impl FnMut(&State, u32)) {
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
