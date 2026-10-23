//! Native egui viewer (§14). Three threads: the stepper keeps the GPU fed
//! and applies interventions, the observer (`observer.rs`) records telemetry
//! and draws frames from asynchronous snapshots, and the UI shows them: a
//! colony map with an atlas for navigation, a dashboard, a chronicle of
//! notable events, trend plots and an inspector with a CPU debugger.
//! Optionally publishes a read-only feed for the three.js viewer (`web`).

pub mod observer;
mod render;
pub mod web;

use anyhow::Result;
use eframe::egui;
use egui::{Color32, RichText};
use observer::{Msg, Observer, Shared, Sink, Want};
use primordia_core::events::Event;
use primordia_core::exec::{TraceStep, exec_slice, flag_snapshot};
use primordia_core::isa;
use primordia_core::params::*;
use primordia_core::{Params, Sim, phases};
use primordia_telemetry::{Recorder, Summary};
pub use render::{Colony, Mode, ViewSettings};
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Default, Debug)]
pub struct CellInfo {
    pub occ: bool,
    pub halted: bool,
    pub family: u32,
    pub copier: bool,
    pub bonded: bool,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct CellBrief {
    pub e: u32,
    pub len: u16,
    pub f: u32,
    pub t: u32,
    pub light: u32,
    pub info: CellInfo,
}

pub struct Image {
    pub label: String,
    pub w: usize,
    pub h: usize,
    pub rgba: Vec<u8>,
    /// the lattice axes the image spans, and the fixed coordinate of the slice
    pub axes: (usize, usize),
    pub fixed: [u32; 3],
}

pub struct Frame {
    pub epoch: u64,
    pub images: Vec<Image>,
    pub bonds: Vec<u8>,
    pub vents: Vec<[u32; 3]>,
    pub domains: Vec<i8>,
    pub dnb: [usize; 3],
    pub size: [u32; 3],
    pub dims: u32,
    /// per-cell summary for hover descriptions (2D only)
    pub cells: Vec<CellBrief>,
    pub colonies: Vec<Colony>,
    /// colony index + 1 per cell, 0 for none (2D only)
    pub colony_of: Vec<u32>,
}

#[derive(Default, Clone)]
pub struct Inspect {
    pub cell: usize,
    pub text: String,
    pub tape: Vec<u8>,
    pub cat_eta: Vec<(usize, f64)>,
    pub trace: Vec<TraceStep>,
}

#[derive(Clone)]
pub struct Note {
    pub epoch: u64,
    pub icon: &'static str,
    pub text: String,
}

/// Status published by the stepper thread.
#[derive(Default)]
struct Status {
    epoch: u64,
    eps: f64,
    running: bool,
    backend: String,
    dims: u32,
    size: [u32; 3],
    k_species: Vec<String>,
    n_signals: usize,
    inspect: Option<Inspect>,
    sub_frames: Vec<Frame>,
    web: Option<String>,
}

enum Cmd {
    Run(bool),
    Step(u64),
    SubSteps,
    View(ViewSettings),
    Event(Event),
    Inspect(usize),
    Snapshot,
    Branch,
    Quit,
}

pub struct ViewerOptions {
    pub autorun: bool,
    /// Address for the read-only web viewer, e.g. "0.0.0.0:8765".
    pub web: Option<String>,
}

pub fn run_viewer(sim: Sim, rec: Recorder, sink: Option<Box<dyn Sink>>, opts: ViewerOptions) -> Result<()> {
    let status = Arc::new(Mutex::new(Status::default()));
    let feed = match &opts.web {
        Some(addr) => {
            let f = web::Feed::new();
            web::serve(addr, f.clone())?;
            status.lock().unwrap().web = Some(addr.clone());
            Some(f)
        }
        None => None,
    };
    let view = ViewSettings::default(&sim.p);
    let obs = Observer::spawn(sim.cfg.clone(), sim.p.clone(), rec, sink, feed, view.clone(), true);
    let shared = obs.shared.clone();
    let (tx, rx) = channel();
    let s2 = status.clone();
    let autorun = opts.autorun;
    let worker = std::thread::spawn(move || stepper(sim, obs, rx, s2, autorun));
    let nopts = eframe::NativeOptions { viewport: egui::ViewportBuilder::default().with_inner_size([1680.0, 1020.0]).with_title("Primordia"), ..Default::default() };
    let app = App::new(tx.clone(), status, shared, view);
    let r = eframe::run_native("Primordia", nopts, Box::new(|cc| {
        style(&cc.egui_ctx);
        Ok(Box::new(app))
    }));
    let _ = tx.send(Cmd::Quit);
    let _ = worker.join();
    r.map_err(|e| anyhow::anyhow!("viewer: {e}"))
}

fn style(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();
    v.panel_fill = Color32::from_rgb(16, 19, 27);
    v.window_fill = Color32::from_rgb(20, 24, 34);
    v.extreme_bg_color = Color32::from_rgb(8, 10, 15);
    v.faint_bg_color = Color32::from_rgb(24, 29, 40);
    v.selection.bg_fill = Color32::from_rgb(52, 120, 110);
    v.hyperlink_color = Color32::from_rgb(110, 210, 190);
    v.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, Color32::from_rgb(40, 46, 60));
    ctx.set_visuals(v);
    ctx.global_style_mut(|s| {
        s.spacing.item_spacing = egui::vec2(8.0, 6.0);
        s.spacing.button_padding = egui::vec2(10.0, 5.0);
    });
}

// ------------------------------------------------------------------ chronicle

/// Watches the run and writes plain-language notes about notable events.
pub struct Chronicler {
    first_repl: bool,
    hoe_over: bool,
    max_body: usize,
    vents: Vec<(u32, u32)>,
    pop_hist: Vec<(u64, u64)>,
    hand: i32,
    families: usize,
    last_boom: u64,
}

impl Default for Chronicler {
    fn default() -> Self {
        Self::new()
    }
}

impl Chronicler {
    pub fn new() -> Self {
        Chronicler { first_repl: false, hoe_over: false, max_body: 1, vents: vec![], pop_hist: vec![], hand: 0, families: 0, last_boom: 0 }
    }
    pub fn observe(&mut self, sim: &Sim, s: &Summary, out: &mut Vec<Note>) {
        self.observe_state(&sim.st, &sim.p, s, out)
    }

    pub fn observe_state(&mut self, st: &primordia_core::State, p: &Params, s: &Summary, out: &mut Vec<Note>) {
        let e = st.epoch;
        let mut note = |icon: &'static str, text: String| out.push(Note { epoch: e, icon, text });
        if !self.first_repl && s.replicators > 0 {
            self.first_repl = true;
            note("🧬", format!("Self-copying programs found: {} of the 100 most common programs copy themselves. Smallest working core: {}", s.replicators, s.best_replicator.clone().unwrap_or_default()));
        }
        if !self.hoe_over && s.hoe > 1.0 {
            self.hoe_over = true;
            note("✨", format!("The soup is no longer random: complexity (high-order entropy) rose to {:.2}", s.hoe));
        }
        if s.max_body >= 8 && s.max_body >= 2 * self.max_body {
            self.max_body = s.max_body;
            note("🔗", format!("Cells have bonded into a body of {} cells — the largest so far", s.max_body));
        }
        let g = &p.geom;
        let now: Vec<(u32, u32)> = st.vents.iter().map(|v| (v.id, v.pos)).collect();
        for &(id, pos) in &now {
            if !self.vents.iter().any(|v| v.0 == id) && e > 1 {
                note("🌋", format!("A new vent erupted at {:?}", &g.coords(pos as usize)[..g.dims.max(1) as usize]));
            }
        }
        for &(id, pos) in &self.vents {
            if !now.iter().any(|v| v.0 == id) {
                note("🪨", format!("The vent at {:?} went quiet", &g.coords(pos as usize)[..g.dims.max(1) as usize]));
            }
        }
        self.vents = now;
        if s.occupied > 0 {
            self.pop_hist.push((e, s.occupied));
            if self.pop_hist.len() > 64 {
                self.pop_hist.remove(0);
            }
            if let Some(&(e0, p0)) = self.pop_hist.iter().find(|x| e - x.0 <= 300) {
                if e > self.last_boom + 300 && p0 > 0 && e > e0 {
                    let r = s.occupied as f64 / p0 as f64;
                    if r > 1.4 {
                        note("📈", format!("Population boom: {} → {} cells in {} epochs", p0, s.occupied, e - e0));
                        self.last_boom = e;
                    } else if r < 0.7 {
                        note("📉", format!("Population crash: {} → {} cells in {} epochs", p0, s.occupied, e - e0));
                        self.last_boom = e;
                    }
                }
            }
        }
        let hand = if s.ee_code > 0.3 { 1 } else if s.ee_code < -0.3 { -1 } else { 0 };
        if hand != self.hand {
            match hand {
                1 => note("🔴", format!("Code is turning left-handed (L): enantiomeric excess {:+.2}", s.ee_code)),
                -1 => note("🔵", format!("Code is turning right-handed (D): enantiomeric excess {:+.2}", s.ee_code)),
                _ => note("⚖", format!("Handedness is balanced again ({:+.2})", s.ee_code)),
            }
            self.hand = hand;
        }
        if s.species > 0 && (self.families == 0 || s.species > self.families * 2 || s.species * 2 < self.families) {
            if self.families != 0 {
                note("🌿", format!("Diversity changed: {} → {} families of similar programs", self.families, s.species));
            }
            self.families = s.species;
        }
    }
}

// ------------------------------------------------------------------ stepper

/// Drives the simulation as fast as the GPU allows; everything that only
/// looks at the world happens on the observer thread.
fn stepper(mut sim: Sim, mut obs: Observer, rx: Receiver<Cmd>, status: Arc<Mutex<Status>>, autorun: bool) {
    let mut running = autorun;
    let mut pending_steps = 0u64;
    let mut lineage: HashMap<u32, u32> = HashMap::new();
    let mut rate_t = Instant::now();
    let mut rate_e = sim.st.epoch;
    let mut eps = 0.0;
    let mut dirty = true;
    let backend = sim.backend.name();
    let mut save: Option<bool> = None;
    let mut wants: BTreeMap<u64, Want> = BTreeMap::new();
    {
        let mut s = status.lock().unwrap();
        s.backend = backend.clone();
        s.dims = sim.p.geom.dims;
        s.size = sim.p.geom.size;
        s.k_species = if sim.p.chem_on { sim.cfg.chemistry.species.clone() } else { vec![] };
        s.n_signals = sim.p.s;
    }
    loop {
        let mut sub = false;
        while let Ok(c) = rx.try_recv() {
            match c {
                Cmd::Quit => {
                    // save the final state so the run can be resumed exactly here
                    if !wants.contains_key(&sim.st.epoch) {
                        sim.request_snapshot();
                        wants.insert(sim.st.epoch, Want { save: true, telemetry: true, ..Default::default() });
                    }
                    for st in sim.take_snapshots(true) {
                        let w = wants.remove(&st.epoch).unwrap_or_default();
                        obs.deliver(st, w, eps, backend.clone());
                    }
                    let b = sim.take_births();
                    obs.send(Msg::Births(b));
                    let _ = obs.finish();
                    return;
                }
                Cmd::Run(r) => running = r,
                Cmd::Step(n) => pending_steps += n,
                Cmd::SubSteps => sub = true,
                Cmd::View(v) => {
                    obs.send(Msg::View(v));
                    dirty = true;
                }
                Cmd::Event(e) => {
                    obs.send(Msg::Event(e.clone()));
                    sim.push_event(e);
                }
                Cmd::Inspect(c) => {
                    sim.sync();
                    let ins = inspect(&sim, c, &lineage);
                    status.lock().unwrap().inspect = Some(ins);
                }
                Cmd::Snapshot => save = Some(false),
                Cmd::Branch => save = Some(true),
            }
        }
        let do_step = running || pending_steps > 0 || sub;
        if do_step {
            if sub {
                let mut frames = vec![];
                let v = ViewSettings::default(&sim.p);
                let pp = sim.p.clone();
                let mut ctx = render::RenderCtx::default();
                sim.step_substeps(|st, i| {
                    let mut f = render::render(st, &pp, &v, &[], &mut ctx);
                    f.images.iter_mut().for_each(|im| im.label = format!("{} — sub-step {}", im.label, i));
                    frames.push(f);
                });
                status.lock().unwrap().sub_frames = frames;
            } else {
                sim.step();
            }
            if pending_steps > 0 {
                pending_steps -= 1;
            }
            dirty = true;
            let b = sim.take_births();
            for x in &b {
                lineage.insert(x[2], x[1]);
            }
            if lineage.len() > 2_000_000 {
                lineage.clear();
            }
            if !b.is_empty() {
                obs.send(Msg::Births(b));
            }
            if rate_t.elapsed() > Duration::from_millis(500) {
                eps = (sim.st.epoch - rate_e) as f64 / rate_t.elapsed().as_secs_f64();
                rate_t = Instant::now();
                rate_e = sim.st.epoch;
            }
        }
        // what the observer needs from the state after this epoch
        let mut w = obs.want(sim.st.epoch, dirty);
        if !do_step {
            // paused: telemetry and run-dir cadences were already served
            w.telemetry = false;
            w.sink = false;
        }
        if let Some(branch) = save.take() {
            w.save = true;
            w.branch = branch;
        }
        if (w.telemetry || w.sink || w.render || w.web || w.save) && !wants.contains_key(&sim.st.epoch) {
            sim.request_snapshot();
            wants.insert(sim.st.epoch, w);
            if w.render {
                dirty = false;
            }
        }
        for st in sim.take_snapshots(false) {
            let w = wants.remove(&st.epoch).unwrap_or_default();
            obs.deliver(st, w, eps, backend.clone());
        }
        for r in sim.rejected.drain(..) {
            obs.send(Msg::Event(Event { epoch: sim.st.epoch, kind: "warning".into(), params: json!(r) }));
        }
        {
            let mut s = status.lock().unwrap();
            s.eps = eps;
            s.epoch = sim.st.epoch;
            s.running = running;
        }
        if !do_step {
            std::thread::sleep(Duration::from_millis(15));
        }
    }
}

/// Current sunlight as a fraction of the configured peak.
pub fn sun_level(sim: &Sim) -> f32 {
    sun_level_at(&sim.p, sim.st.epoch)
}

pub fn sun_level_at(p: &Params, epoch: u64) -> f32 {
    if !p.sun_on || p.sun_i0_q16 == 0 {
        return 0.0;
    }
    let rows = phases::row_lights(p, epoch);
    let avg = rows.iter().map(|&r| r as f64).sum::<f64>() / rows.len().max(1) as f64;
    (avg / p.sun_i0_q16 as f64) as f32
}

fn inspect(sim: &Sim, c: usize, lineage: &HashMap<u32, u32>) -> Inspect {
    let st = &sim.st;
    let p = &sim.p;
    let g = &p.geom;
    let mut t = String::new();
    use std::fmt::Write;
    let q = g.coords(c);
    let _ = writeln!(t, "cell {c}  at {:?}", &q[..g.dims.max(1) as usize]);
    let fl = st.flags(c);
    if fl & FL_OCC == 0 {
        let _ = writeln!(t, "empty");
    } else {
        let o = st.orient(c) as usize;
        let _ = writeln!(
            t,
            "len {}  E {}  {}{}  facing {}  up {}  lineage {:08x}  parent {:08x}  depth {}  born {}",
            st.len(c),
            st.h(c, H_E),
            if fl & FL_HALT != 0 { "HALTED " } else { "" },
            if st.h(c, H_SLEEP) > 0 { format!("sleep {}", st.h(c, H_SLEEP)) } else { String::new() },
            ["+x", "-x", "+y", "-y", "+z", "-z"][p.orient.f[o] as usize],
            ["+x", "-x", "+y", "-y", "+z", "-z"][p.orient.u[o] as usize],
            st.h(c, H_LINEAGE),
            st.h(c, H_PARENT),
            st.h(c, H_DEPTH),
            st.h(c, H_BIRTH)
        );
        let b = st.bonds(c);
        let dirs: Vec<&str> = (0..6).filter(|d| b & (1 << d) != 0).map(|d| ["+x", "-x", "+y", "-y", "+z", "-z"][d]).collect();
        let bodies = phases::components(st, p);
        let lab = bodies.label[c];
        let size = if lab == NONE { 1 } else { bodies.sizes.iter().find(|s| s.0 == lab).map(|s| s.1).unwrap_or(1) };
        let _ = writeln!(t, "bonds [{}]  body {}  size {}", dirs.join(" "), if lab == NONE { "-".into() } else { format!("{lab}") }, size);
        let mut chain = vec![st.h(c, H_LINEAGE)];
        let mut cur = st.h(c, H_PARENT);
        while cur != 0 && chain.len() < 32 {
            chain.push(cur);
            match lineage.get(&cur) {
                Some(&pp) => cur = pp,
                None => break,
            }
        }
        let _ = writeln!(t, "lineage chain: {}", chain.iter().map(|x| format!("{x:08x}")).collect::<Vec<_>>().join(" <- "));
    }
    let _ = writeln!(t, "F {}  T {:.2} K  light {:.1}", st.f(c, F_F), st.f(c, F_T) as f64 / 100.0, st.light[c] as f64 / 65536.0);
    if p.matter_on {
        let _ = writeln!(t, "pools L {} D {} A {}", st.f(c, p.f_pool(0)), st.f(c, p.f_pool(1)), st.f(c, p.f_pool(2)));
    }
    for k in 0..p.k {
        let _ = write!(t, "{} L {} D {}   ", sim.cfg.chemistry.species[k], st.f(c, p.f_chem(k, 0)), st.f(c, p.f_chem(k, 1)));
    }
    if p.k > 0 {
        t.push('\n');
    }
    for j in 0..p.s {
        let _ = write!(t, "sig{} L {} D {}   ", j, st.f(c, p.f_sig(j, 0)), st.f(c, p.f_sig(j, 1)));
    }
    if p.s > 0 {
        t.push('\n');
    }
    let tape = st.tape(c).to_vec();
    let w = p.fold_w as usize;
    let mut cat_eta = vec![];
    for (i, &b) in tape.iter().enumerate() {
        if b == 0x2C || b == 0x2D {
            let h = (b & 1) as usize;
            let (mut ns, mut no) = (0usize, 0usize);
            for k in 1..=w {
                let x = tape[(i + tape.len() * 64 - k) % tape.len()];
                if isa::is_chiral(x) {
                    if (x & 1) as usize == h { ns += 1 } else { no += 1 }
                }
            }
            let emin = p.eta_min_q16 as f64 / 65536.0;
            cat_eta.push((i, emin + (1.0 - emin) * (ns.saturating_sub(no)) as f64 / w as f64));
        }
    }
    // CPU debugger: re-run this cell's next slice on a copy of the world
    let mut trace = vec![];
    if fl & FL_OCC != 0 {
        let mut copy = st.clone();
        copy.hdr[c * HDR + H_FLAGS] &= !(FL_EXEC | FL_HALT);
        copy.hdr[c * HDR + H_SLEEP] = 0;
        let snap = flag_snapshot(&copy);
        let partner = if p.geom.dims == 0 { Some((c + 1) % st.n) } else { None };
        exec_slice(&mut copy, p, &snap, c, partner, Some(&mut trace));
    }
    Inspect { cell: c, text: t, tape, cat_eta, trace }
}

fn card(ui: &mut egui::Ui, title: &str, value: String, sub: String, accent: Color32) {
    egui::Frame::new().fill(Color32::from_rgb(24, 29, 40)).corner_radius(8.0).inner_margin(egui::Margin::symmetric(12, 6)).show(ui, |ui| {
        ui.set_min_width(120.0);
        ui.vertical(|ui| {
            ui.label(RichText::new(title).small().color(Color32::from_rgb(140, 150, 170)));
            ui.label(RichText::new(value).size(20.0).strong().color(accent));
            ui.label(RichText::new(sub).small().color(Color32::from_rgb(120, 130, 150)));
        });
    });
}

fn hand_meter(ui: &mut egui::Ui, ee: f64) {
    egui::Frame::new().fill(Color32::from_rgb(24, 29, 40)).corner_radius(8.0).inner_margin(egui::Margin::symmetric(12, 6)).show(ui, |ui| {
        ui.vertical(|ui| {
            ui.label(RichText::new("Handedness of code").small().color(Color32::from_rgb(140, 150, 170)));
            let (r, p) = ui.allocate_painter(egui::vec2(170.0, 22.0), egui::Sense::hover());
            let rect = r.rect;
            p.rect_filled(rect, 4.0, Color32::from_rgb(40, 44, 56));
            let mid = rect.center().x;
            let w = (ee.clamp(-1.0, 1.0) as f32) * rect.width() / 2.0;
            let bar = if w >= 0.0 { egui::Rect::from_min_max(egui::pos2(mid, rect.top()), egui::pos2(mid + w, rect.bottom())) } else { egui::Rect::from_min_max(egui::pos2(mid + w, rect.top()), egui::pos2(mid, rect.bottom())) };
            p.rect_filled(bar, 4.0, if ee >= 0.0 { Color32::from_rgb(230, 80, 80) } else { Color32::from_rgb(80, 120, 240) });
            p.line_segment([egui::pos2(mid, rect.top()), egui::pos2(mid, rect.bottom())], egui::Stroke::new(1.0, Color32::WHITE));
            p.text(rect.left_center() + egui::vec2(4.0, 0.0), egui::Align2::LEFT_CENTER, "D", egui::FontId::proportional(11.0), Color32::from_rgb(160, 180, 255));
            p.text(rect.right_center() - egui::vec2(4.0, 0.0), egui::Align2::RIGHT_CENTER, "L", egui::FontId::proportional(11.0), Color32::from_rgb(255, 170, 170));
            ui.label(RichText::new(format!("excess {:+.2} ({})", ee, if ee.abs() < 0.1 { "balanced" } else if ee > 0.0 { "left" } else { "right" })).small().color(Color32::from_rgb(120, 130, 150)));
        });
    });
}


/// Plain-language description of a cell for the hover card.
fn describe(cb: &CellBrief, colony: Option<&Colony>) -> (String, String) {
    let temp = cb.t as f32 / 100.0;
    let env = format!("Food here: {}   Temperature: {:.1} K   Sunlight: {:.0}", cb.f, temp, cb.light as f32 / 65536.0);
    if !cb.info.occ {
        return ("Empty space".into(), env);
    }
    let mut s = String::new();
    match colony {
        Some(c) => s += &format!("Member of {} ({} cells)", c.name, c.size),
        None => s += "A loner (not part of a large colony)",
    }
    if cb.info.halted {
        s += " · dormant";
    }
    if cb.info.copier {
        s += " · copier";
    }
    if cb.info.bonded {
        s += " · bonded";
    }
    let health = match cb.e {
        0 => "out of energy",
        1..=99 => "hungry",
        100..=999 => "fed",
        _ => "well fed",
    };
    (s, format!("Energy {} ({health}) · program {} bytes\n{env}", cb.e, cb.len))
}

/// Plain-language summary of a colony.
fn colony_story(c: &Colony, epoch: u64) -> Vec<String> {
    let mut v = vec![];
    v.push(format!("{} cells, about {:.0} cells across, first seen at epoch {} ({} epochs ago).", c.size, 2.0 * c.radius, c.first_seen, epoch.saturating_sub(c.first_seen)));
    v.push(if c.growth > 20.0 {
        format!("Growing fast: +{:.0} cells per 100 epochs.", c.growth)
    } else if c.growth > 1.0 {
        format!("Growing: +{:.0} cells per 100 epochs.", c.growth)
    } else if c.growth < -20.0 {
        format!("Shrinking fast: {:.0} cells per 100 epochs.", c.growth)
    } else if c.growth < -1.0 {
        format!("Shrinking: {:.0} cells per 100 epochs.", c.growth)
    } else {
        "Stable in size.".into()
    });
    v.push(format!("{:.0}% of its cells carry a copy loop; {:.0}% are bonded to neighbours; {:.0}% are dormant.", c.copiers * 100.0, c.bonded * 100.0, c.dormant * 100.0));
    v.push(if c.hand.abs() < 0.15 {
        "Its code uses left- and right-handed instructions about equally.".into()
    } else if c.hand > 0.0 {
        format!("Its code is left-handed (excess {:+.2}).", c.hand)
    } else {
        format!("Its code is right-handed (excess {:+.2}).", c.hand)
    });
    v.push(format!("Programs average {:.0} bytes.", c.mean_len));
    v
}

// ------------------------------------------------------------------ UI

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tool {
    Inspect,
    Paint,
    VentAdd,
    VentRemove,
    Seed,
    Clear,
    Wall,
    MirrorRegion,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Colonies,
    Chronicle,
    Inspector,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Sort {
    Size,
    Growth,
    Age,
}

struct App {
    tx: Sender<Cmd>,
    status: Arc<Mutex<Status>>,
    shared: Arc<Mutex<Shared>>,
    tex: Vec<Option<egui::TextureHandle>>,
    frame_ptr: usize,
    last_frame: Option<Arc<Frame>>,
    view: ViewSettings,
    zoom: f32,
    pan: egui::Vec2,
    /// fly-to target: lattice point and zoom
    fly: Option<([f32; 2], f32)>,
    selected: Option<u32>,
    follow: bool,
    hover_colony: Option<u32>,
    tab: Tab,
    sort: Sort,
    tool: Tool,
    paint_field: String,
    paint_amount: u32,
    paint_species: usize,
    paint_hand: usize,
    radius: f32,
    genome: String,
    mirror_seed: bool,
    seed_energy: u32,
    wall_type: String,
    mirror_a: Option<[i64; 3]>,
    edit_key: String,
    edit_val: String,
    labels: bool,
    show_vents: bool,
    show_bonds: bool,
    show_domains: bool,
    minimap: bool,
    plots: BTreeMap<String, bool>,
    show_plots: bool,
    trace_idx: usize,
    sub_idx: usize,
    show_sub: bool,
}

impl App {
    fn new(tx: Sender<Cmd>, status: Arc<Mutex<Status>>, shared: Arc<Mutex<Shared>>, view: ViewSettings) -> App {
        let mut plots = BTreeMap::new();
        for k in ["population", "high-order entropy", "EE code", "replicators (top 100)"] {
            plots.insert(k.to_string(), true);
        }
        App {
            tx,
            status,
            shared,
            tex: vec![],
            frame_ptr: 0,
            last_frame: None,
            view,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            fly: None,
            selected: None,
            follow: false,
            hover_colony: None,
            tab: Tab::Colonies,
            sort: Sort::Size,
            tool: Tool::Inspect,
            paint_field: "F".into(),
            paint_amount: 500,
            paint_species: 0,
            paint_hand: 0,
            radius: 4.0,
            genome: "36 36 36 36 36 36 36 4d 67 40 2c 12 13 15 c4".into(),
            mirror_seed: false,
            seed_energy: 1000,
            wall_type: "reflect".into(),
            mirror_a: None,
            edit_key: "mutation.background".into(),
            edit_val: "1e-3".into(),
            labels: true,
            show_vents: true,
            show_bonds: true,
            show_domains: false,
            minimap: true,
            plots,
            show_plots: true,
            trace_idx: 0,
            sub_idx: 0,
            show_sub: false,
        }
    }

    fn event(&self, epoch: u64, kind: &str, params: serde_json::Value) {
        let _ = self.tx.send(Cmd::Event(Event { epoch, kind: kind.into(), params }));
    }

    fn fly_to(&mut self, c: &Colony) {
        let z = (60.0 / (c.radius * 2.5 + 8.0)).clamp(1.5, 40.0) * 2.5;
        self.fly = Some(([c.center[0], c.center[1]], z));
        self.selected = Some(c.id);
    }

    fn apply_tool(&mut self, epoch: u64, p: [i64; 3], dims: u32) {
        let center = json!(&p[..dims.max(1) as usize]);
        match self.tool {
            Tool::Inspect => {
                let g = &self.view.params.geom;
                let mut q = [0u32; 3];
                for a in 0..3 {
                    q[a] = p[a].clamp(0, g.size[a] as i64 - 1) as u32;
                }
                let _ = self.tx.send(Cmd::Inspect(g.index(q)));
                self.trace_idx = 0;
                self.tab = Tab::Inspector;
            }
            Tool::Paint => {
                let mut prm = json!({"field": self.paint_field, "center": center, "radius": self.radius, "amount": self.paint_amount});
                prm["species"] = json!(self.paint_species);
                prm["hand"] = json!(self.paint_hand);
                self.event(epoch, "paint", prm);
            }
            Tool::VentAdd => self.event(epoch, "vent_add", json!({"center": center})),
            Tool::VentRemove => self.event(epoch, "vent_remove", json!({"center": center, "radius": self.radius})),
            Tool::Seed => self.event(epoch, "seed", json!({"genome": self.genome, "center": center, "radius": self.radius, "mirror": self.mirror_seed, "energy": self.seed_energy})),
            Tool::Clear => self.event(epoch, "clear", json!({"center": center, "radius": self.radius})),
            Tool::Wall => self.event(epoch, "wall", json!({"center": center, "radius": self.radius, "type": self.wall_type})),
            Tool::MirrorRegion => match self.mirror_a.take() {
                None => self.mirror_a = Some(p),
                Some(a) => {
                    let lo: Vec<i64> = (0..3).map(|i| a[i].min(p[i])).collect();
                    let hi: Vec<i64> = (0..3).map(|i| a[i].max(p[i])).collect();
                    self.event(epoch, "mirror_region", json!({"min": lo, "max": hi}));
                }
            },
        }
    }
}

fn swatch(ui: &mut egui::Ui, col: [u8; 3], size: f32) -> egui::Response {
    let (r, p) = ui.allocate_painter(egui::vec2(size, size), egui::Sense::hover());
    p.rect_filled(r.rect.shrink(1.0), 4.0, Color32::from_rgb(col[0], col[1], col[2]));
    r
}

fn c32(c: [u8; 3]) -> Color32 {
    Color32::from_rgb(c[0], c[1], c[2])
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        ctx.request_repaint_after(Duration::from_millis(33));
        let (epoch, dims, running, eps, backend, species, nsig, inspect, nsub, web) = {
            let s = self.status.lock().unwrap();
            (s.epoch, s.dims, s.running, s.eps, s.backend.clone(), s.k_species.clone(), s.n_signals, s.inspect.clone(), s.sub_frames.len(), s.web.clone())
        };
        let (summary, series, chronicle, top, sun, births, deaths, frame) = {
            let s = self.shared.lock().unwrap();
            (s.summary.clone(), s.series.clone(), s.chronicle.iter().rev().take(200).cloned().collect::<Vec<_>>(), s.top_genomes.clone(), s.sun, s.births, s.deaths, s.frame.clone())
        };
        if self.show_sub && nsub > 0 {
            let s = self.status.lock().unwrap();
            upload(&ctx, &mut self.tex, &s.sub_frames[self.sub_idx.min(nsub - 1)]);
            self.frame_ptr = 0;
        } else if let Some(f) = &frame {
            let ptr = Arc::as_ptr(f) as usize;
            if ptr != self.frame_ptr {
                upload(&ctx, &mut self.tex, f);
                self.frame_ptr = ptr;
            }
            self.last_frame = Some(f.clone());
        }
        let Some(fr) = self.last_frame.clone() else {
            egui::CentralPanel::default().show(ui, |ui| {
                ui.centered_and_justified(|ui| ui.heading("Growing a world…"));
            });
            return;
        };
        let colonies = fr.colonies.clone();

        // keyboard shortcuts
        let (space, home, next, prev) = ui.input(|i| (i.key_pressed(egui::Key::Space), i.key_pressed(egui::Key::Home), i.key_pressed(egui::Key::CloseBracket), i.key_pressed(egui::Key::OpenBracket)));
        if !ctx.egui_wants_keyboard_input() {
            if space {
                let _ = self.tx.send(Cmd::Run(!running));
            }
            if home {
                self.fly = None;
                self.zoom = 1.0;
                self.pan = egui::Vec2::ZERO;
                self.follow = false;
            }
            if (next || prev) && !colonies.is_empty() {
                let i = self.selected.and_then(|id| colonies.iter().position(|c| c.id == id)).map(|i| if next { (i + 1) % colonies.len() } else { (i + colonies.len() - 1) % colonies.len() }).unwrap_or(0);
                let c = colonies[i].clone();
                self.fly_to(&c);
            }
        }

        // ---------------- header: controls + dashboard
        egui::Panel::top("top").show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new("PRIMORDIA").size(22.0).strong().color(Color32::from_rgb(110, 210, 190)));
                ui.separator();
                if ui.button(if running { "⏸  Pause" } else { "▶  Run" }).on_hover_text("Space").clicked() {
                    let _ = self.tx.send(Cmd::Run(!running));
                }
                if ui.button("Step").clicked() {
                    let _ = self.tx.send(Cmd::Step(1));
                }
                if ui.button("Step ×10").clicked() {
                    let _ = self.tx.send(Cmd::Step(10));
                }
                ui.label(RichText::new(format!("epoch {epoch}   ·   {eps:.1} epochs/s")).monospace());
                ui.separator();
                ui.label(RichText::new(if sun > 0.05 { format!("☀ daylight {:.0}%", sun * 100.0) } else { "🌙 night".to_string() }).color(if sun > 0.05 { Color32::from_rgb(255, 210, 110) } else { Color32::from_rgb(150, 160, 220) }));
                if let Some(w) = &web {
                    ui.separator();
                    ui.label(RichText::new(format!("🌐 browser view: http://{w}")).small().color(Color32::from_rgb(110, 210, 190)));
                }
            });
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                card(ui, "Living cells", format!("{}", summary.occupied), format!("+{births} born, −{deaths} died this epoch"), Color32::from_rgb(120, 220, 140));
                card(ui, "Colonies", format!("{}", colonies.len()), "connected groups of relatives".into(), Color32::from_rgb(255, 170, 60));
                let cx = if summary.hoe > 1.0 { "structured: life-like" } else if summary.hoe > 0.3 { "some structure" } else { "random soup" };
                card(ui, "Complexity", format!("{:.2}", summary.hoe), cx.into(), Color32::from_rgb(255, 210, 110));
                card(ui, "Copiers", format!("{} / 100", summary.replicators), "top programs that copy themselves".into(), Color32::from_rgb(110, 200, 255));
                card(ui, "Largest body", format!("{}", summary.max_body), format!("{} bonded bodies", summary.bodies), Color32::from_rgb(200, 170, 255));
                hand_meter(ui, summary.ee_code);
                let ok = summary.ledger_violations == 0;
                card(ui, "Conservation", if ok { "exact".into() } else { format!("{} errors", summary.ledger_violations) }, "energy, matter, atoms".into(), if ok { Color32::from_rgb(120, 220, 140) } else { Color32::RED });
            });
            ui.add_space(2.0);
        });

        // ---------------- left: view, legend, tools
        egui::Panel::left("left").default_size(290.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                let v = &mut self.view;
                let mut changed = false;
                ui.heading("View");
                egui::ComboBox::from_id_salt("mode").width(250.0).selected_text(v.mode.name()).show_ui(ui, |ui| {
                    for m in Mode::all() {
                        changed |= ui.selectable_value(&mut v.mode, m, m.name()).changed();
                    }
                });
                if matches!(v.mode, Mode::Chem | Mode::ChemEe) && !species.is_empty() {
                    egui::ComboBox::from_label("species").selected_text(species[v.chem_k.min(species.len() - 1)].clone()).show_ui(ui, |ui| {
                        for (k, s) in species.iter().enumerate() {
                            changed |= ui.selectable_value(&mut v.chem_k, k, s).changed();
                        }
                    });
                    if v.mode == Mode::Chem {
                        changed |= ui.radio_value(&mut v.chem_h, 0, "L").changed();
                        changed |= ui.radio_value(&mut v.chem_h, 1, "D").changed();
                    }
                }
                if v.mode == Mode::Signals && nsig > 0 {
                    changed |= ui.add(egui::Slider::new(&mut v.sig_j, 0..=nsig - 1).text("signal")).changed();
                }
                if dims == 3 {
                    let size = v.params.geom.size;
                    ui.label("3D slices");
                    for (a, n) in ["x (side view)", "y (side view)", "z (top view)"].iter().enumerate() {
                        changed |= ui.add(egui::Slider::new(&mut v.slice[a], 0..=size[a] - 1).text(*n)).changed();
                    }
                }
                if changed {
                    let _ = self.tx.send(Cmd::View(v.clone()));
                }
                ui.checkbox(&mut self.labels, "Colony names on the map");
                ui.checkbox(&mut self.show_vents, "Vents");
                ui.checkbox(&mut self.minimap, "Minimap when zoomed in");
                ui.checkbox(&mut self.show_bonds, "Bonds when zoomed in");
                ui.checkbox(&mut self.show_domains, "Handedness domains");
                ui.add_space(4.0);
                egui::Frame::new().fill(Color32::from_rgb(22, 27, 38)).corner_radius(6.0).inner_margin(8).show(ui, |ui| {
                    ui.label(RichText::new("Navigate").strong());
                    ui.label(RichText::new("scroll: zoom · right-drag: pan\nclick a colony in the list to fly there\n[ and ]: previous / next colony\nHome: whole world · Space: run / pause").small().weak());
                });
                ui.add_space(6.0);
                ui.heading("Legend");
                for (col, text) in self.view.mode.legend() {
                    ui.horizontal(|ui| {
                        swatch(ui, col, 16.0);
                        ui.label(RichText::new(text).small());
                    });
                }
                ui.add_space(6.0);
                egui::CollapsingHeader::new("Playback").default_open(false).show(ui, |ui| {
                    if ui.button("Step one epoch sub-step by sub-step").on_hover_text("Runs one epoch on the CPU golden model and keeps a frame after each execution sub-step").clicked() {
                        let _ = self.tx.send(Cmd::SubSteps);
                        self.show_sub = true;
                        self.sub_idx = 0;
                    }
                    if nsub > 0 {
                        ui.checkbox(&mut self.show_sub, "show sub-step frames");
                        if self.show_sub {
                            ui.add(egui::Slider::new(&mut self.sub_idx, 0..=nsub - 1).text("sub-step"));
                        }
                    }
                    ui.label(RichText::new(&backend).small().weak());
                });
                egui::CollapsingHeader::new("Tools (interventions)").default_open(false).show(ui, |ui| {
                    ui.label(RichText::new("Every intervention is logged and replays exactly.").small().weak());
                    for (t, n) in [
                        (Tool::Inspect, "🔍 Inspect a cell"),
                        (Tool::Paint, "🖌 Paint resources"),
                        (Tool::VentAdd, "🌋 Place a vent"),
                        (Tool::VentRemove, "🪨 Remove vents"),
                        (Tool::Seed, "🧬 Seed a program"),
                        (Tool::Clear, "🧹 Clear an area"),
                        (Tool::Wall, "🧱 Build a wall"),
                        (Tool::MirrorRegion, "🪞 Mirror a region (2 clicks)"),
                    ] {
                        ui.radio_value(&mut self.tool, t, n);
                    }
                    ui.add(egui::Slider::new(&mut self.radius, 0.0..=64.0).text("radius"));
                    match self.tool {
                        Tool::Paint => {
                            egui::ComboBox::from_label("field").selected_text(self.paint_field.clone()).show_ui(ui, |ui| {
                                for f in ["F", "E", "T", "chem", "pool", "signal"] {
                                    ui.selectable_value(&mut self.paint_field, f.to_string(), f);
                                }
                            });
                            ui.add(egui::DragValue::new(&mut self.paint_amount).prefix("amount "));
                            ui.add(egui::DragValue::new(&mut self.paint_species).prefix("species/signal "));
                            ui.add(egui::DragValue::new(&mut self.paint_hand).range(0..=2).prefix("hand/class (0 L,1 D,2 A) "));
                        }
                        Tool::Seed => {
                            ui.label("genome (hex)");
                            ui.text_edit_multiline(&mut self.genome);
                            egui::ComboBox::from_label("library").selected_text("choose…").show_ui(ui, |ui| {
                                if ui.selectable_label(false, "minimal replicator 12 13 15 c2").clicked() {
                                    self.genome = isa::to_hex(&isa::MINIMAL_REPLICATOR);
                                }
                                if ui.selectable_label(false, "metabolic budding replicator (L)").clicked() {
                                    self.genome = "36 36 36 36 36 36 36 4d 67 40 2c 12 13 15 c4".into();
                                }
                                for g in top.iter().take(10) {
                                    if ui.selectable_label(false, format!("evolved: {g}")).clicked() {
                                        self.genome = g.clone();
                                    }
                                }
                            });
                            ui.checkbox(&mut self.mirror_seed, "also place its mirror image");
                            ui.add(egui::DragValue::new(&mut self.seed_energy).prefix("energy "));
                        }
                        Tool::Wall => {
                            ui.radio_value(&mut self.wall_type, "reflect".to_string(), "reflecting");
                            ui.radio_value(&mut self.wall_type, "absorb".to_string(), "absorbing");
                            ui.radio_value(&mut self.wall_type, "none".to_string(), "remove wall");
                        }
                        Tool::MirrorRegion => {
                            ui.label(if self.mirror_a.is_some() { "click the opposite corner" } else { "click the first corner" });
                        }
                        _ => {}
                    }
                    ui.separator();
                    ui.label("Edit physics (applied next epoch)");
                    ui.text_edit_singleline(&mut self.edit_key);
                    ui.text_edit_singleline(&mut self.edit_val);
                    if ui.button("Apply").clicked() {
                        let val: serde_json::Value = serde_json::from_str(&self.edit_val).unwrap_or(json!(self.edit_val));
                        self.event(epoch, "set", json!({"key": self.edit_key, "value": val}));
                    }
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("💾 Snapshot").clicked() {
                            let _ = self.tx.send(Cmd::Snapshot);
                        }
                        if ui.button("🌿 Branch…").clicked() {
                            let _ = self.tx.send(Cmd::Branch);
                        }
                    });
                });
            });
        });

        // ---------------- right: colonies atlas, chronicle, inspector
        self.hover_colony = None;
        egui::Panel::right("right").default_size(390.0).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Colonies, RichText::new(format!("🗺 Colonies ({})", colonies.len())).strong());
                ui.selectable_value(&mut self.tab, Tab::Chronicle, RichText::new("📜 Chronicle").strong());
                ui.selectable_value(&mut self.tab, Tab::Inspector, RichText::new("🔍 Inspector").strong());
            });
            ui.separator();
            match self.tab {
                Tab::Colonies => {
                    if colonies.is_empty() {
                        ui.label("No sizeable colonies yet. Colonies appear when related cells (copies of each other) cluster together.");
                    }
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("sort by").small().weak());
                        ui.selectable_value(&mut self.sort, Sort::Size, "size");
                        ui.selectable_value(&mut self.sort, Sort::Growth, "growth");
                        ui.selectable_value(&mut self.sort, Sort::Age, "age");
                        ui.checkbox(&mut self.follow, "follow selected");
                    });
                    let mut list: Vec<&Colony> = colonies.iter().collect();
                    match self.sort {
                        Sort::Size => list.sort_by(|a, b| b.size.cmp(&a.size)),
                        Sort::Growth => list.sort_by(|a, b| b.growth.partial_cmp(&a.growth).unwrap_or(std::cmp::Ordering::Equal)),
                        Sort::Age => list.sort_by_key(|c| c.first_seen),
                    }
                    let sel = self.selected;
                    let detail_h = if sel.is_some() { 230.0 } else { 0.0 };
                    let mut clicked: Option<Colony> = None;
                    egui::ScrollArea::vertical().id_salt("atlas").max_height(ui.available_height() - detail_h).show(ui, |ui| {
                        for c in list {
                            let is_sel = sel == Some(c.id);
                            let fill = if is_sel { Color32::from_rgb(34, 48, 56) } else { Color32::from_rgb(22, 27, 38) };
                            let resp = egui::Frame::new().fill(fill).corner_radius(6.0).inner_margin(egui::Margin::symmetric(8, 5)).show(ui, |ui| {
                                ui.set_min_width(ui.available_width());
                                ui.horizontal(|ui| {
                                    swatch(ui, c.color, 22.0);
                                    ui.vertical(|ui| {
                                        ui.horizontal(|ui| {
                                            ui.label(RichText::new(&c.name).strong().color(c32(c.color)));
                                            let (arrow, col) = if c.growth > 1.0 { ("▲", Color32::from_rgb(120, 220, 140)) } else if c.growth < -1.0 { ("▼", Color32::from_rgb(240, 110, 110)) } else { ("●", Color32::GRAY) };
                                            ui.label(RichText::new(format!("{arrow} {}", c.size)).color(col));
                                        });
                                        ui.label(RichText::new(format!("{:.0}% copiers · {:.0}% bonded · {}", c.copiers * 100.0, c.bonded * 100.0, if c.hand.abs() < 0.15 { "mixed hand".to_string() } else if c.hand > 0.0 { "left-handed".into() } else { "right-handed".into() })).small().weak());
                                    });
                                });
                            });
                            let r = resp.response.interact(egui::Sense::click());
                            if r.hovered() {
                                self.hover_colony = Some(c.id);
                            }
                            if r.clicked() {
                                clicked = Some(c.clone());
                            }
                            ui.add_space(3.0);
                        }
                    });
                    if let Some(c) = clicked {
                        self.fly_to(&c);
                    }
                    if let Some(c) = sel.and_then(|id| colonies.iter().find(|c| c.id == id)) {
                        ui.separator();
                        ui.horizontal(|ui| {
                            swatch(ui, c.color, 18.0);
                            ui.label(RichText::new(&c.name).size(17.0).strong().color(c32(c.color)));
                            if ui.small_button("fly there").clicked() {
                                let c2 = c.clone();
                                self.fly_to(&c2);
                            }
                        });
                        for line in colony_story(c, epoch) {
                            ui.label(RichText::new(line).small());
                        }
                        ui.label(RichText::new("How its typical program begins:").small().weak());
                        ui.label(RichText::new(c.prefix.iter().take(12).map(|&b| isa::mnemonic(b)).collect::<Vec<_>>().join(" · ")).small().monospace().color(Color32::from_rgb(200, 210, 225)));
                    } else if self.selected.is_some() {
                        ui.label(RichText::new("The selected colony has dissolved or merged.").small().weak());
                    }
                }
                Tab::Chronicle => {
                    ui.label(RichText::new("Notable things that happened, newest first").small().weak());
                    egui::ScrollArea::vertical().id_salt("chron").show(ui, |ui| {
                        for n in &chronicle {
                            ui.horizontal_wrapped(|ui| {
                                ui.label(RichText::new(n.icon).size(16.0));
                                ui.label(RichText::new(format!("epoch {}", n.epoch)).small().monospace().color(Color32::from_rgb(120, 130, 150)));
                                ui.label(RichText::new(&n.text).small());
                            });
                            ui.add_space(2.0);
                        }
                    });
                }
                Tab::Inspector => match &inspect {
                    None => {
                        ui.label("Choose the 🔍 Inspect tool (Tools section) and click a cell to see its program, energy and lineage, and step through its next slice.");
                    }
                    Some(ins) => {
                        egui::ScrollArea::vertical().id_salt("insp").show(ui, |ui| {
                            ui.label(RichText::new(&ins.text).monospace().small());
                            ui.separator();
                            ui.label("program (chiral ops: red = L, blue = D; shaded = catalyst active site)");
                            egui::Grid::new("tape").striped(true).show(ui, |ui| {
                                for (i, &b) in ins.tape.iter().enumerate() {
                                    let col = if isa::is_chiral(b) {
                                        if b & 1 == 0 { Color32::from_rgb(230, 90, 80) } else { Color32::from_rgb(90, 140, 240) }
                                    } else if b == 0 {
                                        Color32::DARK_GRAY
                                    } else {
                                        Color32::LIGHT_GRAY
                                    };
                                    let in_site = ins.cat_eta.iter().any(|&(ci, _)| i < ci && ci - i <= 8);
                                    let mut txt = RichText::new(format!("{i:3} {b:02x} {}", isa::mnemonic(b))).monospace().small().color(col);
                                    if in_site {
                                        txt = txt.background_color(Color32::from_rgb(60, 60, 20));
                                    }
                                    ui.label(txt);
                                    if let Some(&(_, eta)) = ins.cat_eta.iter().find(|&&(ci, _)| ci == i) {
                                        ui.label(RichText::new(format!("η = {eta:.2}")).small().color(Color32::YELLOW));
                                    }
                                    if i % 2 == 1 {
                                        ui.end_row();
                                    }
                                }
                            });
                            ui.separator();
                            ui.label(format!("CPU debugger: its next slice ({} steps, read-only)", ins.trace.len()));
                            if !ins.trace.is_empty() {
                                ui.add(egui::Slider::new(&mut self.trace_idx, 0..=ins.trace.len() - 1).text("step"));
                                let s = &ins.trace[self.trace_idx.min(ins.trace.len() - 1)];
                                ui.label(
                                    RichText::new(format!(
                                        "IP {:3}  op {:02x} {:<10}\nH0 {:4}  H1 {:4}  E {}\nstack {:?}\npartner {:?}  order {}  len self {}  partner {}",
                                        s.ip,
                                        s.op,
                                        isa::mnemonic(s.op),
                                        s.h0,
                                        s.h1,
                                        s.energy,
                                        s.stack,
                                        s.partner,
                                        if s.partner_first { "partner+self" } else { "self+partner" },
                                        s.len_self,
                                        s.len_partner
                                    ))
                                    .monospace(),
                                );
                            }
                        });
                    }
                },
            }
        });

        // ---------------- bottom: trends
        egui::Panel::bottom("plots").exact_size(if self.show_plots { 220.0 } else { 34.0 }).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.show_plots, RichText::new("Trends").strong());
                if self.show_plots {
                    for k in series.keys() {
                        if !self.plots.contains_key(k) {
                            self.plots.insert(k.clone(), false);
                        }
                    }
                    ui.menu_button("choose metrics ▾", |ui| {
                        for (k, on) in self.plots.iter_mut() {
                            ui.checkbox(on, k.as_str());
                        }
                    });
                }
            });
            if !self.show_plots {
                return;
            }
            let shown: Vec<String> = self.plots.iter().filter(|(_, on)| **on).map(|(k, _)| k.clone()).collect();
            let n = shown.len().max(1);
            let w = (ui.available_width() / n as f32).max(160.0) - 8.0;
            ui.horizontal(|ui| {
                for k in shown {
                    ui.vertical(|ui| {
                        ui.label(RichText::new(&k).small());
                        let pts = series.get(&k).cloned().unwrap_or_default();
                        egui_plot::Plot::new(format!("plot-{k}")).width(w).height(160.0).show_axes([true, true]).allow_scroll(false).allow_drag(false).show(ui, |pu| {
                            pu.line(egui_plot::Line::new(k.clone(), pts).color(Color32::from_rgb(110, 210, 190)));
                        });
                    });
                }
            });
        });

        // ---------------- centre: map
        egui::CentralPanel::default().show(ui, |ui| {
            let f = fr;
            let avail = ui.available_size();
            let n = f.images.len();
            let (cols, rows) = if n >= 3 { (2, 2) } else { (n.max(1), 1) };
            let cell = egui::vec2(avail.x / cols as f32, avail.y / rows as f32);
            let (resp, painter) = ui.allocate_painter(avail, egui::Sense::click_and_drag());
            painter.rect_filled(resp.rect, 0.0, Color32::from_rgb(8, 10, 15));
            let base_fit = |im: &Image| ((cell.x - 8.0) / im.w as f32).min((cell.y - 26.0) / im.h as f32).max(0.01);
            // keyboard pan / zoom
            if resp.hovered() && !ctx.egui_wants_keyboard_input() {
                let (l, r, u, d, zi, zo) = ui.input(|i| (i.key_down(egui::Key::ArrowLeft), i.key_down(egui::Key::ArrowRight), i.key_down(egui::Key::ArrowUp), i.key_down(egui::Key::ArrowDown), i.key_pressed(egui::Key::Plus) || i.key_pressed(egui::Key::Equals), i.key_pressed(egui::Key::Minus)));
                let step = 12.0;
                self.pan += egui::vec2((l as i32 - r as i32) as f32 * step, (u as i32 - d as i32) as f32 * step);
                if zi {
                    self.zoom *= 1.25;
                }
                if zo {
                    self.zoom /= 1.25;
                }
            }
            if resp.hovered() {
                let sc = ui.input(|i| i.smooth_scroll_delta.y);
                if sc != 0.0 {
                    self.fly = None;
                    let old = self.zoom;
                    self.zoom = (self.zoom * (1.0 + sc * 0.002)).clamp(0.3, 300.0);
                    if let Some(pos) = resp.hover_pos() {
                        let anchor = pos - resp.rect.min - egui::vec2(4.0, 24.0) - self.pan;
                        self.pan -= anchor * (self.zoom / old - 1.0);
                    }
                }
            }
            if resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) {
                self.pan += resp.drag_delta();
                self.fly = None;
                self.follow = false;
            }
            // follow the selected colony as it moves
            if self.follow && dims == 2 {
                if let Some(c) = self.selected.and_then(|id| f.colonies.iter().find(|c| c.id == id)) {
                    self.fly = Some(([c.center[0], c.center[1]], self.zoom));
                }
            }
            // smooth fly-to (first image)
            if let (Some((target, tz)), Some(im)) = (self.fly, f.images.first()) {
                if dims == 2 || dims == 0 {
                    let bf = base_fit(im);
                    self.zoom += (tz - self.zoom) * 0.18;
                    let fit = bf * self.zoom;
                    let want = egui::vec2(cell.x / 2.0 - 4.0 - (target[0] + 0.5) * fit, cell.y / 2.0 - 24.0 - (im.h as f32 - target[1] - 0.5) * fit);
                    let d = want - self.pan;
                    self.pan += d * 0.18;
                    if d.length() < 1.0 && (tz - self.zoom).abs() < 0.01 * tz && !self.follow {
                        self.fly = None;
                    }
                }
            }
            let time = ui.input(|i| i.time) as f32;
            let origin = resp.rect.min;
            let mut hovered: Option<([i64; 3], Option<CellBrief>, Option<usize>, egui::Pos2)> = None;
            for (i, im) in f.images.iter().enumerate() {
                let slot = egui::Rect::from_min_size(origin + egui::vec2((i % cols) as f32 * cell.x, (i / cols) as f32 * cell.y), cell);
                let fit = base_fit(im) * self.zoom;
                let img = egui::Rect::from_min_size(slot.min + egui::vec2(4.0, 24.0) + self.pan, egui::vec2(im.w as f32 * fit, im.h as f32 * fit));
                painter.text(slot.min + egui::vec2(6.0, 4.0), egui::Align2::LEFT_TOP, &im.label, egui::FontId::proportional(14.0), Color32::from_rgb(200, 210, 225));
                let clip = painter.with_clip_rect(slot.shrink(1.0).intersect(egui::Rect::from_min_max(slot.min + egui::vec2(0.0, 22.0), slot.max)));
                let to_screen = |x: f32, y: f32| egui::pos2(img.min.x + x * fit, img.max.y - y * fit);
                let disc = dims == 2 && fit >= 7.0 && !f.cells.is_empty();
                if let Some(Some(t)) = self.tex.get(i) {
                    let tint = if disc { Color32::from_gray(80) } else { Color32::WHITE };
                    clip.image(t.id(), img, egui::Rect::from_min_max(egui::pos2(0.0, 1.0), egui::pos2(1.0, 0.0)), tint);
                }
                clip.rect_stroke(img, 0.0, egui::Stroke::new(1.0, Color32::from_rgb(50, 58, 75)), egui::StrokeKind::Outside);
                // zoomed in: cells as discs joined by bonds
                if disc {
                    let vis = clip.clip_rect();
                    let x0 = (((vis.min.x - img.min.x) / fit).floor().max(0.0)) as usize;
                    let x1 = (((vis.max.x - img.min.x) / fit).ceil() as usize).min(im.w);
                    let y0 = (((img.max.y - vis.max.y) / fit).floor().max(0.0)) as usize;
                    let y1 = (((img.max.y - vis.min.y) / fit).ceil() as usize).min(im.h);
                    if (x1.saturating_sub(x0)) * (y1.saturating_sub(y0)) < 60_000 {
                        for y in y0..y1 {
                            for x in x0..x1 {
                                let c = x + y * im.w;
                                let cb = &f.cells[c];
                                if !cb.info.occ {
                                    continue;
                                }
                                let k = (y * im.w + x) * 4;
                                let col = Color32::from_rgb(im.rgba[k], im.rgba[k + 1], im.rgba[k + 2]);
                                let center = to_screen(x as f32 + 0.5, y as f32 + 0.5);
                                if self.show_bonds && !f.bonds.is_empty() {
                                    let b = f.bonds[c];
                                    if b & 1 != 0 {
                                        clip.line_segment([center, to_screen(x as f32 + 1.5, y as f32 + 0.5)], egui::Stroke::new(fit * 0.18, Color32::from_rgb(220, 225, 235)));
                                    }
                                    if b & 4 != 0 {
                                        clip.line_segment([center, to_screen(x as f32 + 0.5, y as f32 + 1.5)], egui::Stroke::new(fit * 0.18, Color32::from_rgb(220, 225, 235)));
                                    }
                                }
                                clip.circle_filled(center, fit * 0.42, col);
                                if cb.info.copier {
                                    clip.circle_stroke(center, fit * 0.42, egui::Stroke::new((fit * 0.08).max(1.0), Color32::WHITE));
                                }
                            }
                        }
                    }
                }
                // vents
                if self.show_vents && dims >= 1 {
                    let (ax, ay) = im.axes;
                    for v in &f.vents {
                        let on = dims <= 2 || (0..3).all(|a| a == ax || a == ay || v[a].abs_diff(im.fixed[a]) <= 2);
                        if !on {
                            continue;
                        }
                        let yv = if dims >= 2 { v[ay] as f32 } else { 0.0 };
                        let pc = to_screen(v[ax] as f32 + 0.5, yv + 0.5);
                        let pulse = 0.5 + 0.5 * (time * 2.5 + v[ax] as f32).sin();
                        let base = (fit * 2.5).max(9.0);
                        for k in 0..3 {
                            let r = base * (1.0 + 0.6 * k as f32 + 0.4 * pulse);
                            let a = (60.0 - 15.0 * k as f32) * (0.6 + 0.4 * pulse);
                            clip.circle_filled(pc, r, Color32::from_rgba_unmultiplied(255, 140, 30, a as u8));
                        }
                        clip.circle_filled(pc, base * 0.5, Color32::from_rgb(255, 220, 120));
                    }
                }
                if self.show_domains && !f.domains.is_empty() && dims == 2 {
                    let (bx, by) = (f.dnb[0], f.dnb[1]);
                    for j in 0..by {
                        for i2 in 0..bx {
                            let s = f.domains[i2 + bx * j];
                            if s == 0 {
                                continue;
                            }
                            let x0 = (i2 * f.size[0] as usize / bx) as f32;
                            let x1 = ((i2 + 1) * f.size[0] as usize / bx) as f32;
                            let y0 = (j * f.size[1] as usize / by) as f32;
                            let y1 = ((j + 1) * f.size[1] as usize / by) as f32;
                            let col = if s > 0 { Color32::from_rgb(255, 80, 80) } else { Color32::from_rgb(80, 120, 255) };
                            clip.rect_stroke(egui::Rect::from_two_pos(to_screen(x0, y0), to_screen(x1, y1)), 0.0, egui::Stroke::new(1.5, col), egui::StrokeKind::Inside);
                        }
                    }
                }
                // colony highlight + labels (2D)
                if dims == 2 && i == 0 {
                    for c in f.colonies.iter() {
                        let hl = self.hover_colony == Some(c.id) || self.selected == Some(c.id);
                        let pc = to_screen(c.center[0] + 0.5, c.center[1] + 0.5);
                        if hl {
                            let r = (c.radius * 1.6 * fit).max(14.0);
                            clip.circle_stroke(pc, r, egui::Stroke::new(3.0, Color32::WHITE));
                            clip.circle_stroke(pc, r + 4.0, egui::Stroke::new(2.0, c32(c.color)));
                        }
                    }
                    if self.labels {
                        let mut placed: Vec<egui::Rect> = vec![];
                        for c in f.colonies.iter().take(16) {
                            let pc = to_screen(c.center[0] + 0.5, c.center[1] + 0.5);
                            if !clip.clip_rect().contains(pc) {
                                continue;
                            }
                            let text = format!("{}  {}", c.name, c.size);
                            let galley = painter.layout_no_wrap(text, egui::FontId::proportional(13.0), Color32::WHITE);
                            let r = egui::Rect::from_center_size(pc, galley.size() + egui::vec2(16.0, 6.0));
                            if placed.iter().any(|p| p.intersects(r)) {
                                continue;
                            }
                            placed.push(r);
                            clip.rect_filled(r, 10.0, Color32::from_rgba_unmultiplied(10, 12, 18, 200));
                            clip.circle_filled(r.left_center() + egui::vec2(7.0, 0.0), 4.0, c32(c.color));
                            clip.galley(r.min + egui::vec2(12.0, 3.0), galley, Color32::WHITE);
                        }
                    }
                }
                if let Some(pos) = resp.hover_pos() {
                    if img.contains(pos) && clip.clip_rect().contains(pos) {
                        let x = ((pos.x - img.min.x) / fit) as i64;
                        let y = ((img.max.y - pos.y) / fit) as i64;
                        let mut q = [im.fixed[0] as i64, im.fixed[1] as i64, im.fixed[2] as i64];
                        let mut brief = None;
                        let mut col = None;
                        match dims {
                            0 => q = [x + y * im.w as i64, 0, 0],
                            1 => q = [x, 0, 0],
                            _ => {
                                q[im.axes.0] = x;
                                q[im.axes.1] = y;
                                if dims == 2 {
                                    let c = x as usize + y as usize * im.w;
                                    brief = f.cells.get(c).copied();
                                    col = f.colony_of.get(c).and_then(|&k| if k > 0 { Some(k as usize - 1) } else { None });
                                    clip.rect_stroke(egui::Rect::from_min_size(to_screen(x as f32, y as f32 + 1.0), egui::vec2(fit, fit)).expand(1.0), 2.0, egui::Stroke::new(1.5, Color32::WHITE), egui::StrokeKind::Outside);
                                }
                            }
                        }
                        hovered = Some((q, brief, col, pos));
                    }
                }
                // minimap (2D, zoomed in)
                if self.minimap && dims == 2 && i == 0 && self.zoom > 1.6 {
                    if let Some(Some(t)) = self.tex.get(0) {
                        let mw = 180.0;
                        let mh = mw * im.h as f32 / im.w as f32;
                        let mr = egui::Rect::from_min_size(egui::pos2(slot.max.x - mw - 12.0, slot.min.y + 30.0), egui::vec2(mw, mh));
                        painter.rect_filled(mr.expand(3.0), 4.0, Color32::from_rgba_unmultiplied(0, 0, 0, 200));
                        painter.image(t.id(), mr, egui::Rect::from_min_max(egui::pos2(0.0, 1.0), egui::pos2(1.0, 0.0)), Color32::WHITE);
                        // viewport rectangle
                        let vis = slot.intersect(img);
                        let lx0 = (vis.min.x - img.min.x) / fit / im.w as f32;
                        let lx1 = (vis.max.x - img.min.x) / fit / im.w as f32;
                        let ly0 = (vis.min.y - img.min.y) / fit / im.h as f32;
                        let ly1 = (vis.max.y - img.min.y) / fit / im.h as f32;
                        let vr = egui::Rect::from_min_max(mr.min + egui::vec2(lx0 * mw, ly0 * mh), mr.min + egui::vec2(lx1 * mw, ly1 * mh));
                        painter.rect_stroke(vr, 0.0, egui::Stroke::new(1.5, Color32::WHITE), egui::StrokeKind::Outside);
                        // click the minimap to jump
                        if let Some(p) = resp.interact_pointer_pos() {
                            if resp.clicked() && mr.contains(p) {
                                let lx = (p.x - mr.min.x) / mw * im.w as f32;
                                let ly = im.h as f32 - (p.y - mr.min.y) / mh * im.h as f32;
                                self.fly = Some(([lx, ly], self.zoom));
                                hovered = None;
                            }
                        }
                    }
                }
            }
            if let Some((q, brief, col, pos)) = hovered {
                let colony = col.and_then(|k| f.colonies.get(k));
                if let Some(c) = colony {
                    self.hover_colony = Some(c.id);
                }
                let head = format!("{:?}", &q[..dims.max(1) as usize]);
                let (title, body) = match brief {
                    Some(cb) => describe(&cb, colony),
                    None => ("".into(), "click to inspect".into()),
                };
                let text = format!("{head}  {title}\n{body}");
                let galley = painter.layout(text, egui::FontId::proportional(13.0), Color32::from_rgb(230, 235, 245), 360.0);
                let mut at = pos + egui::vec2(16.0, 16.0);
                if at.x + galley.size().x + 16.0 > resp.rect.max.x {
                    at.x = pos.x - galley.size().x - 24.0;
                }
                if at.y + galley.size().y + 16.0 > resp.rect.max.y {
                    at.y = pos.y - galley.size().y - 24.0;
                }
                let r = egui::Rect::from_min_size(at, galley.size()).expand(8.0);
                painter.rect_filled(r, 6.0, Color32::from_rgba_unmultiplied(16, 20, 30, 235));
                painter.rect_stroke(r, 6.0, egui::Stroke::new(1.0, colony.map(|c| c32(c.color)).unwrap_or(Color32::from_rgb(70, 90, 110))), egui::StrokeKind::Outside);
                painter.galley(at, galley, Color32::WHITE);
                if resp.clicked_by(egui::PointerButton::Primary) {
                    if self.tool == Tool::Inspect {
                        if let Some(c) = colony {
                            self.selected = Some(c.id);
                            self.tab = Tab::Colonies;
                        }
                        if resp.double_clicked() || colony.is_none() {
                            self.apply_tool(epoch, q, dims);
                        }
                    } else {
                        self.apply_tool(epoch, q, dims);
                    }
                }
            }
        });
    }
}

fn upload(ctx: &egui::Context, tex: &mut Vec<Option<egui::TextureHandle>>, f: &Frame) {
    if tex.len() != f.images.len() {
        *tex = (0..f.images.len()).map(|_| None).collect();
    }
    for (i, im) in f.images.iter().enumerate() {
        let ci = egui::ColorImage::from_rgba_unmultiplied([im.w, im.h], &im.rgba);
        match &mut tex[i] {
            Some(t) => t.set(ci, egui::TextureOptions::NEAREST),
            None => tex[i] = Some(ctx.load_texture(format!("map{i}"), ci, egui::TextureOptions::NEAREST)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use primordia_core::{Config, CpuExec};

    #[test]
    fn render_all_modes_and_inspect() {
        for dims in 0..=3u32 {
            let mut c = Config::preset("primordia-2d-sun").unwrap();
            c.world.dims = dims;
            c.world.size = match dims {
                0 => vec![300],
                1 => vec![60],
                2 => vec![30, 24],
                _ => vec![9, 9, 9],
            };
            c.energy.vents.enabled = true;
            let mut sim = Sim::new(c, Box::new(CpuExec)).unwrap();
            for _ in 0..5 {
                sim.step();
            }
            let mut v = ViewSettings::default(&sim.p);
            let mut ctx = render::RenderCtx::default();
            for m in Mode::all() {
                v.mode = m;
                let f = render::render(&sim.st, &sim.p, &v, &[], &mut ctx);
                assert!(!f.images.is_empty());
                for im in &f.images {
                    assert_eq!(im.rgba.len(), im.w * im.h * 4);
                }
            }
            let c = (0..sim.st.n).find(|&c| sim.st.occupied(c)).unwrap();
            let ins = inspect(&sim, c, &HashMap::new());
            assert!(!ins.trace.is_empty(), "debugger trace for dims {dims}");
            let rec = Recorder::new(None, &sim).unwrap();
            let p = web::payload(&sim, &rec, 1.0, 0.5, &[]);
            assert!(p.1.len() >= sim.st.n * 4);
        }
    }
}
