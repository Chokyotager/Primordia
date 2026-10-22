//! Native egui viewer (§14): live maps, 3D slices, plots, inspector with a
//! CPU-golden-model debugger, and logged interventions.

mod render;

use anyhow::Result;
use eframe::egui;
use primordia_core::events::Event;
use primordia_core::exec::{TraceStep, exec_slice, flag_snapshot};
use primordia_core::isa;
use primordia_core::params::*;
use primordia_core::{Sim, phases};
use primordia_telemetry::{Recorder, Summary};
use render::{Mode, ViewSettings};
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Called after every epoch (and for each intervention / snapshot request).
pub type Hook = Box<dyn FnMut(&Sim, Option<&Event>, bool) -> Result<Option<PathBuf>> + Send>;

pub struct Image {
    pub label: String,
    pub w: usize,
    pub h: usize,
    pub rgba: Vec<u8>,
    /// lattice coordinate of pixel (0,0) and the axes the image spans
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
}

#[derive(Default, Clone)]
pub struct Inspect {
    pub cell: usize,
    pub text: String,
    pub tape: Vec<u8>,
    pub cat_eta: Vec<(usize, f64)>,
    pub trace: Vec<TraceStep>,
}

#[derive(Default)]
struct Out {
    frame: Option<Frame>,
    sub_frames: Vec<Frame>,
    summary: Summary,
    series: BTreeMap<String, Vec<[f64; 2]>>,
    inspect: Option<Inspect>,
    msgs: Vec<String>,
    eps: f64,
    epoch: u64,
    running: bool,
    backend: String,
    dims: u32,
    size: [u32; 3],
    k_species: Vec<String>,
    n_signals: usize,
    top_genomes: Vec<String>,
}

enum Cmd {
    Run(bool),
    Step(u64),
    SubSteps,
    RedrawEvery(u64),
    View(ViewSettings),
    Event(Event),
    Inspect(usize),
    Snapshot,
    Branch,
    Quit,
}

pub fn run_viewer(sim: Sim, rec: Recorder, hook: Hook, autorun: bool) -> Result<()> {
    let out = Arc::new(Mutex::new(Out::default()));
    let (tx, rx) = channel();
    let o2 = out.clone();
    let worker = std::thread::spawn(move || worker(sim, rec, hook, rx, o2, autorun));
    let opts = eframe::NativeOptions { viewport: egui::ViewportBuilder::default().with_inner_size([1600.0, 1000.0]).with_title("Primordia"), ..Default::default() };
    let app = App::new(tx.clone(), out);
    let r = eframe::run_native("Primordia", opts, Box::new(|_cc| Ok(Box::new(app))));
    let _ = tx.send(Cmd::Quit);
    let _ = worker.join();
    r.map_err(|e| anyhow::anyhow!("viewer: {e}"))
}

// ------------------------------------------------------------------ worker

fn worker(mut sim: Sim, mut rec: Recorder, mut hook: Hook, rx: Receiver<Cmd>, out: Arc<Mutex<Out>>, autorun: bool) {
    let mut running = autorun;
    let mut pending_steps = 0u64;
    let mut redraw_every = 1u64;
    let mut view = ViewSettings::default(&sim.p);
    let mut lineage: HashMap<u32, u32> = HashMap::new();
    let mut last_render = Instant::now() - Duration::from_secs(10);
    let mut rate_t = Instant::now();
    let mut rate_e = sim.st.epoch;
    let mut eps = 0.0;
    let mut dirty = true;
    let mut history_1d: Vec<Vec<u8>> = vec![];
    {
        let mut o = out.lock().unwrap();
        o.backend = sim.backend.name();
        o.dims = sim.p.geom.dims;
        o.size = sim.p.geom.size;
        o.k_species = if sim.p.chem_on { sim.cfg.chemistry.species.clone() } else { vec![] };
        o.n_signals = sim.p.s;
    }
    let msg = |out: &Arc<Mutex<Out>>, m: String| {
        let mut o = out.lock().unwrap();
        o.msgs.push(m);
        if o.msgs.len() > 200 {
            o.msgs.remove(0);
        }
    };
    loop {
        let mut sub = false;
        while let Ok(c) = rx.try_recv() {
            match c {
                Cmd::Quit => {
                    let _ = rec.finish();
                    return;
                }
                Cmd::Run(r) => running = r,
                Cmd::Step(n) => pending_steps += n,
                Cmd::SubSteps => sub = true,
                Cmd::RedrawEvery(n) => redraw_every = n.max(1),
                Cmd::View(v) => {
                    // keep the worker's own Params (the UI only knows the geometry)
                    view = ViewSettings { params: view.params.clone(), ..v };
                    dirty = true;
                }
                Cmd::Event(e) => {
                    let _ = hook(&sim, Some(&e), false);
                    msg(&out, format!("epoch {}: {} {}", e.epoch, e.kind, e.params));
                    sim.push_event(e);
                }
                Cmd::Inspect(c) => {
                    let ins = inspect(&sim, c, &lineage);
                    out.lock().unwrap().inspect = Some(ins);
                }
                Cmd::Snapshot | Cmd::Branch => {
                    let branch = matches!(c, Cmd::Branch);
                    match hook(&sim, None, true) {
                        Ok(Some(p)) => {
                            if branch {
                                let diff = p.with_extension("diff.toml");
                                let _ = std::fs::write(&diff, "# config changes for the branch, e.g.\n# [mutation]\n# background = 1e-3\n");
                                msg(&out, format!("branch point saved. Edit {} then run: primordia branch {} {}", diff.display(), p.display(), diff.display()));
                            } else {
                                msg(&out, format!("snapshot written: {}", p.display()));
                            }
                        }
                        Ok(None) => msg(&out, "no run directory: snapshot not written".into()),
                        Err(e) => msg(&out, format!("snapshot failed: {e:#}")),
                    }
                }
            }
        }
        let do_step = running || pending_steps > 0 || sub;
        if do_step {
            if sub {
                let mut frames = vec![];
                let v = view.clone();
                sim.step_substeps(|st, i| {
                    let mut f = render::render(st, &v.params, &v, &[], &mut vec![]);
                    f.images.iter_mut().for_each(|im| im.label = format!("{} — sub-step {}", im.label, i));
                    frames.push(f);
                });
                out.lock().unwrap().sub_frames = frames;
            } else {
                sim.step();
            }
            if pending_steps > 0 {
                pending_steps -= 1;
            }
            for c in 0..sim.st.n {
                let h = &sim.st.hdr[c * HDR..(c + 1) * HDR];
                if h[H_FLAGS] & FL_NEWBORN != 0 {
                    lineage.insert(h[H_LINEAGE], h[H_PARENT]);
                }
            }
            if let Err(e) = rec.record(&sim) {
                msg(&out, format!("telemetry error: {e:#}"));
            }
            if let Err(e) = hook(&sim, None, false) {
                msg(&out, format!("run-dir error: {e:#}"));
            }
            for r in sim.rejected.drain(..) {
                msg(&out, format!("rejected: {r}"));
            }
            dirty = dirty || sim.st.epoch % redraw_every == 0 || !running;
            if rate_t.elapsed() > Duration::from_millis(500) {
                eps = (sim.st.epoch - rate_e) as f64 / rate_t.elapsed().as_secs_f64();
                rate_t = Instant::now();
                rate_e = sim.st.epoch;
            }
        }
        if dirty && last_render.elapsed() > Duration::from_millis(30) {
            let f = render::render(&sim.st, &sim.p, &view, &rec.last_domains, &mut history_1d);
            let mut o = out.lock().unwrap();
            o.frame = Some(f);
            o.summary = rec.summary.clone();
            o.series = rec.series.clone();
            o.top_genomes = rec.last_assay.iter().filter(|a| a.replicator).map(|a| isa::to_hex(&a.minimal)).collect();
            dirty = false;
            last_render = Instant::now();
        }
        {
            let mut o = out.lock().unwrap();
            o.eps = eps;
            o.epoch = sim.st.epoch;
            o.running = running;
        }
        if !do_step {
            std::thread::sleep(Duration::from_millis(15));
        }
    }
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
        // lineage chain
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

struct App {
    tx: Sender<Cmd>,
    out: Arc<Mutex<Out>>,
    tex: Vec<Option<egui::TextureHandle>>,
    tex_epoch: Vec<u64>,
    view: Option<ViewSettings>,
    zoom: f32,
    pan: egui::Vec2,
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
    redraw: u64,
    overlays: (bool, bool, bool),
    plots: BTreeMap<String, bool>,
    trace_idx: usize,
    sub_idx: usize,
    show_sub: bool,
    hover: Option<[i64; 3]>,
}

impl App {
    fn new(tx: Sender<Cmd>, out: Arc<Mutex<Out>>) -> App {
        let mut plots = BTreeMap::new();
        for k in ["high-order entropy", "population", "EE code", "EE chemicals", "EE monomers", "ledger |residual|", "replicators (top 100)", "species"] {
            plots.insert(k.to_string(), true);
        }
        App {
            tx,
            out,
            tex: vec![],
            tex_epoch: vec![],
            view: None,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            tool: Tool::Inspect,
            paint_field: "F".into(),
            paint_amount: 500,
            paint_species: 0,
            paint_hand: 0,
            radius: 4.0,
            genome: isa::to_hex(&isa::MINIMAL_REPLICATOR),
            mirror_seed: false,
            seed_energy: 1000,
            wall_type: "reflect".into(),
            mirror_a: None,
            edit_key: "mutation.background".into(),
            edit_val: "1e-3".into(),
            redraw: 1,
            overlays: (true, true, false),
            plots,
            trace_idx: 0,
            sub_idx: 0,
            show_sub: false,
            hover: None,
        }
    }

    fn event(&self, epoch: u64, kind: &str, params: serde_json::Value) {
        let _ = self.tx.send(Cmd::Event(Event { epoch, kind: kind.into(), params }));
    }

    fn apply_tool(&mut self, epoch: u64, p: [i64; 3], dims: u32) {
        let center = json!(&p[..dims.max(1) as usize]);
        match self.tool {
            Tool::Inspect => {
                let v = self.view.as_ref().unwrap();
                let g = &v.params.geom;
                let mut q = [0u32; 3];
                for a in 0..3 {
                    q[a] = p[a].clamp(0, g.size[a] as i64 - 1) as u32;
                }
                let _ = self.tx.send(Cmd::Inspect(g.index(q)));
                self.trace_idx = 0;
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

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        ctx.request_repaint_after(Duration::from_millis(33));
        let o = self.out.lock().unwrap();
        if self.view.is_none() {
            if o.frame.is_none() {
                drop(o);
                egui::CentralPanel::default().show(ui, |ui| ui.label("starting simulation…"));
                return;
            }
            // first frame: build view settings from the shared geometry
            self.view = Some(ViewSettings::from_out(o.dims, o.size));
        }
        let epoch = o.epoch;
        let dims = o.dims;
        let summary = o.summary.clone();
        let running = o.running;
        let eps = o.eps;
        let backend = o.backend.clone();
        let species = o.k_species.clone();
        let nsig = o.n_signals;
        let top = o.top_genomes.clone();
        let series = o.series.clone();
        let inspect = o.inspect.clone();
        let msgs = o.msgs.clone();
        let nsub = o.sub_frames.len();
        // upload textures
        let frame_ref: Option<&Frame> = if self.show_sub && nsub > 0 { o.sub_frames.get(self.sub_idx.min(nsub - 1)) } else { o.frame.as_ref() };
        let mut meta = None;
        if let Some(f) = frame_ref {
            if self.tex.len() != f.images.len() {
                self.tex = (0..f.images.len()).map(|_| None).collect();
                self.tex_epoch = vec![u64::MAX; f.images.len()];
            }
            for (i, im) in f.images.iter().enumerate() {
                let ci = egui::ColorImage::from_rgba_unmultiplied([im.w, im.h], &im.rgba);
                match &mut self.tex[i] {
                    Some(t) => t.set(ci, egui::TextureOptions::NEAREST),
                    None => self.tex[i] = Some(ctx.load_texture(format!("map{i}"), ci, egui::TextureOptions::NEAREST)),
                }
            }
            meta = Some((
                f.images.iter().map(|im| (im.label.clone(), im.w, im.h, im.axes, im.fixed)).collect::<Vec<_>>(),
                if self.overlays.0 { f.bonds.clone() } else { vec![] },
                f.vents.clone(),
                f.domains.clone(),
                f.dnb,
                f.size,
                f.epoch,
            ));
        }
        drop(o);

        // ---------------- top bar
        egui::Panel::top("top").show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button(if running { "⏸ Pause" } else { "▶ Run" }).clicked() {
                    let _ = self.tx.send(Cmd::Run(!running));
                }
                if ui.button("Step").clicked() {
                    let _ = self.tx.send(Cmd::Step(1));
                }
                if ui.button("Step ×10").clicked() {
                    let _ = self.tx.send(Cmd::Step(10));
                }
                if ui.button("Step sub-steps").on_hover_text("Run one epoch with each execution sub-step rendered (CPU golden model)").clicked() {
                    let _ = self.tx.send(Cmd::SubSteps);
                    self.show_sub = true;
                    self.sub_idx = 0;
                }
                ui.label("redraw every");
                if ui.add(egui::DragValue::new(&mut self.redraw).range(1..=100000)).changed() {
                    let _ = self.tx.send(Cmd::RedrawEvery(self.redraw));
                }
                ui.separator();
                ui.label(format!("epoch {epoch}   {eps:.1} ep/s   pop {}   len {:.1}   HOE {:.3}   EE code {:+.3}   repl {}   species {}   bodies {} (max {})   ledger violations {}", summary.occupied, summary.mean_len, summary.hoe, summary.ee_code, summary.replicators, summary.species, summary.bodies, summary.max_body, summary.ledger_violations));
            });
            ui.label(egui::RichText::new(backend).small().weak());
        });

        // ---------------- left: view + tools
        egui::Panel::left("left").default_size(290.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                let v = self.view.as_mut().unwrap();
                let mut changed = false;
                ui.heading("View");
                egui::ComboBox::from_label("colour").selected_text(v.mode.name()).show_ui(ui, |ui| {
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
                ui.checkbox(&mut self.overlays.0, "bond overlay");
                ui.checkbox(&mut self.overlays.1, "vent markers");
                ui.checkbox(&mut self.overlays.2, "chirality domains");
                if dims == 3 {
                    let size = v.params.geom.size;
                    for (a, n) in ["x (YZ panel)", "y (XZ panel)", "z (XY panel)"].iter().enumerate() {
                        changed |= ui.add(egui::Slider::new(&mut v.slice[a], 0..=size[a] - 1).text(*n)).changed();
                    }
                }
                if changed {
                    let _ = self.tx.send(Cmd::View(v.clone()));
                }
                if nsub > 0 {
                    ui.checkbox(&mut self.show_sub, "show sub-step frames");
                    if self.show_sub {
                        ui.add(egui::Slider::new(&mut self.sub_idx, 0..=nsub - 1).text("sub-step"));
                    }
                }
                if ui.button("reset zoom").clicked() {
                    self.zoom = 1.0;
                    self.pan = egui::Vec2::ZERO;
                }
                ui.separator();
                ui.heading("Tool");
                for (t, n) in [
                    (Tool::Inspect, "Inspect"),
                    (Tool::Paint, "Paint resources"),
                    (Tool::VentAdd, "Place vent"),
                    (Tool::VentRemove, "Remove vents"),
                    (Tool::Seed, "Seed program"),
                    (Tool::Clear, "Clear / kill"),
                    (Tool::Wall, "Wall"),
                    (Tool::MirrorRegion, "Mirror region (2 clicks)"),
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
                            if ui.selectable_label(false, "budding copier (ALLOC + copy)").clicked() {
                                self.genome = "5f 09 40 07 67 12 13 15 c2".into();
                            }
                            for g in top.iter().take(10) {
                                if ui.selectable_label(false, format!("evolved: {g}")).clicked() {
                                    self.genome = g.clone();
                                }
                            }
                        });
                        ui.checkbox(&mut self.mirror_seed, "also place mirror image");
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
                ui.heading("Edit physics");
                ui.text_edit_singleline(&mut self.edit_key);
                ui.text_edit_singleline(&mut self.edit_val);
                if ui.button("apply at next epoch").clicked() {
                    let val: serde_json::Value = serde_json::from_str(&self.edit_val).unwrap_or(json!(self.edit_val));
                    self.event(epoch, "set", json!({"key": self.edit_key, "value": val}));
                }
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Snapshot").clicked() {
                        let _ = self.tx.send(Cmd::Snapshot);
                    }
                    if ui.button("Branch…").clicked() {
                        let _ = self.tx.send(Cmd::Branch);
                    }
                });
                ui.separator();
                ui.heading("Log");
                for m in msgs.iter().rev().take(30) {
                    ui.label(egui::RichText::new(m).small());
                }
            });
        });

        // ---------------- right: inspector
        if let Some(ins) = &inspect {
            egui::Panel::right("inspector").default_size(420.0).show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Inspector");
                    ui.label(egui::RichText::new(&ins.text).monospace().small());
                    ui.separator();
                    ui.label("tape");
                    egui::Grid::new("tape").striped(true).show(ui, |ui| {
                        for (i, &b) in ins.tape.iter().enumerate() {
                            let col = if isa::is_chiral(b) {
                                if b & 1 == 0 { egui::Color32::from_rgb(230, 90, 80) } else { egui::Color32::from_rgb(90, 140, 240) }
                            } else if b == 0 {
                                egui::Color32::DARK_GRAY
                            } else {
                                egui::Color32::LIGHT_GRAY
                            };
                            let in_site = ins.cat_eta.iter().any(|&(ci, _)| i < ci && ci - i <= 8);
                            let mut txt = egui::RichText::new(format!("{i:3} {b:02x} {}", isa::mnemonic(b))).monospace().small().color(col);
                            if in_site {
                                txt = txt.background_color(egui::Color32::from_rgb(60, 60, 20));
                            }
                            ui.label(txt);
                            if let Some(&(_, eta)) = ins.cat_eta.iter().find(|&&(ci, _)| ci == i) {
                                ui.label(egui::RichText::new(format!("η = {eta:.2}")).small().color(egui::Color32::YELLOW));
                            }
                            if i % 2 == 1 {
                                ui.end_row();
                            }
                        }
                    });
                    ui.separator();
                    ui.label(format!("CPU debugger: next slice ({} steps, read-only)", ins.trace.len()));
                    if !ins.trace.is_empty() {
                        ui.add(egui::Slider::new(&mut self.trace_idx, 0..=ins.trace.len() - 1).text("step"));
                        let s = &ins.trace[self.trace_idx.min(ins.trace.len() - 1)];
                        ui.label(
                            egui::RichText::new(format!(
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
            });
        }

        // ---------------- bottom: plots
        egui::Panel::bottom("plots").exact_size(250.0).show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                for (k, on) in self.plots.iter_mut() {
                    ui.checkbox(on, k);
                }
                for k in series.keys() {
                    if !self.plots.contains_key(k) {
                        self.plots.insert(k.clone(), false);
                    }
                }
            });
            let shown: Vec<String> = self.plots.iter().filter(|(_, on)| **on).map(|(k, _)| k.clone()).collect();
            let n = shown.len().max(1);
            let w = (ui.available_width() / n as f32).max(160.0) - 8.0;
            let h = 200.0;
            ui.horizontal(|ui| {
                for k in shown {
                    ui.vertical(|ui| {
                        ui.label(egui::RichText::new(&k).small());
                        let pts = series.get(&k).cloned().unwrap_or_default();
                        egui_plot::Plot::new(format!("plot-{k}")).width(w).height(h - 18.0).show_axes([true, true]).allow_scroll(false).show(ui, |pu| {
                            pu.line(egui_plot::Line::new(k.clone(), pts));
                        });
                    });
                }
            });
        });

        // ---------------- centre: maps
        egui::CentralPanel::default().show(ui, |ui| {
            let Some((ims, bonds, vents, domains, dnb, size, _fe)) = meta else {
                return;
            };
            let avail = ui.available_size();
            let n = ims.len();
            let (cols, rows) = if n >= 3 { (2, 2) } else { (n.max(1), 1) };
            let cell = egui::vec2(avail.x / cols as f32, avail.y / rows as f32);
            let (resp, painter) = ui.allocate_painter(avail, egui::Sense::click_and_drag());
            if resp.hovered() {
                let sc = ui.input(|i| i.smooth_scroll_delta.y);
                if sc != 0.0 {
                    self.zoom = (self.zoom * (1.0 + sc * 0.002)).clamp(0.2, 200.0);
                }
            }
            if resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) {
                self.pan += resp.drag_delta();
            }
            let origin = resp.rect.min;
            let mut hovered = None;
            for (i, (label, w, h, axes, fixed)) in ims.iter().enumerate() {
                let slot = egui::Rect::from_min_size(origin + egui::vec2((i % cols) as f32 * cell.x, (i / cols) as f32 * cell.y), cell);
                let fit = ((slot.width() - 8.0) / *w as f32).min((slot.height() - 22.0) / *h as f32).max(0.01) * self.zoom;
                let img = egui::Rect::from_min_size(slot.min + egui::vec2(4.0, 18.0) + self.pan, egui::vec2(*w as f32 * fit, *h as f32 * fit));
                painter.text(slot.min + egui::vec2(4.0, 2.0), egui::Align2::LEFT_TOP, label, egui::FontId::proportional(13.0), egui::Color32::WHITE);
                let clip = painter.with_clip_rect(slot.shrink(1.0));
                if let Some(Some(t)) = self.tex.get(i) {
                    // flip y so that +y points up
                    clip.image(t.id(), img, egui::Rect::from_min_max(egui::pos2(0.0, 1.0), egui::pos2(1.0, 0.0)), egui::Color32::WHITE);
                }
                let to_screen = |x: f32, y: f32| egui::pos2(img.min.x + x * fit, img.max.y - y * fit);
                let in_img = dims >= 2 || dims == 3;
                // overlays (2D / slices)
                if in_img && dims >= 2 {
                    let (ax, ay) = *axes;
                    if self.overlays.0 && fit >= 4.0 && !bonds.is_empty() && dims == 2 {
                        for y in 0..*h {
                            for x in 0..*w {
                                let c = x + y * *w;
                                let b = bonds[c];
                                if b & 1 != 0 && x + 1 < *w {
                                    clip.line_segment([to_screen(x as f32 + 0.5, y as f32 + 0.5), to_screen(x as f32 + 1.5, y as f32 + 0.5)], egui::Stroke::new(1.5, egui::Color32::WHITE));
                                }
                                if b & 4 != 0 && y + 1 < *h {
                                    clip.line_segment([to_screen(x as f32 + 0.5, y as f32 + 0.5), to_screen(x as f32 + 0.5, y as f32 + 1.5)], egui::Stroke::new(1.5, egui::Color32::WHITE));
                                }
                            }
                        }
                    }
                    if self.overlays.1 {
                        for v in &vents {
                            let on = dims == 2 || (0..3).all(|a| a == ax || a == ay || v[a] == fixed[a]);
                            if on {
                                clip.circle_stroke(to_screen(v[ax] as f32 + 0.5, v[ay] as f32 + 0.5), (3.0f32).max(fit), egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 160, 0)));
                            }
                        }
                    }
                    if self.overlays.2 && !domains.is_empty() && dims == 2 {
                        let (bx, by) = (dnb[0], dnb[1]);
                        for j in 0..by {
                            for i2 in 0..bx {
                                let s = domains[i2 + bx * j];
                                if s == 0 {
                                    continue;
                                }
                                let x0 = (i2 * size[0] as usize / bx) as f32;
                                let x1 = ((i2 + 1) * size[0] as usize / bx) as f32;
                                let y0 = (j * size[1] as usize / by) as f32;
                                let y1 = ((j + 1) * size[1] as usize / by) as f32;
                                let col = if s > 0 { egui::Color32::from_rgb(255, 80, 80) } else { egui::Color32::from_rgb(80, 120, 255) };
                                clip.rect_stroke(egui::Rect::from_two_pos(to_screen(x0, y0), to_screen(x1, y1)), 0.0, egui::Stroke::new(1.0, col), egui::StrokeKind::Inside);
                            }
                        }
                    }
                }
                if let Some(pos) = resp.hover_pos() {
                    if img.contains(pos) {
                        let x = ((pos.x - img.min.x) / fit) as i64;
                        let y = ((img.max.y - pos.y) / fit) as i64;
                        let mut q = [fixed[0] as i64, fixed[1] as i64, fixed[2] as i64];
                        if dims >= 1 {
                            q[axes.0] = x;
                        }
                        if dims >= 2 {
                            q[axes.1] = y;
                        }
                        if dims == 0 {
                            q = [x + y * *w as i64, 0, 0];
                        }
                        if dims == 1 {
                            q = [x, 0, 0];
                        }
                        hovered = Some(q);
                    }
                }
            }
            self.hover = hovered;
            if let Some(q) = hovered {
                painter.text(resp.rect.right_bottom() - egui::vec2(6.0, 6.0), egui::Align2::RIGHT_BOTTOM, format!("{:?}", &q[..dims.max(1) as usize]), egui::FontId::monospace(12.0), egui::Color32::WHITE);
                if resp.clicked_by(egui::PointerButton::Primary) {
                    self.apply_tool(epoch, q, dims);
                }
            }
        });
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
            let mut hist = vec![];
            for m in Mode::all() {
                v.mode = m;
                let f = render::render(&sim.st, &sim.p, &v, &[], &mut hist);
                assert!(!f.images.is_empty());
                for im in &f.images {
                    assert_eq!(im.rgba.len(), im.w * im.h * 4);
                }
            }
            let c = (0..sim.st.n).find(|&c| sim.st.occupied(c)).unwrap();
            let ins = inspect(&sim, c, &HashMap::new());
            assert!(!ins.trace.is_empty(), "debugger trace for dims {dims}");
            assert!(ins.text.contains("lineage"));
            let empty = (0..sim.st.n).find(|&c| !sim.st.occupied(c));
            if let Some(e) = empty {
                inspect(&sim, e, &HashMap::new());
            }
        }
    }
}
