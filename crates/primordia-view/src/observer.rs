//! The observer thread: everything that only *looks* at the simulation
//! (telemetry, run-directory hashes and snapshots, the chronicle, viewer
//! frames, the browser feed) runs here on asynchronous state snapshots, so
//! the thread driving the GPU never waits for it.

use crate::render::{self, ViewSettings};
use crate::{Chronicler, Frame, Note, web};
use anyhow::Result;
use primordia_core::events::Event;
use primordia_core::world::Birth;
use primordia_core::{Config, Params, State};
use primordia_telemetry::{Recorder, Summary};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Where run artefacts go (implemented by the CLI's run directory).
pub trait Sink: Send {
    fn event(&mut self, e: &Event) -> Result<()>;
    /// Called for every observed state; writes hashes / periodic snapshots when due.
    fn state(&mut self, st: &State, cfg: &Config) -> Result<()>;
    fn save_now(&mut self, st: &State, cfg: &Config) -> Result<PathBuf>;
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Want {
    pub telemetry: bool,
    pub sink: bool,
    pub render: bool,
    pub web: bool,
    pub save: bool,
    pub branch: bool,
}

pub enum Msg {
    Births(Vec<Birth>),
    Event(Event),
    State(Box<State>, Want, f64, String),
    View(ViewSettings),
    Done,
}

/// What the viewer UI reads.
#[derive(Default)]
pub struct Shared {
    pub frame: Option<Arc<Frame>>,
    pub summary: Summary,
    pub series: BTreeMap<String, Vec<[f64; 2]>>,
    pub chronicle: Vec<Note>,
    pub top_genomes: Vec<String>,
    pub sun: f32,
    pub births: u64,
    pub deaths: u64,
    pub colonies: Vec<render::Colony>,
}

pub struct Observer {
    tx: SyncSender<Msg>,
    handle: Option<JoinHandle<Summary>>,
    pub shared: Arc<Mutex<Shared>>,
    /// The observer is idle and can take an optional (render / web) snapshot.
    pub idle: Arc<AtomicBool>,
    cadence: primordia_core::config::TelemetryCfg,
    feed: bool,
    last_render: Instant,
    last_web: Instant,
    pub render_every: Duration,
}

impl Observer {
    pub fn spawn(cfg: Config, p: Params, rec: Recorder, sink: Option<Box<dyn Sink>>, feed: Option<Arc<web::Feed>>, view: ViewSettings, render: bool) -> Observer {
        let (tx, rx) = sync_channel::<Msg>(3);
        let shared = Arc::new(Mutex::new(Shared::default()));
        let idle = Arc::new(AtomicBool::new(true));
        let has_feed = feed.is_some();
        let cadence = cfg.telemetry.clone();
        let (s2, i2) = (shared.clone(), idle.clone());
        let handle = std::thread::spawn(move || run(rx, cfg, p, rec, sink, feed, view, s2, i2, render));
        Observer { tx, handle: Some(handle), shared, idle, cadence, feed: has_feed, last_render: Instant::now() - Duration::from_secs(10), last_web: Instant::now() - Duration::from_secs(10), render_every: Duration::from_millis(80) }
    }

    pub fn send(&self, m: Msg) {
        let _ = self.tx.send(m);
    }

    /// What the stepping thread should snapshot after epoch `epoch`.
    pub fn want(&mut self, epoch: u64, render_dirty: bool) -> Want {
        let t = &self.cadence;
        let mut w = Want {
            telemetry: primordia_telemetry::needs_state(t, epoch),
            sink: (t.hash_every > 0 && epoch % t.hash_every == 0) || (t.snapshot_every > 0 && epoch % t.snapshot_every == 0),
            ..Default::default()
        };
        let idle = self.idle.load(Ordering::Acquire);
        if idle && render_dirty && self.last_render.elapsed() >= self.render_every {
            w.render = true;
            self.last_render = Instant::now();
        }
        if idle && self.feed && self.last_web.elapsed() >= Duration::from_millis(1500) {
            w.web = true;
            self.last_web = Instant::now();
        }
        w
    }

    /// Deliver a snapshot; optional work (render / web only) is dropped
    /// rather than stalling the simulation when the observer is busy.
    pub fn deliver(&self, st: State, w: Want, eps: f64, backend: String) {
        let required = w.telemetry || w.sink || w.save;
        if !required && !(w.render || w.web) {
            return;
        }
        if w.render || w.web {
            self.idle.store(false, Ordering::Release);
        }
        let m = Msg::State(Box::new(st), w, eps, backend);
        if required {
            let _ = self.tx.send(m);
        } else if let Err(TrySendError::Full(_)) = self.tx.try_send(m) {
            self.idle.store(true, Ordering::Release);
        }
    }

    pub fn finish(mut self) -> Summary {
        let _ = self.tx.send(Msg::Done);
        self.handle.take().map(|h| h.join().unwrap_or_default()).unwrap_or_default()
    }
}

#[allow(clippy::too_many_arguments)]
fn run(rx: Receiver<Msg>, cfg: Config, p: Params, mut rec: Recorder, mut sink: Option<Box<dyn Sink>>, feed: Option<Arc<web::Feed>>, mut view: ViewSettings, shared: Arc<Mutex<Shared>>, idle: Arc<AtomicBool>, render: bool) -> Summary {
    let mut chron = Chronicler::new();
    let mut ctx = render::RenderCtx::default();
    let note = |shared: &Arc<Mutex<Shared>>, icon: &'static str, text: String, epoch: u64| {
        let mut s = shared.lock().unwrap();
        s.chronicle.push(Note { epoch, icon, text });
        if s.chronicle.len() > 400 {
            s.chronicle.remove(0);
        }
    };
    note(&shared, "🌱", format!("World created: {} sites in {}D", p.geom.n, p.geom.dims), 0);
    while let Ok(m) = rx.recv() {
        match m {
            Msg::Done => break,
            Msg::View(v) => view = ViewSettings { params: view.params.clone(), ..v },
            Msg::Births(b) => {
                let _ = rec.record_births(&b);
            }
            Msg::Event(e) => {
                if let Some(s) = sink.as_mut() {
                    let _ = s.event(&e);
                }
                note(&shared, "✋", format!("Intervention: {} {}", e.kind, e.params), e.epoch);
            }
            Msg::State(st, w, eps, backend) => {
                rec.record_ledger(&st);
                if w.telemetry {
                    if let Err(e) = rec.record_state(&st, &p, &cfg) {
                        note(&shared, "⚠", format!("telemetry error: {e:#}"), st.epoch);
                    }
                    let mut notes = vec![];
                    chron.observe_state(&st, &p, &rec.summary, &mut notes);
                    if !notes.is_empty() {
                        shared.lock().unwrap().chronicle.extend(notes);
                    }
                }
                if let Some(s) = sink.as_mut() {
                    if w.sink {
                        if let Err(e) = s.state(&st, &cfg) {
                            note(&shared, "⚠", format!("run-dir error: {e:#}"), st.epoch);
                        }
                    }
                    if w.save {
                        match s.save_now(&st, &cfg) {
                            Ok(path) if w.branch => {
                                let diff = path.with_extension("diff.toml");
                                let _ = std::fs::write(&diff, "# config changes for the branch, e.g.\n# [mutation]\n# background = 1e-3\n");
                                note(&shared, "🌿", format!("Branch point saved. Edit {} then run: primordia branch {} {}", diff.display(), path.display(), diff.display()), st.epoch);
                            }
                            Ok(path) => note(&shared, "💾", format!("Snapshot written: {}", path.display()), st.epoch),
                            Err(e) => note(&shared, "⚠", format!("Snapshot failed: {e:#}"), st.epoch),
                        }
                    }
                }
                let sun = crate::sun_level_at(&p, st.epoch);
                let (infos, cols, of) = if w.render || w.web {
                    let infos = render::cell_infos(&st);
                    let (c, o) = render::colonies(&st, &p, &infos, &mut ctx.tracker);
                    (infos, c, o)
                } else {
                    (vec![], vec![], vec![])
                };
                if w.web {
                    if let Some(f) = &feed {
                        let chronicle: Vec<Note> = shared.lock().unwrap().chronicle.iter().rev().take(30).cloned().collect();
                        let (m, b) = web::payload_parts(&st, &p, &backend, &rec.summary, &rec.series, eps, sun, &chronicle, &infos, &cols, &of);
                        f.set(m, b);
                    }
                }
                if w.render && render {
                    let f = render::render_with(&st, &p, &view, &rec.last_domains, &mut ctx.hist1d, &infos, cols, of);
                    let mut s = shared.lock().unwrap();
                    s.colonies = f.colonies.clone();
                    s.frame = Some(Arc::new(f));
                    s.summary = rec.summary.clone();
                    s.series = rec.series.clone();
                    s.top_genomes = rec.last_assay.iter().filter(|a| a.replicator).map(|a| primordia_core::isa::to_hex(&a.minimal)).collect();
                    s.sun = sun;
                    s.births = st.tally.births;
                    s.deaths = st.tally.deaths;
                } else if w.telemetry {
                    let mut s = shared.lock().unwrap();
                    s.summary = rec.summary.clone();
                    s.series = rec.series.clone();
                }
                if w.render || w.web {
                    idle.store(true, Ordering::Release);
                }
            }
        }
    }
    rec.finish().unwrap_or_default()
}
