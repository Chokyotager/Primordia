//! `primordia` command line (§15.3).

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand};
use primordia_core::config::Config;
use primordia_core::events::{Event, read_events};
use primordia_core::params::Params;
use primordia_core::state::State;
use primordia_core::world::config_hash;
use primordia_core::{CpuExec, Sim, isa, phases};
use primordia_telemetry::{Recorder, Summary};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "primordia", about = "Open-ended artificial life simulator")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run one world (viewer unless --headless).
    Run {
        /// Config file or preset name
        config: String,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long)]
        headless: bool,
        #[arg(long)]
        epochs: Option<u64>,
        /// Run directory (default runs/<name>-s<seed>)
        #[arg(long)]
        out: Option<PathBuf>,
        /// Use the CPU golden model for execution
        #[arg(long)]
        cpu: bool,
        /// Dotted overrides, e.g. --set world.size=[99,99]
        #[arg(long = "set")]
        set: Vec<String>,
        /// Print progress every N epochs
        #[arg(long, default_value_t = 100)]
        progress: u64,
        /// Intervention log to apply (JSON lines {epoch, kind, params})
        #[arg(long)]
        events: Option<PathBuf>,
        /// Start the viewer running instead of paused
        #[arg(long)]
        autorun: bool,
        /// Serve the read-only three.js viewer on this address, e.g. 0.0.0.0:8765
        #[arg(long)]
        web: Option<String>,
        /// Run phase 6 only on the GPU (state copied every epoch) instead of fully resident
        #[arg(long)]
        exec_only: bool,
        /// Leave this fraction of GPU time idle (viewer only), e.g. 0.2, so a browser
        /// or desktop on the same machine stays smooth
        #[arg(long, default_value_t = 0.0)]
        gpu_headroom: f32,
        /// Continue from a snapshot (<run>/snapshots/e*.psnap); `config` is then ignored
        /// in favour of that run's config.toml (plus any --set overrides)
        #[arg(long)]
        resume: Option<PathBuf>,
    },
    /// Grid sweep over config keys x seeds.
    Batch {
        sweep: PathBuf,
        #[arg(long)]
        cpu: bool,
    },
    /// Re-run from config + seed + events and verify state hashes.
    Replay {
        run_dir: PathBuf,
        #[arg(long)]
        to: Option<u64>,
        #[arg(long)]
        cpu: bool,
    },
    /// New run from a snapshot with config changes.
    Branch {
        snapshot: PathBuf,
        diff: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long)]
        epochs: Option<u64>,
        #[arg(long)]
        cpu: bool,
    },
    /// Mirror-twin check (§11.4).
    Twin {
        config: String,
        #[arg(long)]
        epochs: Option<u64>,
        #[arg(long = "set")]
        set: Vec<String>,
        #[arg(long)]
        cpu: bool,
    },
    /// Run a genome through the replicator assay.
    Assay {
        genome: String,
        #[arg(long)]
        config: Option<String>,
    },
    /// Benchmark per phase and per epoch.
    Bench {
        #[arg(long, default_value_t = 20)]
        epochs: u64,
        #[arg(long)]
        only: Option<String>,
        /// Benchmark a snapshot instead (uses the run directory's config.toml)
        #[arg(long)]
        snapshot: Option<PathBuf>,
        /// Benchmark the exec-only GPU mode (physics on the CPU) with a per-phase breakdown
        #[arg(long)]
        exec_only: bool,
    },
    /// List built-in presets.
    Presets,
}

fn load_config(spec: &str, seed: Option<u64>, sets: &[String]) -> Result<Config> {
    let mut c = if Path::new(spec).exists() {
        Config::from_file(Path::new(spec))?
    } else {
        Config::preset(spec).with_context(|| format!("{spec} is neither a file nor a preset"))?
    };
    if let Some(s) = seed {
        c.seed = s;
    }
    for kv in sets {
        let (k, v) = kv.split_once('=').context("--set key=value")?;
        let val: toml::Value = toml::from_str(&format!("x = {v}")).map(|t: toml::Table| t["x"].clone()).unwrap_or(toml::Value::String(v.to_string()));
        c = c.with_override(k, val)?;
    }
    c.validate()?;
    Ok(c)
}

fn backend(p: &Params, cpu: bool) -> (Box<dyn primordia_core::ExecBackend>, String) {
    let b = primordia_gpu::backend(p, !cpu);
    let n = b.name();
    (b, n)
}

fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

pub struct RunDir {
    pub dir: PathBuf,
    hashes: std::fs::File,
    events: std::fs::File,
}

impl RunDir {
    pub fn create(dir: &Path, cfg: &Config, adapter: &str, parent: Option<&str>) -> Result<RunDir> {
        std::fs::create_dir_all(dir.join("snapshots"))?;
        std::fs::write(dir.join("config.toml"), cfg.to_toml())?;
        let manifest = serde_json::json!({
            "run_id": format!("{}-{:016x}", dir.file_name().unwrap().to_string_lossy(), config_hash(cfg)),
            "parent": parent,
            "git_commit": git_commit(),
            "gpu_adapter": adapter,
            "config_hash": format!("{:016x}", config_hash(cfg)),
            "seed": cfg.seed,
            "bias": !cfg.chirality.bias.is_zero(),
        });
        std::fs::write(dir.join("manifest.json"), serde_json::to_string_pretty(&manifest)?)?;
        let mut hashes = std::fs::File::create(dir.join("hashes.csv"))?;
        writeln!(hashes, "epoch,hash")?;
        let events = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("events.jsonl"))?;
        Ok(RunDir { dir: dir.to_path_buf(), hashes, events })
    }
    pub fn log_event(&mut self, e: &Event) -> Result<()> {
        writeln!(self.events, "{}", serde_json::to_string(e)?)?;
        Ok(())
    }
    pub fn snapshot_state(&self, st: &State, cfg: &Config) -> Result<PathBuf> {
        let path = self.dir.join("snapshots").join(format!("e{:010}.psnap", st.epoch));
        st.write_snapshot(&path, config_hash(cfg))?;
        Ok(path)
    }
}

impl primordia_view::observer::Sink for RunDir {
    fn event(&mut self, e: &Event) -> Result<()> {
        self.log_event(e)
    }
    fn state(&mut self, st: &State, cfg: &Config) -> Result<()> {
        let e = st.epoch;
        let t = &cfg.telemetry;
        if t.hash_every > 0 && e % t.hash_every == 0 {
            writeln!(self.hashes, "{},{:016x}", e, st.hash())?;
            self.hashes.flush()?;
        }
        if t.snapshot_every > 0 && e % t.snapshot_every == 0 {
            self.snapshot_state(st, cfg)?;
        }
        Ok(())
    }
    fn save_now(&mut self, st: &State, cfg: &Config) -> Result<PathBuf> {
        self.snapshot_state(st, cfg)
    }
}

fn print_summary(s: &Summary) {
    println!(
        "epoch {:>7}  pop {:>8}  len {:5.1}  HOE {:6.3}  EE code {:+.3} chem {:+.3} mono {:+.3}  repl {:>3}  species {:>4}  bodies {:>5} (max {})  ledger viol {}",
        s.epoch, s.occupied, s.mean_len, s.hoe, s.ee_code, s.ee_chem, s.ee_mono, s.replicators, s.species, s.bodies, s.max_body, s.ledger_violations
    );
}

/// Headless run loop shared by run, branch and batch. The GPU is kept busy
/// on this thread; telemetry and run-directory output happen on the observer
/// thread from asynchronous snapshots.
fn run_headless(sim: &mut Sim, rd: RunDir, epochs: u64, progress: u64) -> Result<Summary> {
    run_headless_feed(sim, rd, epochs, progress, None, None)
}

/// Carry a parent run's plot history into a resumed run (best effort).
fn seed_history(rec: &mut Recorder, parent: Option<&std::path::Path>, upto: u64) {
    let Some(path) = parent.map(|d| d.join("history.json")).filter(|p| p.exists()) else { return };
    match rec.load_history(&path, upto) {
        Ok(()) => println!("plot history carried over from {}", path.display()),
        Err(e) => eprintln!("warning: could not load {}: {e:#}", path.display()),
    }
}

fn run_headless_feed(sim: &mut Sim, rd: RunDir, epochs: u64, progress: u64, feed: Option<std::sync::Arc<primordia_view::web::Feed>>, parent: Option<&std::path::Path>) -> Result<Summary> {
    use primordia_view::observer::{Msg, Observer, Want};
    let mut rec = Recorder::new(Some(&rd.dir), sim)?;
    seed_history(&mut rec, parent, sim.st.epoch);
    let events = sim.events.clone();
    let view = primordia_view::ViewSettings::default(&sim.p);
    let mut obs = Observer::spawn(sim.cfg.clone(), sim.p.clone(), rec, Some(Box::new(rd)), feed, view, false);
    for e in events {
        obs.send(Msg::Event(e));
    }
    let t0 = Instant::now();
    let start = sim.st.epoch;
    let backend = sim.backend.name();
    let mut wants: std::collections::BTreeMap<u64, Want> = Default::default();
    let deliver = |sim: &mut Sim, obs: &Observer, wants: &mut std::collections::BTreeMap<u64, Want>, wait: bool| {
        let rate = (sim.st.epoch - start) as f64 / t0.elapsed().as_secs_f64().max(1e-9);
        for st in sim.take_snapshots(wait) {
            let w = wants.remove(&st.epoch).unwrap_or_default();
            obs.deliver(st, w, rate, backend.clone());
        }
    };
    while sim.st.epoch < start + epochs {
        sim.step();
        let b = sim.take_births();
        if !b.is_empty() {
            obs.send(Msg::Births(b));
        }
        let mut w = obs.want(sim.st.epoch, false);
        if sim.st.epoch == start + epochs && sim.cfg.telemetry.snapshot_every > 0 {
            w.save = true;
        }
        if w.telemetry || w.sink || w.web || w.save {
            sim.request_snapshot();
            wants.insert(sim.st.epoch, w);
        }
        deliver(sim, &obs, &mut wants, false);
        for r in sim.rejected.drain(..) {
            eprintln!("warning: {r}");
        }
        if progress > 0 && sim.st.epoch % progress == 0 {
            let rate = (sim.st.epoch - start) as f64 / t0.elapsed().as_secs_f64();
            print!("[{:6.1} ep/s] ", rate);
            print_summary(&obs.shared.lock().unwrap().summary);
        }
    }
    deliver(sim, &obs, &mut wants, true);
    let b = sim.take_births();
    obs.send(Msg::Births(b));
    sim.sync();
    let mut s = obs.finish();
    s.ledger_violations = sim.st.ledger.violations;
    s.energy_residual = sim.st.ledger.energy_residual;
    s.matter_residual = sim.st.ledger.matter_residual;
    s.atom_residual = sim.st.ledger.atom_residual;
    Ok(s)
}

fn default_dir(cfg: &Config, spec: &str) -> PathBuf {
    let name = Path::new(spec).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "run".into());
    PathBuf::from("runs").join(format!("{name}-s{}", cfg.seed))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Presets => {
            for (n, _) in primordia_core::config::PRESETS {
                println!("{n}");
            }
        }
        Cmd::Run { config, seed, headless, epochs, out, cpu, set, progress, events, autorun, web, exec_only, resume, gpu_headroom } => {
            let parent_dir = resume.as_ref().map(|sp| sp.parent().and_then(|d| d.parent()).map(|d| d.to_path_buf()).context("snapshot must live in <run>/snapshots/")).transpose()?;
            let cfg = match &parent_dir {
                Some(d) => load_config(&d.join("config.toml").display().to_string(), seed, &set)?,
                None => load_config(&config, seed, &set)?,
            };
            let epochs = epochs.unwrap_or(cfg.epochs);
            let dir = out.unwrap_or_else(|| default_dir(&cfg, &config));
            let p = Params::new(&cfg);
            let be = primordia_gpu::backend_mode(&p, !(cpu || !cfg.gpu.exec), !exec_only);
            let name = be.name();
            println!("primordia run: {} cells, dims {}, backend {name}, out {}", p.geom.n, p.geom.dims, dir.display());
            let mut rd = RunDir::create(&dir, &cfg, &name, None)?;
            let st0 = match &resume {
                Some(sp) => {
                    let (mut st, _) = State::read_snapshot(sp, &p)?;
                    st.reset_ledger(&p);
                    println!("resuming from {} at epoch {}", sp.display(), st.epoch);
                    st
                }
                None => State::init(&cfg, &p),
            };
            let start_epoch = st0.epoch;
            let mut sim = Sim::from_state(cfg.clone(), p.clone(), st0, be)?;
            let mut evs = match (&events, &parent_dir) {
                (Some(ev), _) => read_events(ev)?,
                (None, Some(d)) => read_events(&d.join("events.jsonl"))?,
                _ => vec![],
            };
            evs.retain(|e| e.epoch >= start_epoch);
            for e in evs {
                sim.push_event(e);
            }
            if headless {
                let feed = match &web {
                    Some(addr) => {
                        let f = primordia_view::web::Feed::new();
                        primordia_view::web::serve(addr, f.clone())?;
                        println!("read-only browser viewer on http://{addr}");
                        Some(f)
                    }
                    None => None,
                };
                let s = run_headless_feed(&mut sim, rd, epochs, progress, feed, parent_dir.as_deref())?;
                print_summary(&s);
                println!("{}", serde_json::to_string_pretty(&s)?);
            } else {
                let mut rec = Recorder::new(Some(&dir), &sim)?;
                seed_history(&mut rec, parent_dir.as_deref(), start_epoch);
                for e in &sim.events {
                    rd.log_event(e)?;
                }
                let rd = rd;
                primordia_view::run_viewer(sim, rec, Some(Box::new(rd)), primordia_view::ViewerOptions { autorun, web, gpu_headroom })?;
            }
        }
        Cmd::Replay { run_dir, to, cpu } => replay(&run_dir, to, cpu)?,
        Cmd::Branch { snapshot, diff, out, epochs, cpu } => {
            let parent_dir = snapshot.parent().and_then(|p| p.parent()).context("snapshot must live in <run>/snapshots/")?;
            let base: toml::Value = toml::from_str(&std::fs::read_to_string(parent_dir.join("config.toml"))?)?;
            let d: toml::Value = toml::from_str(&std::fs::read_to_string(&diff)?)?;
            let mut merged = base.clone();
            merge(&mut merged, &d);
            let cfg: Config = merged.try_into()?;
            cfg.validate()?;
            let p = Params::new(&cfg);
            let (st, _) = State::read_snapshot(&snapshot, &p)?;
            let dir = out.unwrap_or_else(|| PathBuf::from(format!("{}-branch-e{}", parent_dir.display(), st.epoch)));
            let (be, name) = backend(&p, cpu || !cfg.gpu.exec);
            let rd = RunDir::create(&dir, &cfg, &name, Some(&snapshot.display().to_string()))?;
            std::fs::write(dir.join("branch.json"), serde_json::to_string_pretty(&serde_json::json!({"snapshot": snapshot, "diff": d.to_string(), "start_epoch": st.epoch}))?)?;
            let mut sim = Sim::from_state(cfg.clone(), p, st, be)?;
            sim.st.reset_ledger(&sim.p.clone());
            let s = run_headless(&mut sim, rd, epochs.unwrap_or(cfg.epochs), 100)?;
            print_summary(&s);
        }
        Cmd::Twin { config, epochs, set, cpu } => twin(&config, epochs, &set, cpu)?,
        Cmd::Assay { genome, config } => {
            let txt = if Path::new(&genome).exists() { std::fs::read_to_string(&genome)? } else { genome.clone() };
            let g = isa::parse_hex_genome(&txt)?;
            let cfg = load_config(config.as_deref().unwrap_or("paper-0d"), None, &[])?;
            let r = primordia_telemetry::analysis::assay_genome(&cfg, &g, 1);
            println!("genome   : {}", isa::to_hex(&g));
            println!("disasm   : {}", g.iter().map(|&b| isa::mnemonic(b)).collect::<Vec<_>>().join(" ; "));
            println!("identity : {:.3}", r.identity);
            println!("replicator: {}", r.replicator);
            if r.replicator {
                println!("minimal  : {}  ({} bytes)", isa::to_hex(&r.minimal), r.minimal.len());
            }
        }
        Cmd::Bench { epochs, only, snapshot, exec_only } => bench(epochs, only.as_deref(), snapshot.as_deref(), exec_only)?,
        Cmd::Batch { sweep, cpu } => batch(&sweep, cpu)?,
    }
    Ok(())
}

fn merge(base: &mut toml::Value, over: &toml::Value) {
    match (base, over) {
        (toml::Value::Table(b), toml::Value::Table(o)) => {
            for (k, v) in o {
                match b.get_mut(k) {
                    Some(bv) if bv.is_table() && v.is_table() => merge(bv, v),
                    _ => {
                        b.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (b, o) => *b = o.clone(),
    }
}

fn replay(run_dir: &Path, to: Option<u64>, cpu: bool) -> Result<()> {
    let cfg = Config::from_file(&run_dir.join("config.toml"))?;
    let events = read_events(&run_dir.join("events.jsonl"))?;
    let hashes: Vec<(u64, String)> = std::fs::read_to_string(run_dir.join("hashes.csv"))?
        .lines()
        .skip(1)
        .filter_map(|l| l.split_once(',').map(|(a, b)| (a.parse().unwrap(), b.to_string())))
        .collect();
    let p = Params::new(&cfg);
    let branch = run_dir.join("branch.json");
    let st = if branch.exists() {
        let b: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&branch)?)?;
        let snap = PathBuf::from(b["snapshot"].as_str().unwrap());
        let (mut st, _) = State::read_snapshot(&snap, &p)?;
        st.reset_ledger(&p);
        st
    } else {
        State::init(&cfg, &p)
    };
    let (be, name) = backend(&p, cpu || !cfg.gpu.exec);
    let mut sim = Sim::from_state(cfg, p, st, be)?;
    for e in events {
        sim.push_event(e);
    }
    let last = to.unwrap_or(hashes.last().map(|h| h.0).unwrap_or(0));
    println!("replaying {} to epoch {last} on {name}: {} hash checkpoints", run_dir.display(), hashes.iter().filter(|h| h.0 <= last).count());
    let mut checked = 0;
    for (e, h) in hashes.iter().filter(|h| h.0 <= last) {
        while sim.st.epoch < *e {
            sim.step();
        }
        sim.sync();
        let got = format!("{:016x}", sim.st.hash());
        if &got != h {
            bail!("replay diverged at epoch {e}: recorded {h}, replayed {got}");
        }
        checked += 1;
    }
    println!("replay OK: {checked} hashes identical");
    Ok(())
}

fn twin(spec: &str, epochs: Option<u64>, sets: &[String], cpu: bool) -> Result<()> {
    let cfg = load_config(spec, None, sets)?;
    ensure!(cfg.chirality.bias.is_zero(), "twin check requires all bias = 0 (or a mirrored bias twin)");
    let epochs = epochs.unwrap_or(cfg.epochs);
    let every = cfg.telemetry.hash_every.max(1);
    let p = Params::new(&cfg);
    let a0 = State::init(&cfg, &p);
    let b0 = a0.mirror(&p);
    let (ba, name) = backend(&p, cpu || !cfg.gpu.exec);
    let (bb, _) = backend(&p, cpu || !cfg.gpu.exec);
    let mut a = Sim::from_state(cfg.clone(), p.clone(), a0, ba)?;
    let mut b = Sim::from_state(cfg.clone(), p.clone(), b0, bb)?;
    println!("twin check: {} cells, {epochs} epochs, check every {every}, backend {name}", p.geom.n);
    let mut good = (a.st.clone(), b.st.clone());
    let t0 = Instant::now();
    while a.st.epoch < epochs {
        a.step();
        b.step();
        if a.st.epoch % every == 0 || a.st.epoch == epochs {
            a.sync();
            b.sync();
            let ha = a.st.mirror(&p).hash();
            let hb = b.st.hash();
            if ha != hb {
                println!("TWIN DIVERGENCE between epochs {} and {}: bisecting on CPU", good.0.epoch, a.st.epoch);
                bisect(&cfg, &p, good.0.clone(), good.1.clone())?;
                bail!("twin check failed");
            }
            good = (a.st.clone(), b.st.clone());
            let occ = (0..a.st.n).filter(|&c| a.st.occupied(c)).count();
            println!("epoch {:>7}: hash(M(A)) == hash(B) = {:016x}   pop {}   ({:.1} ep/s)", a.st.epoch, hb, occ, a.st.epoch as f64 / t0.elapsed().as_secs_f64());
        }
    }
    println!("twin OK: {epochs} epochs, exact mirror at every check");
    Ok(())
}

fn bisect(cfg: &Config, p: &Params, a: State, b: State) -> Result<()> {
    let mut a = Sim::from_state(cfg.clone(), p.clone(), a, Box::new(CpuExec))?;
    let mut b = Sim::from_state(cfg.clone(), p.clone(), b, Box::new(CpuExec))?;
    for _ in 0..10_000 {
        let e = a.st.epoch;
        let check = |a: &Sim, b: &Sim, phase: &str| -> Option<String> {
            a.st.mirror(p).first_diff(&b.st).map(|(c, what)| {
                let g = &p.geom;
                format!("first divergence at epoch {e}, phase {phase}: cell {c} ({:?}) {what}\n  A[M(c)] hdr {:?}\n  B[c]    hdr {:?}", g.coords(c), &a.st.hdr[g.mirror_cell(c) * 9..g.mirror_cell(c) * 9 + 9], &b.st.hdr[c * 9..c * 9 + 9])
            })
        };
        macro_rules! ph {
            ($name:expr, $f:expr) => {
                $f(&mut a);
                $f(&mut b);
                if let Some(m) = check(&a, &b, $name) {
                    println!("{m}");
                    return Ok(());
                }
            };
        }
        ph!("sources", |s: &mut Sim| phases::sources(&mut s.st, &s.p));
        ph!("transport", |s: &mut Sim| phases::transport(&mut s.st, &s.p));
        ph!("chemistry", |s: &mut Sim| phases::chemistry(&mut s.st, &s.p));
        ph!("absorb", |s: &mut Sim| phases::absorb_upkeep(&mut s.st, &s.p));
        ph!("exec", |s: &mut Sim| primordia_core::exec::exec_phase_cpu(&mut s.st, &s.p));
        ph!("bodies", |s: &mut Sim| {
            phases::collect_site_counters(&mut s.st);
            phases::body_move(&mut s.st, &s.p)
        });
        ph!("mutate", |s: &mut Sim| {
            let m = phases::mutation_map(&s.st, &s.p, s.map_image.as_ref());
            phases::mutate_decay(&mut s.st, &s.p, m.as_ref())
        });
        a.st.epoch += 1;
        b.st.epoch += 1;
    }
    println!("no divergence found on the CPU golden model (GPU-only divergence?)");
    Ok(())
}

fn bench(epochs: u64, only: Option<&str>, snapshot: Option<&Path>, exec_only: bool) -> Result<()> {
    let snap_cfg = match snapshot {
        Some(sp) => {
            let dir = sp.parent().and_then(|d| d.parent()).context("snapshot must live in <run>/snapshots/")?;
            Some(Config::from_file(&dir.join("config.toml"))?)
        }
        None => None,
    };
    let cases: Vec<(&str, Config)> = if let Some(c) = snap_cfg { vec![("snapshot", c)] } else { vec![
        ("2d-1M (1023^2, full physics)", Config::preset("primordia-2d-sun")?),
        ("2d-250k (501^2, full physics)", {
            let mut c = Config::preset("primordia-2d-sun")?;
            c.world.size = vec![501, 501];
            c
        }),
        ("3d-970k (99^3, full physics)", Config::preset("primordia-3d")?),
        ("paper-0d (2^17 x 64B)", Config::preset("paper-0d")?),
    ] };
    for (name, cfg) in cases {
        if let Some(o) = only {
            if !name.contains(o) {
                continue;
            }
        }
        let p = Params::new(&cfg);
        let st = match snapshot {
            Some(sp) => {
                let (mut st, _) = State::read_snapshot(sp, &p)?;
                st.reset_ledger(&p);
                st
            }
            None => State::init(&cfg, &p),
        };
        let be = primordia_gpu::backend_mode(&p, true, !exec_only);
        let bname = be.name();
        let mut sim = Sim::from_state(cfg, p, st, be)?;
        sim.step(); // warm-up
        if sim.backend.resident() {
            let t0 = Instant::now();
            for _ in 0..epochs {
                sim.step();
            }
            let total = t0.elapsed().as_secs_f64();
            println!("\n== {name}  [{bname}]  {:.2} epochs/s ({:.1} ms/epoch, whole epoch on the GPU)", epochs as f64 / total, 1000.0 * total / epochs as f64);
            if let Some(gt) = sim.backend.timing() {
                let k = (epochs + 1) as f64;
                println!("   gpu: submit+kernels {:.2} ms, tally read-back {:.2} ms per epoch", 1000.0 * gt[1] / k, 1000.0 * gt[2] / k);
            }
            continue;
        }
        let mut t = [0f64; 9];
        let names = ["interventions+sources", "transport", "chemistry", "absorb/upkeep", "exec (backend)", "site counters", "bodies", "mutation/decay", "ledger"];
        let t0 = Instant::now();
        for _ in 0..epochs {
            let s = &mut sim;
            let mut lap = Instant::now();
            let mut mark = |i: usize, lap: &mut Instant| {
                t[i] += lap.elapsed().as_secs_f64();
                *lap = Instant::now();
            };
            phases::sources(&mut s.st, &s.p);
            mark(0, &mut lap);
            phases::transport(&mut s.st, &s.p);
            mark(1, &mut lap);
            phases::chemistry(&mut s.st, &s.p);
            mark(2, &mut lap);
            phases::absorb_upkeep(&mut s.st, &s.p);
            mark(3, &mut lap);
            let p = s.p.clone();
            s.backend.exec(&mut s.st, &p);
            mark(4, &mut lap);
            phases::collect_site_counters(&mut s.st);
            mark(5, &mut lap);
            phases::body_move(&mut s.st, &s.p);
            mark(6, &mut lap);
            let m = phases::mutation_map(&s.st, &s.p, None);
            phases::mutate_decay(&mut s.st, &s.p, m.as_ref());
            mark(7, &mut lap);
            phases::ledger_check(&mut s.st, &s.p);
            s.st.tally = Default::default();
            mark(8, &mut lap);
            s.st.epoch += 1;
        }
        let total = t0.elapsed().as_secs_f64();
        println!("\n== {name}  [{bname}]  {:.2} epochs/s", epochs as f64 / total);
        for (i, n) in names.iter().enumerate() {
            println!("   {:<24} {:8.2} ms/epoch", n, 1000.0 * t[i] / epochs as f64);
        }
        if let Some(gt) = sim.backend.timing() {
            let k = (epochs + 1) as f64;
            println!("   gpu: upload {:.2} ms, kernels {:.2} ms, download {:.2} ms per epoch", 1000.0 * gt[0] / k, 1000.0 * gt[1] / k, 1000.0 * gt[2] / k);
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ batch

fn batch(sweep: &Path, cpu: bool) -> Result<()> {
    let sw: toml::Table = toml::from_str(&std::fs::read_to_string(sweep)?)?;
    let base = sw.get("base").and_then(|v| v.as_str()).context("sweep needs base")?.to_string();
    let epochs = sw.get("epochs").and_then(|v| v.as_integer()).map(|x| x as u64);
    let seeds: Vec<u64> = match sw.get("seeds") {
        Some(toml::Value::Integer(n)) => (1..=*n as u64).collect(),
        Some(toml::Value::Array(a)) => a.iter().filter_map(|v| v.as_integer()).map(|x| x as u64).collect(),
        _ => vec![1],
    };
    let out = PathBuf::from(sw.get("out").and_then(|v| v.as_str()).unwrap_or("runs/batch"));
    let progress = sw.get("progress").and_then(|v| v.as_integer()).unwrap_or(0) as u64;
    let mut sets: Vec<String> = vec![];
    if let Some(toml::Value::Table(t)) = sw.get("set") {
        for (k, v) in t {
            sets.push(format!("{k}={v}"));
        }
    }
    let mut grid: Vec<(String, Vec<toml::Value>)> = vec![];
    if let Some(toml::Value::Table(t)) = sw.get("grid") {
        for (k, v) in t {
            grid.push((k.clone(), v.as_array().cloned().unwrap_or_else(|| vec![v.clone()])));
        }
    }
    let mut combos: Vec<Vec<(String, toml::Value)>> = vec![vec![]];
    for (k, vals) in &grid {
        combos = combos.into_iter().flat_map(|c| vals.iter().map(move |v| {
            let mut c2 = c.clone();
            c2.push((k.clone(), v.clone()));
            c2
        })).collect();
    }
    std::fs::create_dir_all(&out)?;
    let mut csv = std::fs::File::create(out.join("summary.csv"))?;
    writeln!(csv, "combo,seed,epoch,occupied,mean_len,hoe,ee_code,ee_chem,ee_mono,first_replicator,replicators,t_b,winning_hand,species,bodies,organisms,ledger_violations,seconds")?;
    let mut rows = vec![];
    for (ci, combo) in combos.iter().enumerate() {
        let label = if combo.is_empty() { "base".to_string() } else { combo.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(";") };
        for &seed in &seeds {
            let mut all = sets.clone();
            all.extend(combo.iter().map(|(k, v)| format!("{k}={v}")));
            let mut cfg = load_config(&base, Some(seed), &all)?;
            if let Some(e) = epochs {
                cfg.epochs = e;
            }
            let dir = out.join(format!("c{ci}-s{seed}"));
            let p = Params::new(&cfg);
            let (be, name) = backend(&p, cpu || !cfg.gpu.exec);
            let rd = RunDir::create(&dir, &cfg, &name, None)?;
            let mut sim = Sim::from_state(cfg.clone(), p.clone(), State::init(&cfg, &p), be)?;
            let t0 = Instant::now();
            let s = run_headless(&mut sim, rd, cfg.epochs, progress)?;
            let secs = t0.elapsed().as_secs_f64();
            println!(
                "[{label}] seed {seed:>3}: HOE {:6.3}  first-repl {:>7}  repl {:>3}  EE {:+.3}  t_b {:>7}  pop {:>7}  ({secs:.0}s)",
                s.hoe,
                s.first_replicator.map(|x| x.to_string()).unwrap_or("-".into()),
                s.replicators,
                s.ee_code,
                s.t_b.map(|x| x.to_string()).unwrap_or("-".into()),
                s.occupied
            );
            writeln!(
                csv,
                "\"{label}\",{seed},{},{},{:.3},{:.4},{:.4},{:.4},{:.4},{},{},{},{},{},{},{},{},{:.1}",
                s.epoch,
                s.occupied,
                s.mean_len,
                s.hoe,
                s.ee_code,
                s.ee_chem,
                s.ee_mono,
                s.first_replicator.map(|x| x.to_string()).unwrap_or_default(),
                s.replicators,
                s.t_b.map(|x| x.to_string()).unwrap_or_default(),
                s.winning_hand.clone().unwrap_or_default(),
                s.species,
                s.bodies,
                s.organisms,
                s.ledger_violations,
                secs
            )?;
            csv.flush()?;
            rows.push((label.clone(), s));
        }
    }
    // summary table per combo
    println!("\n{:<50} {:>6} {:>9} {:>10} {:>9} {:>8} {:>10}", "combo", "runs", "HOE>1", "first-repl", "mean|EE|", "L:D", "binom p");
    let mut labels: Vec<String> = rows.iter().map(|r| r.0.clone()).collect();
    labels.dedup();
    for l in labels {
        let rs: Vec<&Summary> = rows.iter().filter(|r| r.0 == l).map(|r| &r.1).collect();
        let n = rs.len();
        let trans = rs.iter().filter(|s| s.hoe > 1.0).count();
        let fr: Vec<u64> = rs.iter().filter_map(|s| s.first_replicator).collect();
        let fr_med = if fr.is_empty() { "-".to_string() } else {
            let mut f = fr.clone();
            f.sort();
            f[f.len() / 2].to_string()
        };
        let ee: f64 = rs.iter().map(|s| s.ee_code.abs()).sum::<f64>() / n as f64;
        let nl = rs.iter().filter(|s| s.winning_hand.as_deref() == Some("L")).count();
        let nd = rs.iter().filter(|s| s.winning_hand.as_deref() == Some("D")).count();
        let pval = binom_two_sided(nl, nl + nd);
        println!("{:<50} {:>6} {:>4}/{:<4} {:>10} {:>9.3} {:>4}:{:<3} {:>10.4}", l, n, trans, n, fr_med, ee, nl, nd, pval);
    }
    println!("\nwrote {}", out.join("summary.csv").display());
    Ok(())
}

/// Two-sided exact binomial test against p = 1/2.
fn binom_two_sided(k: usize, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let lnc = |n: usize, k: usize| -> f64 { (1..=k).map(|i| ((n - k + i) as f64).ln() - (i as f64).ln()).sum() };
    let pk = |i: usize| (lnc(n, i) - n as f64 * 2f64.ln()).exp();
    let obs = pk(k);
    (0..=n).map(pk).filter(|&p| p <= obs * (1.0 + 1e-9)).sum::<f64>().min(1.0)
}
