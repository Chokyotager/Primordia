use std::path::{Path, PathBuf};
use std::process::{Command, Output};
struct Temp(PathBuf);
impl Temp { fn new() -> Self {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let p = std::env::temp_dir().join(format!("primordia-cli-{}-{}", std::process::id(), N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
    std::fs::create_dir(&p).unwrap(); Self(p)
} }
impl Drop for Temp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_primordia")).args(args).env("RAYON_NUM_THREADS", "2").output().unwrap()
}
fn ok(args: &[&str]) -> Output {
    let o = cli(args); assert!(o.status.success(), "{args:?}\n{}\n{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr)); o
}
fn path(p: &Path) -> &str { p.to_str().unwrap() }
fn last_hash(dir: &Path) -> String { std::fs::read_to_string(dir.join("hashes.csv")).unwrap().lines().last().unwrap().to_owned() }
#[test]
fn resume_and_branch_use_effective_config_and_replay_without_parent() {
    let t = Temp::new(); let parent = t.0.join("parent"); let resumed = t.0.join("resumed"); let branch = t.0.join("branch");
    let events = t.0.join("events.jsonl");
    std::fs::write(&events, concat!(
        "{\"epoch\":0,\"kind\":\"seed\",\"params\":{\"center\":[4,4,0],\"radius\":0,\"genome\":\"6f\",\"energy\":10000}}\n",
        "{\"epoch\":0,\"kind\":\"set\",\"params\":{\"key\":\"energy.dissipation\",\"value\":0.5}}\n",
        "{\"epoch\":0,\"kind\":\"set\",\"params\":{\"key\":\"telemetry.hash_every\",\"value\":2}}\n"
    )).unwrap();
    ok(&["run", "primordia-2d-sun", "--cpu", "--headless", "--epochs", "10", "--out", path(&parent), "--events", path(&events),
        "--set", "world.size=[9,9]", "--set", "world.init_occupancy=0.0", "--set", "energy.sun.enabled=false", "--set", "energy.vents.enabled=false",
        "--set", "chemistry.enabled=false", "--set", "mutation.enabled=false", "--set", "decay.enabled=false", "--set", "energy.dissipation=0.1",
        "--set", "telemetry.hash_every=1", "--set", "telemetry.snapshot_every=5", "--progress", "0"]);
    let hashes = std::fs::read_to_string(parent.join("hashes.csv")).unwrap();
    assert_eq!(hashes.lines().skip(1).map(|l| l.split(',').next().unwrap()).collect::<Vec<_>>(), ["2", "4", "6", "8", "10"]);
    let snapshot = parent.join("snapshots/e0000000005.psnap");
    let effective = primordia_core::Config::from_file(&snapshot.with_extension("toml")).unwrap(); assert_eq!(effective.energy.dissipation, 0.5);
    ok(&["replay", path(&parent), "--cpu"]);
    ok(&["run", "ignored", "--cpu", "--headless", "--resume", path(&snapshot), "--epochs", "5", "--out", path(&resumed), "--progress", "0"]);
    assert_eq!(last_hash(&parent), last_hash(&resumed));
    // Legacy checkpoints have no config sidecar: recover the earlier set events.
    std::fs::remove_file(snapshot.with_extension("toml")).unwrap();
    let diff = t.0.join("diff.toml"); std::fs::write(&diff, "# unchanged rules\n").unwrap();
    ok(&["branch", path(&snapshot), path(&diff), "--cpu", "--epochs", "5", "--out", path(&branch)]);
    assert_eq!(last_hash(&parent), last_hash(&branch));
    // Replay origins are local copies, not references to the parent directory.
    std::fs::rename(&parent, t.0.join("parent-moved")).unwrap();
    ok(&["replay", path(&resumed), "--cpu"]); ok(&["replay", path(&branch), "--cpu"]);
    let before = std::fs::read(resumed.join("hashes.csv")).unwrap();
    assert!(!cli(&["run", "paper-0d", "--cpu", "--headless", "--epochs", "1", "--out", path(&resumed)]).status.success());
    assert_eq!(before, std::fs::read(resumed.join("hashes.csv")).unwrap());
}
#[test]
fn explicit_missing_events_fail_without_creating_a_run() {
    let t = Temp::new(); let out = t.0.join("out"); let missing = t.0.join("missing.jsonl");
    let o = cli(&["run", "paper-0d", "--cpu", "--headless", "--epochs", "1", "--out", path(&out), "--events", path(&missing), "--set", "world.size=[2]"]);
    assert!(!o.status.success()); assert!(!out.exists());
}
