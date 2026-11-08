use primordia_core::{Config, CpuExec, Params, Sim, State, isa};
use primordia_core::params::*;
use primordia_telemetry::{Recorder, analysis, clusters::{Kind, Tracker}};

fn cfg() -> Config {
    let mut c = Config::preset("primordia-2d-sun").unwrap();
    c.world.size = vec![9, 9]; c.world.init_occupancy = 0.0;
    c.energy.sun.day_period = 4; c.telemetry.metrics_every = 3;
    c
}
#[test]
fn every_exported_minimal_tape_passes_its_own_assay() {
    let mut c = Config::preset("paper-0d").unwrap(); c.individual.step_budget = 256;
    let original = isa::parse_hex_genome("12 80 40 80 6f 15 13 12 13 15 c2 01 18 13 18").unwrap();
    for genome in [original, isa::MINIMAL_REPLICATOR.to_vec()] {
        let assay = analysis::assay_genome(&c, &genome, 1);
        assert!(assay.replicator);
        assert!(analysis::assay_genome(&c, &assay.minimal, 1).replicator);
    }
}
#[test]
fn nop_prefixed_kin_form_a_cluster() {
    let c = cfg(); let p = Params::new(&c); let mut st = State::empty(&p);
    for i in 0..st.n {
        *st.hm(i, H_FLAGS) = FL_OCC; *st.hm(i, H_LEN) = 5;
        let base = i * st.lmax; st.tapes[base..base + 5].copy_from_slice(&[0, 0x12, 0x13, 0x15, 0xc2]);
    }
    let (stats, _) = Tracker::default().observe(&st, &p);
    let kin = &stats.iter().find(|(k, _)| *k == Kind::Kin).unwrap().1;
    assert_eq!(kin.clusters, 1); assert_eq!(kin.largest, 81);
}
#[test]
fn births_use_their_epoch_and_late_records_are_reported() {
    let c = cfg(); let sim = Sim::new(c.clone(), Box::new(CpuExec)).unwrap();
    let mut rec = Recorder::new(None, &sim).unwrap(); let mut st = sim.st.clone();
    rec.record_births(&[[3, 1, 2, 4, 1, 0], [7, 1, 3, 5, 1, 0]]).unwrap();
    st.epoch = 4; rec.record_state(&st, &sim.p, &c).unwrap();
    assert_eq!(rec.summary.births_last_day, 1);
    assert!(rec.record_births(&[[3, 1, 4, 6, 1, 0]]).is_err());
    st.epoch = 8; rec.record_state(&st, &sim.p, &c).unwrap();
    assert_eq!(rec.summary.births_last_day, 1);
}
#[test]
fn day_boundaries_are_observed_between_metrics_samples() {
    let c = cfg(); let mut sim = Sim::new(c, Box::new(CpuExec)).unwrap();
    let mut rec = Recorder::new(None, &sim).unwrap();
    for _ in 0..4 { sim.step(); rec.record(&mut sim).unwrap(); }
    assert_eq!(rec.series["population at sunset"][0][0], 2.0);
    assert_eq!(rec.series["population at sunrise"][0][0], 4.0);
}

#[test]
fn live_day_changes_start_accounting_at_the_edit_epoch() {
    let mut c = cfg(); c.telemetry.metrics_every = 100;
    let mut sim = Sim::new(c, Box::new(CpuExec)).unwrap();
    let mut rec = Recorder::new(None, &sim).unwrap();
    for _ in 0..4 { sim.step(); rec.record(&mut sim).unwrap(); }
    sim.push_event(primordia_core::events::Event { epoch: 4, kind: "set".into(), params: serde_json::json!({"key":"energy.sun.day_period","value":6}) });
    sim.step(); rec.record(&mut sim).unwrap();
    // e=5 is neither a metric nor a day-boundary sample. The new interval
    // must start here, so its first epoch's birth is retained at e=6.
    rec.record_births(&[[4, 1, 2, 4, 1, 0]]).unwrap();
    sim.step(); rec.record(&mut sim).unwrap();
    assert_eq!(rec.summary.births_last_day, 1);
}

#[test]
fn a_cluster_split_has_only_one_heir_to_the_old_identity() {
    let mut c = cfg(); c.world.dims = 1; c.world.size = vec![153];
    let p = Params::new(&c); let mut st = State::empty(&p);
    for i in 0..101 {
        *st.hm(i, H_FLAGS) = FL_OCC; *st.hm(i, H_LEN) = 4;
        let at = i * st.lmax; st.tapes[at..at + 4].copy_from_slice(&isa::MINIMAL_REPLICATOR);
    }
    let mut tracker = Tracker::default(); let (_, before) = tracker.observe(&st, &p);
    let id = before.iter().find(|c| c.kind == Kind::Kin).unwrap().id;
    *st.hm(50, H_FLAGS) = 0; st.epoch = 1;
    let (_, after) = tracker.observe(&st, &p); let kin: Vec<_> = after.iter().filter(|c| c.kind == Kind::Kin).collect();
    assert_eq!(kin.len(), 2); assert_ne!(kin[0].id, kin[1].id); assert_eq!(kin.iter().filter(|c| c.id == id).count(), 1);
}

/// With resumed execution and a small step budget (M17: 128), the assay's
/// one slice still has room for the seeded 20-byte ancestor to copy itself.
#[test]
fn the_assay_sees_replicators_in_resume_worlds() {
    let cfg = Config::from_file(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../runs/live-m17.toml"))).unwrap();
    assert!(cfg.individual.resume && cfg.individual.step_budget <= 128);
    let ancestor = isa::parse_hex_genome("42 0a 17 1c 46 08 18 6a 1c 67 1c 12 13 15 06 01 0d c5 02 d2").unwrap();
    let r = analysis::assay_genome(&cfg, &ancestor, 1);
    assert!(r.replicator, "identity {}", r.identity);
}

/// Unwritten newborns (empty tapes) in a body are neither a cell type nor a role.
#[test]
fn empty_newborns_are_not_a_role() {
    let mut c = Config::preset("primordia-2d-sun").unwrap();
    c.world.size = vec![30, 30]; c.world.init_occupancy = 0.0;
    let p = Params::new(&c); let mut st = State::init(&c, &p);
    let copier = [0x12u8, 0x13, 0x15, 0xC2, 0x40, 0x41, 0x42, 0x43];
    for x in 0..15usize {
        let cell = x + 30 * 5;
        let mut fl = FL_OCC;
        if x > 0 { fl |= 1 << (BOND_SHIFT + 1); }
        if x < 14 { fl |= 1 << BOND_SHIFT; }
        st.hdr[cell * HDR + H_FLAGS] = fl; st.hdr[cell * HDR + H_LEN] = 8;
        if x < 10 { st.tapes[cell * st.lmax..cell * st.lmax + 8].copy_from_slice(&copier); }
    }
    let bodies = primordia_core::phases::components(&st, &p);
    let bt = primordia_telemetry::body_types(&st, &p, &bodies);
    assert_eq!((bt.bodies, bt.types_max, bt.roles_max), (1, 1, 1));
}
