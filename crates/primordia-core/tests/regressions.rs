//! Regression cases from the code audit. These assert model behavior as well
//! as implementation parity: shared CPU/GPU bugs must not pass unnoticed.
use primordia_core::{Config, CpuExec, Params, Sim, State, exec, fixed, isa, phases};
use primordia_core::config::PairOrder;
use primordia_core::events::{self, Event};
use primordia_core::params::*;

fn cfg() -> Config {
    let mut c = Config::preset("primordia-2d-sun").unwrap();
    c.world.size = vec![9, 9];
    c.world.init_occupancy = 0.0;
    c.individual.random_initial_facing = false;
    c.individual.pair_order = PairOrder::SelfFirst;
    c.individual.step_budget = 8;
    c.mutation.enabled = false;
    c.decay.enabled = false;
    c.energy.sun.enabled = false;
    c.energy.vents.enabled = false;
    c.energy.upkeep.base = 0.0;
    c.energy.upkeep.per_byte = 0.0;
    c.energy.absorb_alpha = 0.0;
    c
}
fn put(s: &mut State, p: &Params, at: usize, g: &[u8]) {
    let base = at * s.lmax;
    s.tapes[base..base + g.len()].copy_from_slice(g);
    *s.hm(at, H_LEN) = g.len() as u32;
    *s.hm(at, H_FLAGS) = FL_OCC | (p.orient.valid[2][0] << ORIENT_SHIFT);
    *s.hm(at, H_E) = 10_000;
}
fn run(s: &mut State, p: &Params, at: usize) -> usize {
    let flags = exec::flag_snapshot(s);
    let mut trace = vec![];
    exec::exec_slice(s, p, &flags, at, None, Some(&mut trace));
    trace.len()
}
fn tmp(name: &str) -> std::path::PathBuf {
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    std::env::temp_dir().join(format!("primordia-{}-{}-{name}", std::process::id(), SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)))
}

#[test]
fn ordinary_mutation_kernel_is_mirror_neutral_for_every_byte() {
    for a in 0..=255u8 {
        let mut ordinary = [0u32; 256];
        let mut reflected = [0u32; 256];
        for draw in 0..16 {
            ordinary[isa::mutate_bit(a, draw, false) as usize] += 1;
            reflected[isa::mirror(isa::mutate_bit(isa::mirror(a), draw, false)) as usize] += 1;
            assert_eq!(isa::mirror(isa::mutate_bit(a, draw, false)), isa::mutate_bit(isa::mirror(a), draw, true));
        }
        assert_eq!(ordinary, reflected, "ordinary-world transition law for {a:02x}");
    }
}

#[test]
fn poisson_distribution_across_overflow_underflow_and_cap() {
    let table = fixed::poisson_exp_table();
    for lambda in [4.0f64, 4.01, 8.0, 10.0, 20.0, 22.0, 23.0, 30.0, 63.0, 64.0, 100.0, 128.0, 255.0] {
        let mut prob = (-lambda).exp();
        let mut cdf = 0.0;
        let mut expected = 0.0;
        let mut second = 0.0;
        let mut expected_cdf = [0.0; 64];
        for k in 0..64 {
            cdf += prob;
            expected += k as f64 * prob;
            second += (k * k) as f64 * prob;
            expected_cdf[k] = cdf;
            prob *= lambda / (k + 1) as f64;
        }
        expected += 64.0 * (1.0 - cdf);
        second += 4096.0 * (1.0 - cdf);
        let mut hist = [0usize; 65];
        let n = 100_000;
        for i in 0..n {
            let u = primordia_core::rng::philox_u32(1, 2, [i as u32, 0, 0, 0]);
            hist[fixed::poisson24(&table, (lambda * 16777216.0) as u32, u, 64) as usize] += 1;
        }
        let mean = hist.iter().enumerate().map(|(k, n)| k as f64 * *n as f64).sum::<f64>() / n as f64;
        let variance = hist.iter().enumerate().map(|(k, n)| (k as f64 - mean).powi(2) * *n as f64).sum::<f64>() / n as f64;
        assert!((mean - expected).abs() < 0.08, "lambda {lambda}: mean {mean}, expected {expected}");
        assert!((variance - (second - expected * expected)).abs() < 0.03 * lambda.max(1.0), "lambda {lambda}: variance {variance}");
        let mut count = 0;
        for k in 0..64 { count += hist[k]; assert!((count as f64 / n as f64 - expected_cdf[k]).abs() < 0.01); }
    }
    assert_eq!(fixed::poisson24(&table, 0, u32::MAX, 64), 0);
}

#[test]
fn divide_accepts_one_past_end() {
    let c = cfg(); let p = Params::new(&c); let mut s = State::init(&c, &p);
    let at = p.geom.index([4, 4, 0]); let child = p.geom.index([5, 4, 0]);
    let g = [0x41, 0x17, 0x45, 0x18, 0x68, 0x77, 0x78, 0x6f];
    put(&mut s, &p, at, &g); run(&mut s, &p, at);
    assert_eq!(s.tape(child), &g[2..]); assert_eq!(s.tape(at), &g[..2]);
}

#[test]
fn sleep_skips_the_full_number_of_future_epochs() {
    for energy in [false, true] { for scale in [1, 3] { for n in [1, 4] {
        let mut c = cfg(); c.energy.enabled = energy; c.individual.sleep_scale = scale;
        let p = Params::new(&c); let mut s = State::init(&c, &p); let at = p.geom.index([4, 4, 0]);
        put(&mut s, &p, at, &[0x40 + n, isa::op::SLEEP, isa::op::HALT]);
        assert_eq!(run(&mut s, &p, at), 2);
        for _ in 0..n as u32 * scale { phases::absorb_upkeep(&mut s, &p); assert_eq!(run(&mut s, &p, at), 0); }
        phases::absorb_upkeep(&mut s, &p); assert_eq!(run(&mut s, &p, at), 2);
    } } }
}

#[test]
fn step_charges_its_own_cost() {
    let mut c = cfg(); c.individual.step_budget = 1;
    c.costs.insert("MOVE".into(), 20); c.costs.insert("STEP_L".into(), 1); c.costs.insert("STEP_D".into(), 1);
    c.validate().unwrap(); let p = Params::new(&c); let mut s = State::init(&c, &p);
    let at = p.geom.index([4, 4, 0]); put(&mut s, &p, at, &[isa::op::STEP_L]);
    let before: u32 = (0..s.n).map(|i| s.h(i, H_E)).sum(); run(&mut s, &p, at);
    let after: u32 = (0..s.n).map(|i| s.h(i, H_E)).sum(); assert_eq!(before - after, 1);
}

#[test]
fn validation_and_empty_chemistry_are_safe() {
    let mut invalid = cfg(); invalid.world.size.clear(); assert!(Sim::new(invalid, Box::new(CpuExec)).is_err());
    let mut c = cfg(); c.energy.absorb_alpha = 2.0; assert!(c.validate().is_err());
    c = cfg(); c.chemistry.reactions = vec![c.chemistry.reactions[0].clone(); 17]; assert!(c.validate().is_err());
    c = cfg(); c.chemistry.species.clear(); c.chemistry.mass.clear(); c.chemistry.init.clear(); c.chemistry.reactions.clear();
    c.energy.vents.enabled = true; c.energy.vents.count = 1; c.validate().unwrap();
    let p = Params::new(&c); let mut s = State::init(&c, &p); assert!(p.x4_species.is_none()); phases::vent_inject(&mut s, &p);
}

#[test]
fn mirror_region_moves_walls_and_requests_with_cells() {
    let c = cfg(); let p = Params::new(&c); let mut s = State::init(&c, &p);
    let a = p.geom.index([3, 4, 0]); let b = p.geom.index([5, 4, 0]);
    put(&mut s, &p, a, &[0x41]); *s.hm(a, H_FLAGS) |= 1 << REQ_SHIFT;
    s.site[b * SITE + S_WALL] = WALL_REFLECT;
    events::mirror_region(&mut s, &p, [3, 4, 0], [5, 4, 0]).unwrap();
    assert!(s.occupied(b)); assert_eq!(s.wall(b), 0); assert_eq!(s.wall(a), WALL_REFLECT);
    assert_eq!((s.flags(b) & REQ_MASK) >> REQ_SHIFT, 2);
}

#[test]
fn snapshots_preserve_pending_ledger_and_reject_malformed_data() {
    let mut c = cfg(); c.telemetry.ledger_every = 10;
    let mut sim = Sim::new(c.clone(), Box::new(CpuExec)).unwrap();
    for _ in 0..5 { sim.step(); }
    sim.st.ledger.violations = 123;
    let path = tmp("roundtrip.psnap"); sim.st.write_snapshot(&path, 99).unwrap();
    let (st, hash) = State::read_snapshot(&path, &sim.p).unwrap();
    assert_eq!(hash, 99); assert_eq!(st.tally, sim.st.tally); assert_eq!(st.ledger, sim.st.ledger);
    let mut resumed = Sim::from_state(c, sim.p.clone(), st, Box::new(CpuExec)).unwrap();
    for _ in 0..5 { sim.step(); resumed.step(); }
    assert_eq!(sim.st.hash(), resumed.st.hash()); assert_eq!(sim.st.ledger, resumed.st.ledger);
    let bad = zstd::encode_all(&b"PSNP\x02"[..], 1).unwrap(); std::fs::write(&path, bad).unwrap();
    assert!(State::read_snapshot(&path, &sim.p).is_err()); std::fs::remove_file(path).unwrap();
}

#[test]
fn live_reshape_is_rejected_and_missing_events_are_errors() {
    let mut c = cfg(); c.world.size = vec![9, 6]; let mut sim = Sim::new(c, Box::new(CpuExec)).unwrap();
    sim.push_event(Event { epoch: 0, kind: "set".into(), params: serde_json::json!({"key":"world.size","value":[6,9]}) });
    sim.step(); assert_eq!(sim.rejected.len(), 1); assert_eq!(sim.p.geom.size, [9, 6, 1]);
    assert!(events::read_events(&tmp("missing.jsonl")).is_err());
}

#[test]
fn allocated_offspring_can_reproduce_after_isolation() {
    // A controlled lifecycle acceptance case, not an assertion that random
    // soup or the live preset evolves this behavior unaided.
    let mut c = cfg(); c.individual.step_budget = 64;
    c.energy.init_f = 20000; c.energy.absorb_alpha = 0.2;
    c.energy.wake_energy = 50; c.energy.dissipation = 0.0;
    c.bonds.birth_bond = 0.0; c.shielding.enabled = false;
    let genome = [isa::op::LEN, isa::op::ALLOC, isa::op::CPY, isa::op::H0INC, isa::op::H1INC, 0xc2];
    let p = Params::new(&c); let at = p.geom.index([4, 4, 0]); let child = p.geom.index([5, 4, 0]);
    let mut sim = Sim::new(c.clone(), Box::new(CpuExec)).unwrap(); put(&mut sim.st, &p, at, &genome);
    *sim.st.hm(at, H_LINEAGE) = 7; sim.st.reset_ledger(&p);
    for _ in 0..3 {
        sim.step();
        assert_eq!(sim.st.tape(at), genome); assert_eq!(sim.st.tape(child), genome);
        assert_eq!(sim.st.h(child, H_PARENT), sim.st.h(at, H_LINEAGE));
        assert_eq!(sim.st.ledger.violations, 0);
        let mut isolated = Sim::new(c.clone(), Box::new(CpuExec)).unwrap();
        isolated.st.epoch = sim.st.epoch;
        let from = child * sim.st.lmax; let to = at * isolated.st.lmax;
        isolated.st.tapes[to..to + genome.len()].copy_from_slice(&sim.st.tapes[from..from + genome.len()]);
        isolated.st.hdr[at * HDR..(at + 1) * HDR].copy_from_slice(&sim.st.hdr[child * HDR..(child + 1) * HDR]);
        isolated.st.reset_ledger(&p); sim = isolated;
    }
}

#[test]
fn invalid_image_edits_are_atomic_and_valid_maps_reload() {
    let mut sim = Sim::new(cfg(), Box::new(CpuExec)).unwrap();
    sim.push_event(Event { epoch: 0, kind: "set".into(), params: serde_json::json!({"key":"mutation.map","value":"image"}) });
    sim.step(); assert_eq!(sim.cfg.mutation.map, "uniform"); assert_eq!(sim.rejected.len(), 1);
    let path = tmp("map.png"); image::GrayImage::from_pixel(9, 9, image::Luma([255])).save(&path).unwrap();
    for (key, value) in [
        ("mutation.map_image", serde_json::json!(path)),
        ("mutation.map_a", serde_json::json!(1.0)),
        ("mutation.map", serde_json::json!("image")),
    ] { sim.push_event(Event { epoch: 1, kind: "set".into(), params: serde_json::json!({"key":key,"value":value}) }); }
    sim.step(); assert_eq!(sim.map_image.as_ref().unwrap()[0], 2 << 16);
    sim.push_event(Event { epoch: 2, kind: "set".into(), params: serde_json::json!({"key":"mutation.map_a","value":2.0}) });
    sim.step(); assert_eq!(sim.map_image.as_ref().unwrap()[0], 3 << 16);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn seed_events_allocate_their_full_tape_length() {
    let sources = [include_str!("../../../runs/seed-501.jsonl"), include_str!("../../../runs/seed-201.jsonl"),
        include_str!("../../../runs/seed-3d.jsonl"), include_str!("../../../runs/seed-small.jsonl")];
    for line in sources.iter().flat_map(|s| s.lines()).filter(|l| !l.trim().is_empty()) {
        let event: Event = serde_json::from_str(line).unwrap();
        let genome = isa::parse_hex_genome(event.params["genome"].as_str().unwrap()).unwrap();
        let mut c = cfg(); c.individual.step_budget = 9;
        let p = Params::new(&c); let mut st = State::init(&c, &p);
        let at = p.geom.index([4, 4, 0]); let child = p.geom.index([5, 4, 0]);
        put(&mut st, &p, at, &genome); run(&mut st, &p, at);
        assert!(st.occupied(child)); assert_eq!(st.len(child), genome.len());
    }
}

#[test]
fn mutation_only_copier_population_has_no_systematic_right_bias() {
    let mut c = cfg(); c.world.dims = 0; c.world.size = vec![100_000];
    c.individual.l_max = 4; c.individual.l_init = 4;
    c.energy.enabled = false; c.chemistry.enabled = false; c.matter.enabled = false; c.bonds.enabled = false;
    c.mutation.enabled = true; c.mutation.background = 0.1; c.mutation.activation = 0.0;
    c.mutation.radiation = 0.0; c.mutation.hand_flip = 0.0;
    for seed in 1..=3 {
        c.seed = seed; let p = Params::new(&c); let mut st = State::empty(&p);
        for i in 0..st.n {
            st.tapes[i * 4..i * 4 + 4].copy_from_slice(&isa::MINIMAL_REPLICATOR);
            *st.hm(i, H_LEN) = 4; *st.hm(i, H_FLAGS) = FL_OCC; *st.fm(i, F_T) = 30000;
        }
        phases::mutate_decay(&mut st, &p, None);
        let mut hands = [0i64; 2];
        for &b in &st.tapes { if isa::is_chiral(b) { hands[(b & 1) as usize] += 1; } }
        let ee = (hands[0] - hands[1]) as f64 / (hands[0] + hands[1]) as f64;
        eprintln!("mutation-only seed {seed}: L={} D={} EE={ee:.5}", hands[0], hands[1]);
        assert!(ee.abs() < 0.05, "seed {seed}: EE={ee}, hands={hands:?}");
    }
}

#[test]
fn legacy_snapshots_rebase_accounting_and_bad_array_lengths_fail() {
    let c = cfg(); let p = Params::new(&c); let mut st = State::init(&c, &p);
    st.ledger.energy_prev = -1; st.ledger.violations = 123;
    let path = tmp("legacy.psnap"); st.write_snapshot(&path, 42).unwrap();
    let mut raw = zstd::decode_all(std::fs::File::open(&path).unwrap()).unwrap();
    // Rebuild a version 1 file: its first six blocks, with 9-word headers.
    let (mut pos, mut blocks) = (25, vec![]);
    for _ in 0..6 { let n = u64::from_le_bytes(raw[pos..pos + 8].try_into().unwrap()) as usize; blocks.push(raw[pos + 8..pos + 8 + n].to_vec()); pos += 8 + n; }
    blocks[1] = blocks[1].chunks_exact(HDR * 4).flat_map(|row| row[..HDR_V2 * 4].to_vec()).collect();
    raw.truncate(25); raw[4..8].copy_from_slice(&1u32.to_le_bytes());
    for b in blocks { raw.extend_from_slice(&(b.len() as u64).to_le_bytes()); raw.extend_from_slice(&b); }
    std::fs::write(&path, zstd::encode_all(&raw[..], 1).unwrap()).unwrap();
    let (loaded, hash) = State::read_snapshot(&path, &p).unwrap();
    assert_eq!(hash, 42); assert_eq!(loaded.hash(), st.hash()); assert_eq!(loaded.ledger.energy_prev, st.energy_total());
    let header_block = 25 + 8 + st.tapes.len();
    raw[header_block..header_block + 8].copy_from_slice(&0u64.to_le_bytes());
    std::fs::write(&path, zstd::encode_all(&raw[..], 1).unwrap()).unwrap();
    assert!(State::read_snapshot(&path, &p).is_err()); std::fs::remove_file(path).unwrap();
}
