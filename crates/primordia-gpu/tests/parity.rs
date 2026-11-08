//! T4: GPU execution phase vs CPU golden model, identical state hash every epoch.

use primordia_core::isa::op;
use primordia_core::{Config, CpuExec, Params, Sim};
use primordia_gpu::GpuExec;

fn cfg(dims: u32) -> Config {
    let mut c = Config::preset(if dims == 3 { "primordia-3d" } else { "primordia-2d-sun" }).unwrap();
    c.world.dims = dims;
    c.world.size = match dims {
        0 => vec![4096],
        1 => vec![300],
        2 => vec![63, 63],
        _ => vec![21, 21, 21],
    };
    c.world.init_occupancy = 0.7;
    c.energy.vents.enabled = true;
    c.energy.vents.count = 4;
    c.energy.vents.floor_only = dims == 3;
    c.energy.sun.day_period = 40;
    c.mutation.background = 2e-3;
    c.mutation.copy.sub = 0.02;
    c.mutation.copy.ins = 0.01;
    c.mutation.copy.del = 0.01;
    c.decay.rate = 1e-3;
    c.individual.step_budget = 256;
    c.energy.wake_energy = 50;
    c.validate().unwrap();
    c
}

fn seed(sim: &mut Sim) {
    let progs: [&[u8]; 5] = [
        &[op::BOND, 0x26, op::MOVE, 0x60, 0x43, op::SHARE, 0x41, 0x2C, 0x40, 0x2D, op::CPY, op::H0INC, op::H1INC, 0xC2],
        &[0x45, op::ALLOC, op::CPY, op::H0INC, op::H1INC, 0xC2],
        &[0x2A, 0x27, op::BOND, 0x60, 0x2B, 0x20, 0x60, op::TUMBLE, 0x60, 0x22, 0x24, 0x25],
        &[0x41, 0x30, 0x41, 0x32, 0x40, 0x34, 0x44, 0x2E, 0x43, op::DIGEST, 0x41, op::SETH0, 0x45, op::SETH1, op::DIVIDE],
        &[op::RAND, op::WR, op::H1INC, op::RD, op::SLEEP, op::ENRG, op::TEMP, op::FREE, op::POOL, 0x28, 0x29, op::SENSE, op::GROW, op::SHRINK, op::LEN, op::PLEN, 0xC8],
    ];
    let l = sim.st.lmax;
    for c in 0..sim.st.n {
        if sim.st.occupied(c) && c % 2 == 0 {
            let pr = progs[(c / 2) % 5];
            sim.st.tapes[c * l..c * l + pr.len()].copy_from_slice(pr);
        }
    }
    let p = sim.p.clone();
    sim.st.reset_ledger(&p);
}

fn parity(dims: u32, epochs: u32, paper: bool) {
    parity_mode(dims, epochs, paper, false);
    parity_mode(dims, epochs, paper, true);
}

fn parity_mode(dims: u32, epochs: u32, paper: bool, resident: bool) {
    let c = if paper {
        let mut c = Config::preset("paper-0d").unwrap();
        c.world.size = vec![8192];
        c
    } else {
        cfg(dims)
    };
    check(c, epochs, !paper, resident);
}

/// Step CPU and GPU side by side and require identical state every epoch.
fn check(c: Config, epochs: u32, seeded: bool, resident: bool) -> Sim {
    let dims = c.world.dims;
    let p = Params::new(&c);
    let gpu = match GpuExec::with_mode(&p, resident) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping GPU parity: {e:#}");
            return Sim::new(c, Box::new(CpuExec)).unwrap();
        }
    };
    let mut a = Sim::new(c.clone(), Box::new(CpuExec)).unwrap();
    let mut b = Sim::new(c.clone(), Box::new(gpu)).unwrap();
    if seeded {
        seed(&mut a);
        seed(&mut b);
    }
    b.touch();
    for e in 0..epochs {
        a.step();
        b.step();
        b.sync();
        if let Some(d) = a.st.first_diff(&b.st) {
            let c = d.0;
            panic!(
                "dims {dims} resident {resident}: GPU != CPU at epoch {e}: {:?}\n cpu hdr {:?}\n gpu hdr {:?}\n cpu f {:?}\n gpu f {:?}",
                d,
                &a.st.hdr[c * primordia_core::params::HDR..(c + 1) * primordia_core::params::HDR],
                &b.st.hdr[c * primordia_core::params::HDR..(c + 1) * primordia_core::params::HDR],
                &a.st.fields[c * a.st.nf..(c + 1) * a.st.nf],
                &b.st.fields[c * b.st.nf..(c + 1) * b.st.nf]
            );
        }
        assert_eq!(a.st.tally, b.st.tally, "tally at epoch {e}");
        assert_eq!(a.st.ledger, b.st.ledger, "ledger at epoch {e}");
        assert_eq!(a.take_births(), b.take_births(), "births at epoch {e}");
        assert_eq!(b.faults, 0, "labelling fault at epoch {e}");
        assert_eq!(a.st.hash(), b.st.hash());
    }
    let occ = (0..a.st.n).filter(|&c| a.st.occupied(c)).count();
    eprintln!("dims {dims} resident {resident}: {epochs} epochs identical, {occ} occupied, hash {:016x}", a.st.hash());
    a
}

/// Old-age death and moonlight: identical on both backends, nobody outlives
/// max_age, and nights are lit at the moon level.
#[test]
fn t4_ageing_and_moonlight() {
    let mut c = cfg(2);
    c.individual.max_age = 50;
    c.energy.sun.moon = 0.15;
    c.individual.sleep_scale = 16; // the seeded programs include SLEEP
    c.individual.sleep_shield = true;
    c.individual.sleep_absorb = 0.0;
    c.individual.sleep_upkeep = 0.5;
    c.validate().unwrap();
    let p = Params::new(&c);
    let night = primordia_core::phases::row_lights(&p, 30); // day_period 40: epoch 30 is mid-night
    assert!(night.iter().all(|&r| r > 0), "moonlit night has light");
    let mut dark = c.clone();
    dark.energy.sun.moon = 0.0;
    assert!(primordia_core::phases::row_lights(&Params::new(&dark), 30).iter().all(|&r| r == 0));
    for resident in [false, true] {
        // 70 epochs: the founding cohort (born at 0) has just reached max_age.
        let a = check(c.clone(), 70, true, resident);
        let st = &a.st;
        let ages: Vec<u32> = (0..st.n).filter(|&i| st.occupied(i)).map(|i| (st.epoch as u32) - st.h(i, primordia_core::params::H_BIRTH)).collect();
        assert!(!ages.is_empty(), "population survived");
        // checked before the epoch advances, so a cell is last seen at age max_age
        assert!(ages.iter().all(|&g| g <= 50), "oldest cell {}", ages.iter().max().unwrap());
        eprintln!("ageing: {} alive, oldest {}", ages.len(), ages.iter().max().unwrap());
    }
}

/// Dormancy penalties (idle-only counting, grace period, no absorption and
/// fast rot once dormant) and light_cover: identical on both backends.
#[test]
fn t4_dormancy_and_light_cover() {
    for dims in [2, 3] {
        let mut c = cfg(dims);
        c.individual.starve_after = 30;
        c.individual.dormant_idle = true;
        c.individual.dormant_grace = 5;
        c.individual.dormant_absorb = 0.0;
        c.individual.dormant_rot = 20.0;
        c.individual.sleep_scale = 16;
        c.shielding.light_cover = 0.0;
        c.energy.share_beta = if dims == 2 { 0.25 } else { 0.16 }; // at the stability limit: the limiter engages
        c.bonds.mutual = true;
        c.bonds.birth_bond = 0.5;
        c.bonds.kin_only = dims == 3;
        c.energy.sun.moon = 0.05;
        c.decay.light_factor = 3.0;
        c.validate().unwrap();
        for resident in [false, true] {
            let a = check(c.clone(), 80, true, resident);
            let st = &a.st;
            let dormant = (0..st.n).filter(|&i| st.occupied(i) && st.h(i, primordia_core::params::H_ZERO) > 0).count();
            eprintln!("dormancy dims {dims} resident {resident}: {dormant} cells dormant at the end");
        }
    }
}

#[test]
fn t4_gpu_cpu_parity_2d() {
    parity(2, 150, false);
}
#[test]
fn t4_gpu_cpu_parity_3d() {
    parity(3, 100, false);
}
#[test]
fn t4_gpu_cpu_parity_1d() {
    parity(1, 150, false);
}
#[test]
fn t4_gpu_cpu_parity_paper0d() {
    parity(0, 60, true);
}

/// Resident GPU vs CPU over random configs with boundaries, walls, vents,
/// mutation maps and interventions (which force a download / re-upload).
#[test]
fn t4_resident_fuzz() {
    use primordia_core::events::Event;
    let mut s = 777u64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for case in 0..8 {
        let dims = 1 + (rnd() % 3) as u32;
        let mut c = cfg(dims);
        c.world.size = match dims {
            1 => vec![90],
            2 => vec![30, 27],
            _ => vec![12, 9, 9],
        };
        c.seed = rnd();
        let bnd = [primordia_core::geom::Boundary::Torus, primordia_core::geom::Boundary::Reflecting, primordia_core::geom::Boundary::Absorbing];
        c.world.boundary = (0..3).map(|_| bnd[(rnd() % 3) as usize]).collect();
        c.mutation.map = if rnd() % 2 == 0 { "vent_radial".into() } else { "uniform".into() };
        c.energy.vents.drift = 0.3;
        c.energy.vents.spawn_rate = 0.1;
        c.energy.vents.lifetime = 30;
        c.chemistry.global_sink = if rnd() % 2 == 0 { 1e-3 } else { 0.0 };
        c.matter.enabled = rnd() % 4 != 0;
        c.chemistry.enabled = rnd() % 4 != 0;
        c.energy.enabled = rnd() % 5 != 0;
        c.shielding.granularity = if rnd() % 2 == 0 { primordia_core::config::Granularity::Byte } else { primordia_core::config::Granularity::Slice };
        c.bonds.break_rate = 1e-2;
        c.individual.starve_after = if rnd() % 2 == 0 { 20 } else { 0 };
        c.individual.max_age = if rnd() % 2 == 0 { 30 } else { 0 };
        c.energy.sun.moon = if rnd() % 2 == 0 { 0.1 } else { 0.0 };
        c.individual.sleep_scale = if rnd() % 2 == 0 { 16 } else { 1 };
        c.individual.sleep_shield = rnd() % 2 == 0;
        c.individual.sleep_absorb = [1.0, 0.0, 0.3][(rnd() % 3) as usize];
        c.individual.sleep_upkeep = [0.125, 0.5, 1.0][(rnd() % 3) as usize];
        c.individual.dormant_idle = rnd() % 2 == 0;
        c.individual.dormant_grace = [0u32, 5, 20][(rnd() % 3) as usize];
        c.individual.dormant_absorb = [1.0, 0.0, 0.25][(rnd() % 3) as usize];
        c.individual.dormant_rot = [1.0, 20.0, 5.0][(rnd() % 3) as usize];
        c.shielding.light_cover = [1.0, 0.0, 0.5][(rnd() % 3) as usize];
        c.energy.share_beta = [0.1, 0.16][(rnd() % 2) as usize];
        c.bonds.mutual = rnd() % 2 == 0;
        c.bonds.kin_only = rnd() % 2 == 0;
        c.bonds.birth_bond = [0.0, 0.5, 1.0][(rnd() % 3) as usize];
        c.energy.upkeep.base = 4.0;
        if c.validate().is_err() {
            continue;
        }
        let p = Params::new(&c);
        let Ok(gpu) = GpuExec::with_mode(&p, true) else { return };
        let mut a = Sim::new(c.clone(), Box::new(CpuExec)).unwrap();
        let mut b = Sim::new(c.clone(), Box::new(gpu)).unwrap();
        seed(&mut a);
        seed(&mut b);
        b.touch();
        let evs = [
            Event { epoch: 10, kind: "wall".into(), params: serde_json::json!({"center":[5,5,4],"radius":1.5,"type":"absorb"}) },
            Event { epoch: 15, kind: "paint".into(), params: serde_json::json!({"field":"F","center":[8,8,3],"radius":3,"amount":400}) },
            Event { epoch: 20, kind: "seed".into(), params: serde_json::json!({"genome":"36 36 36 36 36 36 36 4f 67 40 2c 12 13 15 c4","center":[3,6,2],"radius":2,"mirror":true}) },
            Event { epoch: 25, kind: "set".into(), params: serde_json::json!({"key":"energy.absorb_alpha","value":0.2}) },
            Event { epoch: 30, kind: "wall".into(), params: serde_json::json!({"center":[12,3,6],"radius":1,"type":"reflect"}) },
        ];
        for e in &evs {
            a.push_event(e.clone());
            b.push_event(e.clone());
        }
        for e in 0..60 {
            a.step();
            b.step();
            if e % 3 == 0 || e >= 50 {
                b.sync();
                assert_eq!(a.take_births(), b.take_births(), "fuzz case {case} births at epoch {e}");
                assert_eq!(b.faults, 0);
                if let Some(d) = a.st.first_diff(&b.st) {
                    panic!("fuzz case {case} (dims {dims}, {:?}): resident GPU != CPU at epoch {e}: {:?}", c.world.boundary, d);
                }
            }
            
            
        }
        b.sync();
        assert_eq!(a.st.ledger, b.st.ledger, "fuzz case {case} ledger");
        assert_eq!(a.st.ledger.violations, 0);
        eprintln!("fuzz case {case}: dims {dims} {:?} identical", c.world.boundary);
    }
}

/// The pipelined path: many epochs in flight, asynchronous snapshots, no
/// syncing; snapshot hashes, births and the ledger must match the CPU model.
#[test]
fn t4_pipelined_snapshots() {
    let c = cfg(2);
    let p = Params::new(&c);
    let Ok(gpu) = GpuExec::with_mode(&p, true) else { return };
    let mut a = Sim::new(c.clone(), Box::new(CpuExec)).unwrap();
    let mut b = Sim::new(c.clone(), Box::new(gpu)).unwrap();
    seed(&mut a);
    seed(&mut b);
    b.touch();
    let mut want = std::collections::BTreeMap::new();
    let mut births_a = vec![];
    for _ in 0..80 {
        a.step();
        births_a.extend(a.take_births());
        b.step();
        if a.st.epoch % 7 == 0 {
            want.insert(a.st.epoch, (a.st.hash(), a.st.ledger.clone(), a.st.tally.clone()));
            b.request_snapshot();
        }
    }
    let snaps = b.take_snapshots(true);
    assert_eq!(snaps.len(), want.len());
    for s in &snaps {
        let expected = &want[&s.epoch];
        assert_eq!(s.hash(), expected.0, "snapshot at epoch {}", s.epoch);
        assert_eq!(s.ledger, expected.1, "snapshot ledger at epoch {}", s.epoch);
        assert_eq!(s.tally, expected.2, "snapshot tally at epoch {}", s.epoch);
    }
    b.sync();
    assert_eq!(a.st.hash(), b.st.hash());
    assert_eq!(births_a, b.take_births());
    assert_eq!(a.st.ledger, b.st.ledger);
    assert_eq!(b.faults, 0);
    eprintln!("pipelined: {} snapshots identical", snaps.len());
}

#[test]
fn live_boundary_and_reaction_edits_refresh_gpu_state() {
    use primordia_core::{events::Event, params::F_F};
    use primordia_core::geom::Boundary;
    let mut c = cfg(2); c.world.size = vec![9, 9]; c.world.init_occupancy = 0.0;
    c.energy.sun.enabled = false; c.energy.vents.enabled = false; c.energy.kappa_f = 0.1;
    c.energy.dissipation = 0.1; c.mutation.enabled = false; c.decay.enabled = false;
    let p = Params::new(&c); let Ok(gpu) = GpuExec::with_mode(&p, true) else { return };
    let mut a = Sim::new(c.clone(), Box::new(CpuExec)).unwrap(); let mut b = Sim::new(c.clone(), Box::new(gpu)).unwrap();
    for _ in 0..5 { a.step(); b.step(); }
    b.request_snapshot(); let old = a.st.clone(); b.sync();
    let mut reactions = c.chemistry.reactions.clone(); reactions.push(reactions[0].clone());
    for sim in [&mut a, &mut b] {
        for at in 0..sim.st.n { *sim.st.fm(at, F_F) = 0; }
        let at = sim.p.geom.index([0, 4, 0]); *sim.st.fm(at, F_F) = 10000; sim.st.reset_ledger(&sim.p); sim.touch();
        for (key, value) in [
            ("world.boundary", serde_json::json!(["reflecting", "torus"])),
            ("chemistry.reactions", serde_json::to_value(&reactions).unwrap()),
        ] { sim.push_event(Event { epoch: 5, kind: "set".into(), params: serde_json::json!({"key":key,"value":value}) }); }
    }
    a.step(); b.step(); b.request_snapshot(); let new = a.st.clone();
    for _ in 0..5 { a.step(); b.step(); }
    let snapshots = b.take_snapshots(true); b.sync();
    assert!(a.rejected.is_empty() && b.rejected.is_empty()); assert_eq!(a.st.hash(), b.st.hash());
    assert_eq!(snapshots.len(), 2);
    for (s, expected) in snapshots.iter().zip([old, new]) { assert_eq!(s.hash(), expected.hash()); assert_eq!(s.ledger, expected.ledger); }
    assert_eq!(snapshots[0].params.geom.boundary[0], Boundary::Torus);
    assert_eq!(snapshots[1].params.geom.boundary[0], Boundary::Reflecting);
    assert_eq!(snapshots[1].f(p.geom.index([8, 4, 0]), F_F), 0);
}

#[test]
fn maximum_field_counts_and_more_than_4096_vents_match_cpu() {
    let mut c = cfg(2); c.world.size = vec![9, 9]; c.world.init_occupancy = 0.0;
    c.chemistry.species = (0..8).map(|i| format!("X{i}")).collect();
    c.chemistry.mass = vec![1; 8]; c.chemistry.init = vec![100; 8]; c.chemistry.reactions.clear(); c.signals.count = 8;
    c.energy.vents.count = 4097; c.energy.vents.rate = 1; c.energy.vents.heat = 1; c.energy.vents.x4_rate = 1;
    c.energy.vents.drift = 0.0; c.energy.vents.stagger = false; c.validate().unwrap();
    check(c, 3, false, true);
}

#[test]
fn high_mean_mutation_matches_cpu() {
    use primordia_core::events::Event;
    let mut c = cfg(2); c.world.size = vec![9, 9]; c.world.init_occupancy = 1.0;
    c.individual.l_init = 256; c.individual.l_max = 256; c.individual.step_budget = 0;
    c.energy.enabled = false; c.energy.sun.enabled = false; c.energy.vents.enabled = false;
    c.chemistry.enabled = false; c.matter.enabled = false; c.bonds.enabled = false; c.decay.enabled = false;
    c.mutation.activation = 0.0; c.mutation.radiation = 0.0; c.mutation.hand_flip = 0.0; c.mutation.background = 0.04;
    let p = Params::new(&c); let Ok(gpu) = GpuExec::with_mode(&p, true) else { return };
    let mut a = Sim::new(c.clone(), Box::new(CpuExec)).unwrap(); let mut b = Sim::new(c, Box::new(gpu)).unwrap();
    for sim in [&mut a, &mut b] {
        for (epoch, rate) in [(2, 0.1), (4, 0.3), (6, 1.0)] {
            sim.push_event(Event { epoch, kind: "set".into(), params: serde_json::json!({"key":"mutation.background","value":rate}) });
        }
    }
    for _ in 0..8 { a.step(); b.step(); b.sync(); assert_eq!(a.st.hash(), b.st.hash()); assert_eq!(a.st.ledger, b.st.ledger); }
}

#[test]
fn birth_log_can_hold_every_newborn_in_an_epoch() {
    use primordia_core::params::*;
    // Repeating executor/empty/empty triples ensure each ALLOC has a free
    // neighbor and creates more than the former 16,384-record cap at once.
    let mut c = cfg(1); c.world.size = vec![3 * 16_500]; c.world.init_occupancy = 0.0;
    c.individual.l_max = 4; c.individual.l_init = 0; c.individual.step_budget = 2;
    c.individual.pair_order = primordia_core::config::PairOrder::SelfFirst; c.individual.random_initial_facing = false;
    c.energy.enabled = false; c.energy.sun.enabled = false; c.energy.vents.enabled = false;
    c.chemistry.enabled = false; c.matter.enabled = false; c.bonds.enabled = false;
    c.mutation.enabled = false; c.decay.enabled = false;
    let p = Params::new(&c); let Ok(gpu) = GpuExec::with_mode(&p, true) else { return };
    let mut a = Sim::new(c.clone(), Box::new(CpuExec)).unwrap(); let mut b = Sim::new(c, Box::new(gpu)).unwrap();
    for sim in [&mut a, &mut b] {
        for i in (0..sim.st.n).step_by(3) {
            let base = i * sim.st.lmax; sim.st.tapes[base..base + 2].copy_from_slice(&[0x44, op::ALLOC]);
            *sim.st.hm(i, H_LEN) = 2; *sim.st.hm(i, H_FLAGS) = FL_OCC | (p.orient.valid[1][0] << ORIENT_SHIFT);
            *sim.st.hm(i, H_LINEAGE) = i as u32 + 1;
        }
        sim.st.reset_ledger(&p); sim.touch();
    }
    a.step(); b.step(); b.sync(); let births = b.take_births();
    assert_eq!(births.len(), 16_500); assert_eq!(births, a.take_births());
    assert_eq!(a.st.hash(), b.st.hash()); assert!(b.rejected.is_empty());
}

/// individual.resume (saved IP/heads/stack across slices, including after
/// MOVE and SLEEP), bonds.protect and energy.copy_rest: identical on both
/// backends, with random and fixed pair order / facing.
#[test]
fn lifecycle_mechanisms_match_cpu() {
    use primordia_core::config::PairOrder;
    for (dims, fixed) in [(2, true), (2, false), (3, true)] {
        let mut c = cfg(dims);
        c.individual.resume = true;
        c.individual.step_budget = 48; // slices end mid-program
        c.individual.sleep_scale = 4;
        if fixed {
            c.individual.pair_order = PairOrder::SelfFirst;
            c.individual.random_initial_facing = false;
        }
        c.bonds.protect = true;
        c.bonds.construct_window = 12;
        c.bonds.mutual = true;
        c.bonds.birth_bond = 0.5;
        c.energy.copy_rest = 10;
        c.energy.copy_absorb = 0.3;
        c.validate().unwrap();
        for resident in [false, true] {
            let a = check(c.clone(), 80, true, resident);
            let st = &a.st;
            let ctx = (0..st.n).filter(|&i| st.flags(i) & primordia_core::params::FL_CTX != 0).count();
            let rest = (0..st.n).filter(|&i| st.h(i, primordia_core::params::H_REST) > 0).count();
            eprintln!("lifecycle dims {dims} fixed {fixed} resident {resident}: {ctx} saved contexts, {rest} resting");
            assert!(ctx > 0 && rest > 0);
        }
    }
}

/// bonds.gamma 0 (impermeable bonded faces) and a gamma large enough to hit
/// the stability cap: the GPU used to read gamma 0 as "no gamma" (= 1).
#[test]
fn bonded_gamma_extremes_match_cpu() {
    for gamma in [0.0, 50.0] {
        let mut c = cfg(2);
        c.bonds.gamma = gamma;
        c.bonds.birth_bond = 0.5;
        c.validate().unwrap();
        check(c, 40, true, true);
    }
}

/// shielding.private (foreign writes only into cells under construction) with
/// resumed execution: identical on both backends.
#[test]
fn private_tapes_match_cpu() {
    let mut c = cfg(2);
    c.shielding.private = true;
    c.bonds.construct_window = 12;
    c.individual.resume = true;
    c.individual.step_budget = 48;
    c.validate().unwrap();
    for resident in [false, true] {
        check(c.clone(), 60, true, resident);
    }
}
