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
                &a.st.hdr[c * 9..c * 9 + 9],
                &b.st.hdr[c * 9..c * 9 + 9],
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
            Event { epoch: 20, kind: "seed".into(), params: serde_json::json!({"genome":"36 36 36 36 36 36 36 4d 67 40 2c 12 13 15 c4","center":[3,6,2],"radius":2,"mirror":true}) },
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
            want.insert(a.st.epoch, a.st.hash());
            b.request_snapshot();
        }
    }
    let snaps = b.take_snapshots(true);
    assert_eq!(snaps.len(), want.len());
    for s in &snaps {
        assert_eq!(s.hash(), want[&s.epoch], "snapshot at epoch {}", s.epoch);
    }
    b.sync();
    assert_eq!(a.st.hash(), b.st.hash());
    assert_eq!(births_a, b.take_births());
    assert_eq!(a.st.ledger, b.st.ledger);
    assert_eq!(b.faults, 0);
    eprintln!("pipelined: {} snapshots identical", snaps.len());
}
