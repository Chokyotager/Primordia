//! T1-T3, T6-T9, T12 on the CPU golden model.

use primordia_core::config::Config;
use primordia_core::events::Event;
use primordia_core::exec::{exec_slice, flag_snapshot};
use primordia_core::isa::{self, op};
use primordia_core::params::*;
use primordia_core::state::State;
use primordia_core::{CpuExec, Params, Sim};

fn small(dims: u32, preset: &str) -> Config {
    let mut c = Config::preset(preset).unwrap();
    c.world.dims = dims;
    c.world.size = match dims {
        0 => vec![256],
        1 => vec![48],
        2 => vec![24, 18],
        _ => vec![9, 9, 9],
    };
    c.gpu.exec = false;
    c.validate().unwrap();
    c
}

/// A blank world with one individual at `c` holding `tape` (and optional partner).
fn blank(cfg: &Config) -> (Params, State) {
    let mut cfg = cfg.clone();
    cfg.world.init_occupancy = 0.0;
    let p = Params::new(&cfg);
    let st = State::init(&cfg, &p);
    (p, st)
}

fn put(st: &mut State, p: &Params, c: usize, tape: &[u8], orient: u32, e: u32) {
    let l = st.lmax;
    st.tapes[c * l..c * l + tape.len()].copy_from_slice(tape);
    st.hdr[c * HDR + H_LEN] = tape.len() as u32;
    st.hdr[c * HDR + H_FLAGS] = FL_OCC | (orient << ORIENT_SHIFT);
    st.hdr[c * HDR + H_E] = e;
    st.hdr[c * HDR + H_LINEAGE] = 7;
    let _ = p;
}

fn run1(st: &mut State, p: &Params, c: usize) {
    let snap = flag_snapshot(st);
    exec_slice(st, p, &snap, c, None, None);
}

fn deterministic_cfg(dims: u32) -> Config {
    let mut c = small(dims, "primordia-2d-sun");
    c.individual.random_initial_facing = false;
    c.individual.pair_order = primordia_core::config::PairOrder::SelfFirst;
    c.individual.step_budget = 64;
    c
}

// ------------------------------------------------------------------ T1

#[test]
fn t1_stack_ops() {
    let cfg = deterministic_cfg(2);
    let (p, mut st) = blank(&cfg);
    let c = p.geom.index([5, 5, 0]);
    // PUSH 3, PUSH 4, ADD, DUP, SHL4 -> stack [7, 112]; then WR at H1 (= len, i.e. partner empty -> wraps to self)
    let o = p.orient.valid[2][0];
    put(&mut st, &p, c, &[0x43, 0x44, op::ADD, op::DUP, op::SHL4, op::HALT], o, 1000);
    let mut tr = vec![];
    let snap = flag_snapshot(&st);
    exec_slice(&mut st, &p, &snap, c, None, Some(&mut tr));
    let last = tr.last().unwrap();
    assert_eq!(last.op, op::HALT);
    assert_eq!(last.stack, vec![7, 112]);
    // energy: 5 ops x 1, HALT 0
    assert_eq!(st.h(c, H_E), 1000 - 5);
}

#[test]
fn t1_underflow_and_overflow_end_slice() {
    let cfg = deterministic_cfg(2);
    let (p, mut st) = blank(&cfg);
    let c = p.geom.index([5, 5, 0]);
    put(&mut st, &p, c, &[op::DROP, 0x41, 0x41], 0, 1000);
    let mut tr = vec![];
    let snap = flag_snapshot(&st);
    exec_slice(&mut st, &p, &snap, c, None, Some(&mut tr));
    assert_eq!(tr.len(), 1, "DROP on empty stack ends the slice");
    // overflow: 17 pushes via a loop PUSH1; JMP-2 -> 16 pushes ok, 17th ends
    let (p, mut st) = blank(&cfg);
    put(&mut st, &p, c, &[0x41, 0xC0], 0, 1000);
    let mut tr = vec![];
    let snap = flag_snapshot(&st);
    exec_slice(&mut st, &p, &snap, c, None, Some(&mut tr));
    let pushes = tr.iter().filter(|s| s.op == 0x41).count();
    assert_eq!(pushes, 17);
    assert_eq!(tr.last().unwrap().stack.len(), 16);
}

#[test]
fn t1_jumps_and_skips() {
    let cfg = deterministic_cfg(2);
    let (p, mut st) = blank(&cfg);
    let c = p.geom.index([5, 5, 0]);
    // 0: PUSH0 1: SKIPZ 2: HALT(skipped) 3: JMP+2 (0x80) -> 5  4: HALT  5: PUSH 9 6: HALT
    put(&mut st, &p, c, &[0x40, op::SKIPZ, op::HALT, 0x80, op::HALT, 0x49, op::HALT], 0, 1000);
    let mut tr = vec![];
    let snap = flag_snapshot(&st);
    exec_slice(&mut st, &p, &snap, c, None, Some(&mut tr));
    let ips: Vec<u32> = tr.iter().map(|s| s.ip).collect();
    assert_eq!(ips, vec![0, 1, 3, 5, 6]);
    // JNZ: PUSH1 PUSH3 JNZ -> ip 2+3 = 5
    let (p, mut st) = blank(&cfg);
    put(&mut st, &p, c, &[0x41, 0x43, op::JNZ, op::HALT, op::HALT, op::HALT], 0, 1000);
    let mut tr = vec![];
    let snap = flag_snapshot(&st);
    exec_slice(&mut st, &p, &snap, c, None, Some(&mut tr));
    assert_eq!(tr.last().unwrap().ip, 5);
}

#[test]
fn t1_wrap_and_empty_partner() {
    let cfg = deterministic_cfg(2);
    let (p, mut st) = blank(&cfg);
    let c = p.geom.index([5, 5, 0]);
    // no partner: address space = own tape (len 3). IP wraps.
    put(&mut st, &p, c, &[0x41, op::DROP, op::NOP], 0, 1000);
    let mut tr = vec![];
    let snap = flag_snapshot(&st);
    exec_slice(&mut st, &p, &snap, c, None, Some(&mut tr));
    assert_eq!(tr.len(), 64);
    assert_eq!(tr[3].ip, 0);
    assert!(tr.iter().all(|s| s.len_partner == 0));
}

#[test]
fn t1_cpy_matter_exchange_and_failure() {
    let mut cfg = deterministic_cfg(2);
    cfg.mutation.copy.sub = 0.0;
    cfg.mutation.copy.ins = 0.0;
    cfg.mutation.copy.del = 0.0;
    cfg.shielding.enabled = false;
    cfg.chirality.stereo_copy = false;
    cfg.individual.step_budget = 16;
    let (p, mut st) = blank(&cfg);
    let c = p.geom.index([5, 5, 0]);
    let o = p.orient.valid[2][0]; // facing +x
    let t = p.geom.index([6, 5, 0]);
    put(&mut st, &p, c, &isa::MINIMAL_REPLICATOR, o, 10_000);
    put(&mut st, &p, t, &[0, 0, 0, 0], o, 0);
    st.hdr[t * HDR + H_FLAGS] |= FL_HALT;
    // empty all pools: copies must fail
    for cell in 0..st.n {
        for cl in 0..3 {
            *st.fm(cell, p.f_pool(cl)) = 0;
        }
    }
    run1(&mut st, &p, c);
    assert_eq!(st.tape(t), &[0, 0, 0, 0], "no monomers -> write fails");
    // give the target pool A monomers: now the copy succeeds
    st.hdr[c * HDR + H_FLAGS] &= !FL_EXEC;
    *st.fm(t, p.f_pool(isa::CL_A as usize)) = 4;
    let before = st.matter_totals(&p);
    run1(&mut st, &p, c);
    assert_eq!(st.tape(t), &isa::MINIMAL_REPLICATOR);
    assert_eq!(st.f(t, p.f_pool(2)), 0);
    assert_eq!(st.matter_totals(&p), before, "matter conserved");
}

#[test]
fn t1_move_bond_alloc_divide() {
    let cfg = deterministic_cfg(2);
    let (p, mut st) = blank(&cfg);
    let o = p.orient.valid[2][0]; // +x
    let c = p.geom.index([5, 5, 0]);
    put(&mut st, &p, c, &[op::MOVE, op::NOP], o, 1000);
    run1(&mut st, &p, c);
    let t = p.geom.index([6, 5, 0]);
    assert!(!st.occupied(c) && st.occupied(t));
    assert_eq!(st.h(t, H_E), 1000 - 20);
    // ALLOC 5 into faced empty cell, then BOND
    let (p, mut st) = blank(&cfg);
    put(&mut st, &p, c, &[0x45, op::ALLOC, op::BOND, op::HALT], o, 1000);
    run1(&mut st, &p, c);
    assert!(st.occupied(t));
    assert_eq!(st.len(t), 5);
    assert_eq!(st.h(t, H_PARENT), 7);
    assert!(st.bonds(c) & 1 != 0 && st.bonds(t) & 2 != 0);
    // DIVIDE: H0 = 2, H1 = 4 -> child gets bytes [2,4)
    let (p, mut st) = blank(&cfg);
    // PUSH 2 SETH0 (h0 = ip(1)+2=3)... use GETH-free form: PUSH1 SETH0 -> h0 = 1+1 = 2; PUSH2 SETH1 -> h1 = 3+2 = 5
    put(&mut st, &p, c, &[0x41, op::SETH0, 0x42, op::SETH1, op::DIVIDE, 0x77, 0x78, op::HALT], o, 1000);
    run1(&mut st, &p, c);
    assert_eq!(st.tape(t), &[0x42, op::SETH1, op::DIVIDE]);
    assert_eq!(st.tape(c), &[0x41, op::SETH0, 0x77, 0x78, op::HALT]);
}

#[test]
fn t1_halt_on_energy() {
    let cfg = deterministic_cfg(2);
    let (p, mut st) = blank(&cfg);
    let c = p.geom.index([5, 5, 0]);
    put(&mut st, &p, c, &[0x41, 0x41, 0x41, 0x41], 0, 3);
    let mut tr = vec![];
    let snap = flag_snapshot(&st);
    exec_slice(&mut st, &p, &snap, c, None, Some(&mut tr));
    assert_eq!(tr.len(), 3);
    assert!(st.flags(c) & FL_HALT != 0);
    assert_eq!(st.site[c * SITE + S_SPENT], 3);
}

/// `SLEEP n` sleeps min(n, 255) x individual.sleep_scale epochs.
#[test]
fn t1_sleep_scale() {
    for (scale, n, want) in [(1u32, 0x45u8, 5u32), (16, 0x45, 80), (16, 0x5F, 496)] {
        let mut cfg = deterministic_cfg(2);
        cfg.individual.sleep_scale = scale;
        let (p, mut st) = blank(&cfg);
        let c = p.geom.index([5, 5, 0]);
        put(&mut st, &p, c, &[n, op::SLEEP, 0x41], 0, 1000);
        let snap = flag_snapshot(&st);
        exec_slice(&mut st, &p, &snap, c, None, None);
        assert_eq!(st.h(c, H_SLEEP), want, "scale {scale}");
    }
}

/// With individual.sleep_shield a sleeping neighbour cannot be copied over;
/// once it wakes (or without the shield) the same copy succeeds.
#[test]
fn t1_sleep_shield() {
    for shield in [false, true] {
        let mut cfg = deterministic_cfg(2);
        cfg.mutation.copy.sub = 0.0;
        cfg.mutation.copy.ins = 0.0;
        cfg.mutation.copy.del = 0.0;
        cfg.shielding.enabled = false;
        cfg.chirality.stereo_copy = false;
        cfg.individual.step_budget = 16;
        cfg.individual.sleep_shield = shield;
        let (p, mut st) = blank(&cfg);
        let c = p.geom.index([5, 5, 0]);
        let o = p.orient.valid[2][0]; // facing +x
        let t = p.geom.index([6, 5, 0]);
        put(&mut st, &p, c, &isa::MINIMAL_REPLICATOR, o, 10_000);
        put(&mut st, &p, t, &[0, 0, 0, 0], o, 0);
        st.hdr[t * HDR + H_SLEEP] = 100;
        *st.fm(t, p.f_pool(isa::CL_A as usize)) = 4;
        run1(&mut st, &p, c);
        let want: &[u8] = if shield { &[0, 0, 0, 0] } else { &isa::MINIMAL_REPLICATOR };
        assert_eq!(st.tape(t), want, "shield {shield}: sleeping target");
        if shield {
            st.hdr[t * HDR + H_SLEEP] = 0;
            st.hdr[c * HDR + H_FLAGS] &= !FL_EXEC;
            run1(&mut st, &p, c);
            assert_eq!(st.tape(t), &isa::MINIMAL_REPLICATOR, "awake target is copied");
        }
    }
}

// ------------------------------------------------------------------ T2

#[test]
fn t2_mirror_map() {
    for b in 0..=255u8 {
        assert_eq!(isa::mirror(isa::mirror(b)), b);
        let (c, m) = (isa::class(b), isa::class(isa::mirror(b)));
        match c {
            0 => assert_eq!(m, 1),
            1 => assert_eq!(m, 0),
            x => assert_eq!(m, x),
        }
    }
}

/// Each _L op on S equals the _D op on M(S), reflected: run every chiral op
/// in a random neighbourhood and compare.
#[test]
fn t2_chiral_ops_equivariant() {
    for dims in [1u32, 2, 3] {
        let mut cfg = small(dims, "primordia-2d-sun");
        cfg.world.init_occupancy = 0.6;
        cfg.individual.step_budget = 40;
        let p = Params::new(&cfg);
        for b in (0x20u8..0x40).step_by(2) {
            for seed in 0..4u64 {
                let mut cfg2 = cfg.clone();
                cfg2.seed = seed * 31 + b as u64;
                let p = Params::new(&cfg2);
                let mut st = State::init(&cfg2, &p);
                let c = p.geom.index([4, if dims > 1 { 4 } else { 0 }, if dims > 2 { 4 } else { 0 }]);
                // program: push operands then the op, a few times
                let prog = [0x43, b, 0x45, b, 0x41, 0x42, b, op::RAND, b, 0x44, b];
                let oc = st.orient(c);
                put(&mut st, &p, c, &prog, oc, 5000);
                let mut ms = st.mirror(&p);
                let mc = p.geom.mirror_cell(c);
                run1(&mut st, &p, c);
                run1(&mut ms, &p, mc);
                let back = ms.mirror(&p);
                assert!(back.first_diff(&st).is_none(), "dims {dims} op {:02x} seed {seed}: {:?}", b, back.first_diff(&st));
            }
        }
        let _ = p;
    }
}

// ------------------------------------------------------------------ T3

#[test]
fn t3_minimal_replicator_paper() {
    let cfg = small(0, "paper-0d");
    let p = Params::new(&cfg);
    let mut st = State::init(&cfg, &p);
    let l = st.lmax;
    // slot 0: replicator + random rest; slot 1: random. Pair (0,1), self first.
    let mut cfg2 = cfg.clone();
    cfg2.individual.pair_order = primordia_core::config::PairOrder::SelfFirst;
    cfg2.individual.step_budget = 512;
    let p = Params::new(&cfg2);
    st.tapes[0..4].copy_from_slice(&isa::MINIMAL_REPLICATOR);
    let src: Vec<u8> = st.tapes[0..l].to_vec();
    let snap = flag_snapshot(&st);
    exec_slice(&mut st, &p, &snap, 0, Some(1), None);
    assert_eq!(&st.tapes[l..2 * l], &src[..], "partner is an exact copy");
}

// ------------------------------------------------------------------ T6, T7

fn full_cfg(dims: u32) -> Config {
    let mut c = small(dims, if dims == 3 { "primordia-3d" } else { "primordia-2d-sun" });
    c.world.init_occupancy = 0.7;
    c.energy.vents.enabled = true;
    c.energy.vents.count = 3;
    c.energy.vents.spawn_rate = 0.05;
    c.energy.vents.drift = 0.2;
    c.energy.vents.lifetime = 40;
    c.energy.vents.floor_only = dims == 3;
    c.energy.sun.day_period = 30;
    c.mutation.map = "vent_radial".into();
    c.mutation.background = 2e-3;
    c.mutation.copy.sub = 0.02;
    c.mutation.copy.ins = 0.01;
    c.mutation.copy.del = 0.01;
    c.mutation.hand_flip = 1e-3;
    c.decay.rate = 1e-3;
    c.bonds.break_rate = 1e-2;
    c.chirality.racemization.rate = 1e-2;
    c.individual.step_budget = 128;
    c.energy.wake_energy = 50;
    c.validate().unwrap();
    c
}

/// Programs that exercise bodies, moves, catalysis and copying.
fn seed_programs(sim: &mut Sim) {
    let progs: [&[u8]; 4] = [
        &[op::BOND, 0x26, op::MOVE, 0x60, 0x43, op::SHARE, 0x41, 0x2C, 0x40, 0x2D, op::CPY, op::H0INC, op::H1INC, 0xC2],
        &[0x45, op::ALLOC, op::CPY, op::H0INC, op::H1INC, 0xC2],
        &[0x2A, 0x27, op::BOND, 0x60, 0x2B, 0x20, 0x60, op::TUMBLE, 0x60],
        &[0x41, 0x30, 0x41, 0x32, 0x40, 0x34, 0x44, 0x2E, 0x43, op::DIGEST, 0x41, op::SETH0, 0x45, op::SETH1, op::DIVIDE],
    ];
    let p = &sim.p;
    for c in 0..sim.st.n {
        if sim.st.occupied(c) && c % 3 == 0 {
            let pr = progs[(c / 3) % 4];
            let l = sim.st.lmax;
            sim.st.tapes[c * l..c * l + pr.len()].copy_from_slice(pr);
        }
    }
    sim.st.reset_ledger(p);
}

pub fn check_bonds(s: &Sim, w: &str) {
    for x in 0..s.st.n {
        let b = s.st.bonds(x);
        for d in 0..6 {
            if b & (1 << d) != 0 {
                let t = s.p.nbr[x * 6 + d];
                assert!(t != NONE, "{w}: cell {x} bond dir {d} to nothing, flags {:x}", s.st.flags(x));
                assert!(s.st.bonds(t as usize) & (1 << (d ^ 1)) != 0, "{w}: asym bond {x} {d}");
                assert!(s.st.occupied(x) && s.st.occupied(t as usize), "{w}: bond on empty cell {x} {d}");
            }
        }
    }
}

#[test]
fn t6_ledgers_balance() {
    for dims in [1u32, 2, 3] {
        let cfg = full_cfg(dims);
        let mut sim = Sim::new(cfg, Box::new(CpuExec)).unwrap();
        seed_programs(&mut sim);
        for _ in 0..150 {
            sim.begin_epoch();
            check_bonds(&sim, &format!("dims {dims} begin {}", sim.st.epoch));
            let p = sim.p.clone();
            sim.backend.exec(&mut sim.st, &p);
            check_bonds(&sim, &format!("dims {dims} exec {}", sim.st.epoch));
            sim.end_epoch();
            check_bonds(&sim, &format!("dims {dims} end {}", sim.st.epoch));
            let l = &sim.st.ledger;
            assert_eq!(
                (l.energy_residual, l.matter_residual, l.atom_residual),
                (0, [0, 0, 0], 0),
                "dims {dims} epoch {} ledger residual",
                sim.st.epoch
            );
        }
    }
}

#[test]
fn t7_twin_exact_mirror() {
    for dims in [1u32, 2, 3] {
        let cfg = full_cfg(dims);
        let mut a = Sim::new(cfg.clone(), Box::new(CpuExec)).unwrap();
        seed_programs(&mut a);
        let mst = a.st.mirror(&a.p);
        let mut b = Sim::from_state(cfg.clone(), Params::new(&cfg), mst, Box::new(CpuExec)).unwrap();
        for e in 0..120 {
            a.step();
            b.step();
            let ma = a.st.mirror(&a.p);
            if let Some(d) = ma.first_diff(&b.st) {
                panic!("dims {dims}: twin diverged at epoch {e}: {:?}", d);
            }
            assert_eq!(ma.hash(), b.st.hash());
        }
        // something actually happened
        assert!(a.st.tally.body_moves + a.st.vents.len() as u64 > 0 || dims == 1);
    }
}

// ------------------------------------------------------------------ T8, T9

#[test]
fn t8_resume_from_snapshot() {
    let cfg = full_cfg(2);
    let mut a = Sim::new(cfg.clone(), Box::new(CpuExec)).unwrap();
    for _ in 0..40 {
        a.step();
    }
    let dir = std::env::temp_dir().join(format!("psnap-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("s.psnap");
    a.st.write_snapshot(&path, 0).unwrap();
    for _ in 0..40 {
        a.step();
    }
    let p = Params::new(&cfg);
    let (st, _) = State::read_snapshot(&path, &p).unwrap();
    let mut b = Sim::from_state(cfg, p, st, Box::new(CpuExec)).unwrap();
    for _ in 0..40 {
        b.step();
    }
    assert_eq!(a.st.hash(), b.st.hash());
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn t9_replay_with_interventions() {
    let cfg = full_cfg(2);
    let evs = vec![
        Event { epoch: 5, kind: "paint".into(), params: serde_json::json!({"field":"F","center":[5,5],"radius":3,"amount":500}) },
        Event { epoch: 7, kind: "seed".into(), params: serde_json::json!({"genome":"12 13 15 c2","center":[10,10],"radius":2,"mirror":true}) },
        Event { epoch: 9, kind: "clear".into(), params: serde_json::json!({"center":[3,3],"radius":2}) },
        Event { epoch: 11, kind: "wall".into(), params: serde_json::json!({"center":[15,9],"radius":1,"type":"absorb"}) },
        Event { epoch: 13, kind: "set".into(), params: serde_json::json!({"key":"energy.absorb_alpha","value":0.2}) },
        Event { epoch: 15, kind: "mirror_region".into(), params: serde_json::json!({"min":[2,2],"max":[8,8]}) },
        Event { epoch: 16, kind: "paint".into(), params: serde_json::json!({"field":"chem","species":1,"hand":1,"center":[5,5],"radius":2,"amount":30}) },
    ];
    let run = || {
        let mut s = Sim::new(cfg.clone(), Box::new(CpuExec)).unwrap();
        for e in &evs {
            s.push_event(e.clone());
        }
        let mut hs = vec![];
        for _ in 0..30 {
            s.step();
            hs.push(s.st.hash());
            let l = &s.st.ledger;
            assert_eq!((l.energy_residual, l.matter_residual, l.atom_residual), (0, [0, 0, 0], 0), "epoch {}", s.st.epoch);
        }
        assert!(s.rejected.is_empty(), "{:?}", s.rejected);
        (hs, s.cfg.energy.absorb_alpha)
    };
    let (a, alpha) = run();
    let (b, _) = run();
    assert_eq!(a, b);
    assert_eq!(alpha, 0.2);
}

// ------------------------------------------------------------------ T12

#[test]
fn t12_fuzz_configs() {
    let mut s = 12345u64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for i in 0..12 {
        let dims = (rnd() % 4) as u32;
        let mut c = full_cfg(dims.max(1));
        c.world.dims = dims;
        c.world.size = match dims {
            0 => vec![64],
            1 => vec![30],
            2 => vec![12, 15],
            _ => vec![6, 6, 6],
        };
        c.seed = rnd();
        c.energy.enabled = rnd() % 4 != 0;
        c.matter.enabled = rnd() % 4 != 0;
        c.chemistry.enabled = rnd() % 4 != 0;
        c.shielding.enabled = rnd() % 2 == 0;
        c.shielding.granularity = if rnd() % 2 == 0 { primordia_core::config::Granularity::Byte } else { primordia_core::config::Granularity::Slice };
        c.bonds.enabled = rnd() % 3 != 0;
        c.chirality.stereo = rnd() % 2 == 0;
        c.individual.l_max = [16u32, 64, 128][(rnd() % 3) as usize];
        c.individual.l_init = c.individual.l_max / 2;
        c.individual.interaction = if rnd() % 5 == 0 { primordia_core::config::Interaction::PairConcat } else { primordia_core::config::Interaction::Hybrid };
        c.world.boundary = vec![[primordia_core::geom::Boundary::Torus, primordia_core::geom::Boundary::Reflecting, primordia_core::geom::Boundary::Absorbing][(rnd() % 3) as usize]; 3];
        if c.validate().is_err() {
            continue;
        }
        let mut sim = Sim::new(c, Box::new(CpuExec)).unwrap();
        for _ in 0..40 {
            sim.step();
            let l = &sim.st.ledger;
            assert_eq!((l.energy_residual, l.matter_residual, l.atom_residual), (0, [0, 0, 0], 0), "fuzz case {i} epoch {}", sim.st.epoch);
        }
    }
}
