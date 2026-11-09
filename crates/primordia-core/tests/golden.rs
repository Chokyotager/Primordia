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

/// individual.sleep_absorb: a sleeping cell keeps that fraction of its light
/// absorption (0 = none); an awake cell and the default are unaffected.
#[test]
fn t1_sleep_absorb() {
    let take = |absorb: f64, asleep: bool| {
        let mut cfg = deterministic_cfg(2);
        cfg.individual.sleep_absorb = absorb;
        let (p, mut st) = blank(&cfg);
        let c = p.geom.index([5, 5, 0]);
        put(&mut st, &p, c, &[0x41, 0x41], 0, 1000);
        st.hdr[c * HDR + H_SLEEP] = if asleep { 10 } else { 0 };
        *st.fm(c, F_F) = 20_000;
        primordia_core::phases::absorb_upkeep(&mut st, &p);
        20_000 - st.f(c, F_F)
    };
    let full = take(1.0, true);
    assert!(full > 0);
    assert_eq!(take(0.0, true), 0, "a sleeping spore absorbs nothing");
    assert_eq!(take(0.0, false), full, "awake cells are unaffected");
    let half = take(0.5, true);
    assert!(half > 0 && half < full, "half {half} of {full}");
}

/// bonds.mutual: BOND only leaves a request until the neighbour bonds back;
/// a request lapses after an epoch in which its owner did not execute.
/// Without the rule BOND bonds at once.
#[test]
fn t1_bond_mutual() {
    for mutual in [false, true] {
        let mut cfg = deterministic_cfg(2);
        cfg.bonds.mutual = mutual;
        let (p, mut st) = blank(&cfg);
        let east = p.orient.valid[2].iter().copied().find(|&o| p.orient.f[o as usize] == 0).unwrap();
        let west = p.orient.valid[2].iter().copied().find(|&o| p.orient.f[o as usize] == 1).unwrap();
        let (a, b) = (p.geom.index([5, 5, 0]), p.geom.index([6, 5, 0]));
        put(&mut st, &p, a, &[op::BOND, op::HALT], east, 1000);
        put(&mut st, &p, b, &[op::BOND, op::HALT], west, 1000);
        run1(&mut st, &p, a);
        if !mutual {
            assert_eq!(st.bonds(a), 1, "bonds at once without the rule");
            continue;
        }
        assert_eq!(st.bonds(a), 0, "a one-sided BOND does not bond");
        assert_eq!(st.flags(a) & REQ_MASK, 1 << REQ_SHIFT, "it leaves a request toward +x");
        run1(&mut st, &p, b);
        assert_eq!((st.bonds(a), st.bonds(b)), (1, 2), "the answer bonds both");
        assert_eq!((st.flags(a) | st.flags(b)) & REQ_MASK, 0, "requests are used up");
        // a lone request lapses after an epoch without a slice
        st.hdr[a * HDR + H_FLAGS] &= !(BOND_MASK | FL_EXEC);
        st.hdr[b * HDR + H_FLAGS] &= !(BOND_MASK | FL_EXEC);
        run1(&mut st, &p, a);
        primordia_core::phases::absorb_upkeep(&mut st, &p);
        assert_ne!(st.flags(a) & REQ_MASK, 0, "kept through the epoch it was made");
        primordia_core::phases::absorb_upkeep(&mut st, &p);
        assert_eq!(st.flags(a) & REQ_MASK, 0, "lapsed after an epoch without a slice");
    }
}

/// bonds.kin_only: BOND joins only cells whose programs are kin; a stranger
/// is refused, a relative accepted, and birth bonds are exempt.
#[test]
fn t1_bond_kin_only() {
    for (kin_only, partner_kin, want) in [(false, false, 1u32), (true, false, 0), (true, true, 1)] {
        let mut cfg = deterministic_cfg(2);
        cfg.bonds.kin_only = kin_only;
        let (p, mut st) = blank(&cfg);
        let east = p.orient.valid[2].iter().copied().find(|&o| p.orient.f[o as usize] == 0).unwrap();
        let (a, b) = (p.geom.index([5, 5, 0]), p.geom.index([6, 5, 0]));
        let me = [op::BOND, op::HALT, 0x41, 0x42, 0x43, 0x44];
        let stranger = [0x45u8, 0x46, 0x47, 0x48, 0x49, 0x4A];
        put(&mut st, &p, a, &me, east, 1000);
        put(&mut st, &p, b, if partner_kin { &me } else { &stranger }, east, 1000);
        run1(&mut st, &p, a);
        assert_eq!(st.bonds(a), want, "kin_only {kin_only} partner kin {partner_kin}");
    }
    let s = |t: &[u8]| { let mut v = t.to_vec(); v.resize(16, 0); v };
    assert!(primordia_core::exec::tapes_kin(&s(&[1, 2, 3, 4]), &s(&[1, 2, 3, 4])));
    assert!(!primordia_core::exec::tapes_kin(&s(&[]), &s(&[])), "empty tapes are not kin");
    // birth bonds ignore the rule (the newborn's tape is still empty)
    let mut cfg = deterministic_cfg(2);
    cfg.bonds.kin_only = true;
    cfg.bonds.birth_bond = 1.0;
    let (p, mut st) = blank(&cfg);
    let east = p.orient.valid[2].iter().copied().find(|&o| p.orient.f[o as usize] == 0).unwrap();
    let a = p.geom.index([5, 5, 0]);
    put(&mut st, &p, a, &[0x44, op::ALLOC, op::HALT], east, 1000);
    run1(&mut st, &p, a);
    assert_eq!(st.bonds(a), 1, "birth bond despite kin_only");
}

/// bonds.birth_bond: a child made by ALLOC stays bonded to its parent with
/// that probability (1 = always, 0 = never), even under bonds.mutual.
#[test]
fn t1_birth_bond() {
    for (pb, want) in [(0.0, 0u32), (1.0, 1)] {
        let mut cfg = deterministic_cfg(2);
        cfg.bonds.birth_bond = pb;
        cfg.bonds.mutual = true;
        let (p, mut st) = blank(&cfg);
        let east = p.orient.valid[2].iter().copied().find(|&o| p.orient.f[o as usize] == 0).unwrap();
        let (a, b) = (p.geom.index([5, 5, 0]), p.geom.index([6, 5, 0]));
        put(&mut st, &p, a, &[0x44, op::ALLOC, op::HALT], east, 1000);
        run1(&mut st, &p, a);
        assert!(st.occupied(b), "ALLOC made a child");
        assert_eq!(st.bonds(a), want, "birth_bond {pb}");
        assert_eq!(st.bonds(b), want << 1);
    }
}

/// The mirror twin stays exact with mutual adhesion and birth bonds on
/// (requests are directional, like bonds).
#[test]
fn t7_twin_mutual_and_birth_bonds() {
    for dims in [2u32, 3] {
        let mut cfg = full_cfg(dims);
        cfg.bonds.mutual = true;
        cfg.bonds.birth_bond = 0.5;
        cfg.bonds.kin_only = dims == 3;
        cfg.validate().unwrap();
        let mut a = Sim::new(cfg.clone(), Box::new(CpuExec)).unwrap();
        seed_programs(&mut a);
        let mst = a.st.mirror(&a.p);
        let mut b = Sim::from_state(cfg.clone(), Params::new(&cfg), mst, Box::new(CpuExec)).unwrap();
        for e in 0..80 {
            a.step();
            b.step();
            if let Some(d) = a.st.mirror(&a.p).first_diff(&b.st) {
                panic!("dims {dims}: twin diverged at epoch {e}: {:?}", d);
            }
        }
        let reqs = (0..a.st.n).filter(|&c| a.st.flags(c) & REQ_MASK != 0).count();
        let bonded = (0..a.st.n).filter(|&c| a.st.bonds(c) != 0).count();
        eprintln!("dims {dims}: {reqs} cells with pending requests, {bonded} bonded");
    }
}

/// The mirror twin stays exact with resumed execution, protected bonds and
/// the reproduction rest period on (saved contexts are frame-independent).
#[test]
fn t7_twin_lifecycle_mechanisms() {
    for dims in [2u32, 3] {
        let mut cfg = full_cfg(dims);
        cfg.individual.resume = true;
        cfg.individual.step_budget = 48;
        cfg.shielding.private = dims == 3;
        cfg.shielding.predation = dims == 3;
        cfg.shielding.parent_only = dims == 3;
        cfg.shielding.gestation = dims == 3;
        cfg.shielding.armor = dims == 3;
        cfg.tasks.enabled = dims == 3;
        cfg.tasks.reward = 8;
        cfg.bonds.colony_generations = if dims == 3 { 2 } else { 0 };
        cfg.energy.sun.season_period = 50;
        cfg.energy.sun.season_band = 0.5;
        cfg.individual.max_age = 40;
        cfg.individual.max_age_spread = 0.5;
        cfg.bonds.protect = true;
        cfg.bonds.construct_window = 12;
        cfg.bonds.birth_bond = 0.5;
        cfg.energy.copy_rest = 10;
        cfg.energy.copy_absorb = 0.3;
        cfg.validate().unwrap();
        let mut a = Sim::new(cfg.clone(), Box::new(CpuExec)).unwrap();
        seed_programs(&mut a);
        let mst = a.st.mirror(&a.p);
        let mut b = Sim::from_state(cfg.clone(), Params::new(&cfg), mst, Box::new(CpuExec)).unwrap();
        for e in 0..80 {
            a.step();
            b.step();
            if let Some(d) = a.st.mirror(&a.p).first_diff(&b.st) {
                panic!("dims {dims}: twin diverged at epoch {e}: {:?}", d);
            }
        }
        let ctx = (0..a.st.n).filter(|&c| a.st.flags(c) & FL_CTX != 0).count();
        assert!(ctx > 0, "dims {dims}: some contexts were saved");
    }
}

/// energy.vents.stagger: the initial vents start at distinct random ages
/// below their lifetime (identically in the mirror twin); off, all start at 0.
#[test]
fn t1_vent_stagger() {
    let mut cfg = small(2, "primordia-2d-sun");
    cfg.energy.vents.enabled = true;
    cfg.energy.vents.count = 8;
    cfg.energy.vents.lifetime = 1000;
    let ages = |stagger: bool| {
        let mut c = cfg.clone();
        c.energy.vents.stagger = stagger;
        let p = Params::new(&c);
        let st = State::init(&c, &p);
        let m = st.mirror(&p);
        assert_eq!(st.vents.iter().map(|v| v.age).collect::<Vec<_>>(), m.vents.iter().map(|v| v.age).collect::<Vec<_>>());
        st.vents.iter().map(|v| v.age).collect::<Vec<u64>>()
    };
    assert!(ages(false).iter().all(|&a| a == 0));
    let a = ages(true);
    assert!(a.iter().all(|&x| x < 1000), "{a:?}");
    let mut d = a.clone();
    d.sort();
    d.dedup();
    assert!(d.len() >= 6, "ages should be spread: {a:?}");
}

/// individual.sleep_upkeep: a sleeper pays this fraction of its upkeep; the
/// default 0.125 is exactly the spec's integer 1/8.
#[test]
fn t1_sleep_upkeep() {
    let paid = |frac: f64| {
        let mut cfg = deterministic_cfg(2);
        cfg.individual.sleep_upkeep = frac;
        cfg.energy.upkeep.base = 7.3;
        let (p, mut st) = blank(&cfg);
        let c = p.geom.index([5, 5, 0]);
        put(&mut st, &p, c, &[0x41; 40], 0, 100_000);
        st.hdr[c * HDR + H_SLEEP] = 10;
        primordia_core::phases::absorb_upkeep(&mut st, &p);
        let up = (p.upkeep_base_q16 as u32) + (p.upkeep_byte_q16 as u32) * 40;
        (100_000u32.wrapping_sub(st.h(c, H_E)), up)
    };
    let mut cfg = deterministic_cfg(2);
    cfg.individual.sleep_upkeep = 0.125;
    let p = Params::new(&cfg);
    assert_eq!(primordia_core::fixed::mulshr16(123_457, p.sleep_upkeep_q16), 123_457 / 8, "default is exactly /8");
    // absorption adds energy too, so compare deltas between settings
    let (eighth, _) = paid(0.125);
    let (half, up) = paid(0.5);
    let (full, _) = paid(1.0);
    assert!(half > eighth && full > half, "eighth {eighth}, half {half}, full {full}");
    let d = (half - eighth) as i64 - ((up / 2 - up / 8) >> 16) as i64;
    assert!(d.abs() <= 1, "half - eighth differs from the upkeep difference by {d}");
}

/// individual.dormant_idle: with starvation on, a halted cell's dormancy
/// counter grows; with dormant_idle it grows only if the cell did not execute
/// this epoch (a cell that ran and exhausted itself is not dormant).
#[test]
fn t1_dormant_idle() {
    for idle in [false, true] {
        for executed in [false, true] {
            let mut cfg = deterministic_cfg(2);
            cfg.individual.starve_after = 100;
            cfg.individual.dormant_idle = idle;
            cfg.mutation.enabled = false;
            cfg.decay.enabled = false;
            let (p, mut st) = blank(&cfg);
            let c = p.geom.index([5, 5, 0]);
            put(&mut st, &p, c, &[0x41, 0x41], 0, 0);
            st.hdr[c * HDR + H_FLAGS] |= FL_HALT | if executed { FL_EXEC } else { 0 };
            st.hdr[c * HDR + H_ZERO] = 5;
            primordia_core::phases::mutate_decay(&mut st, &p, None);
            let want = if idle && executed { 0 } else { 6 };
            assert_eq!(st.h(c, H_ZERO), want, "idle {idle} executed {executed}");
        }
    }
}

/// individual.dormant_absorb: once dormant for dormant_grace epochs a halted
/// cell keeps only that fraction of its light absorption; before the grace
/// period ends, and for awake cells, absorption is unchanged.
#[test]
fn t1_dormant_absorb() {
    let take = |absorb: f64, halted: bool, zero: u32| {
        let mut cfg = deterministic_cfg(2);
        cfg.individual.dormant_grace = 20;
        cfg.individual.dormant_absorb = absorb;
        let (p, mut st) = blank(&cfg);
        let c = p.geom.index([5, 5, 0]);
        put(&mut st, &p, c, &[0x41, 0x41], 0, 0);
        if halted {
            st.hdr[c * HDR + H_FLAGS] |= FL_HALT;
        }
        st.hdr[c * HDR + H_ZERO] = zero;
        *st.fm(c, F_F) = 20_000;
        primordia_core::phases::absorb_upkeep(&mut st, &p);
        20_000 - st.f(c, F_F)
    };
    let full = take(1.0, true, 20);
    assert!(full > 0);
    assert_eq!(take(0.0, true, 20), 0, "dormant past grace absorbs nothing");
    assert_eq!(take(0.0, true, 19), full, "within the grace period");
    assert_eq!(take(0.0, false, 50), full, "awake cells are unaffected");
    let q = take(0.25, true, 50);
    assert!(q > 0 && q < full / 2, "quarter {q} of {full}");
}

/// individual.dormant_rot: decay is multiplied for a cell dormant past the
/// grace period, and only for that cell.
#[test]
fn t1_dormant_rot() {
    let left = |rot: f64, halted: bool| {
        let mut cfg = deterministic_cfg(2);
        cfg.individual.dormant_grace = 10;
        cfg.individual.dormant_rot = rot;
        cfg.mutation.enabled = false;
        cfg.decay.rate = 1e-3;
        cfg.decay.activation = 0.0;
        let (p, mut st) = blank(&cfg);
        let c = p.geom.index([5, 5, 0]);
        put(&mut st, &p, c, &[0x41; 64], 0, 0);
        if halted {
            st.hdr[c * HDR + H_FLAGS] |= FL_HALT;
        }
        st.hdr[c * HDR + H_ZERO] = 10;
        for _ in 0..5 {
            primordia_core::phases::mutate_decay(&mut st, &p, None);
            st.epoch += 1;
        }
        st.tape(c).iter().filter(|&&b| b != 0).count()
    };
    // about 6 expected decays per epoch at x100 (the Poisson sampler is
    // accurate only below about 20 events per cell and epoch)
    let (base, rot, awake) = (left(1.0, true), left(100.0, true), left(100.0, false));
    assert!(base >= 60, "base decay is slow ({base} left)");
    assert!(rot <= 50, "a dormant tape rots ({rot} left)");
    assert!(awake >= 60, "awake cells are unaffected ({awake} left)");
}

/// shielding.light_cover: face cover reduces light absorption by this weight
/// (1 = the spec's exposure, 0 = bonded neighbours do not shade).
#[test]
fn t1_light_cover() {
    let take = |cover: f64| {
        let mut cfg = deterministic_cfg(2);
        cfg.shielding.light_cover = cover;
        let (p, mut st) = blank(&cfg);
        let c = p.geom.index([5, 5, 0]);
        put(&mut st, &p, c, &[0x41, 0x41], 0, 1000);
        for d in 0..4u32 {
            let t = p.nbr[c * 6 + d as usize] as usize;
            put(&mut st, &p, t, &[0x41, 0x41], 0, 1000);
            st.hdr[c * HDR + H_FLAGS] |= 1 << (BOND_SHIFT + d);
            st.hdr[t * HDR + H_FLAGS] |= 1 << (BOND_SHIFT + (d ^ 1));
        }
        *st.fm(c, F_F) = 20_000;
        primordia_core::phases::absorb_upkeep(&mut st, &p);
        20_000 - st.f(c, F_F)
    };
    let (shaded, half, open) = (take(1.0), take(0.5), take(0.0));
    assert_eq!(shaded, 0, "fully bonded cell is fully shaded");
    assert!(open > 0 && half > 0 && half < open, "half {half}, open {open}");
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
