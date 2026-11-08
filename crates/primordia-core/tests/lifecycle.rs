//! Lifecycle mechanisms: individual.resume (execution continues across
//! slices), bonds.protect (tapes closed to bonded neighbours after a
//! construction window) and energy.copy_rest / copy_absorb (germ-soma
//! trade-off). All default off; the GPU parity and twin tests cover them on.
use primordia_core::config::PairOrder;
use primordia_core::isa::op;
use primordia_core::{Config, CpuExec, Params, Sim, State, exec, phases};
use primordia_core::params::*;

/// The hand-written ancestor for resume worlds (`runs/seed-m17.jsonl`): reset
/// the heads, TUMBLE, ALLOC a child of its own length, copy itself with a
/// stack counter, start over. Achiral, so its mirror image is itself.
const ANCESTOR: [u8; 20] = [
    0x42, op::NEG, op::SETH0, // h0 = 0
    op::LEN, 0x46, op::SUB, op::SETH1, // h1 = len (start of the partner)
    op::TUMBLE, op::LEN, op::ALLOC, op::LEN, // child; counter = len
    op::CPY, op::H0INC, op::H1INC, op::DEC, op::DUP, op::SKIPZ, 0xC5, // copy loop
    op::DROP, 0xD2, // back to the start
];

fn cfg() -> Config {
    let mut c = Config::preset("primordia-2d-sun").unwrap();
    c.world.size = vec![9, 9];
    c.world.init_occupancy = 0.0;
    c.individual.random_initial_facing = false;
    c.individual.pair_order = PairOrder::SelfFirst;
    c.individual.step_budget = 1024;
    c.mutation.enabled = false;
    c.decay.enabled = false;
    c.energy.sun.enabled = false;
    c.energy.vents.enabled = false;
    c.energy.upkeep.base = 0.0;
    c.energy.upkeep.per_byte = 0.0;
    c.energy.absorb_alpha = 0.0;
    c.shielding.enabled = false;
    c.matter.enabled = false;
    c.bonds.birth_bond = 0.0;
    c
}
fn put(s: &mut State, p: &Params, at: usize, g: &[u8], len: usize) {
    let base = at * s.lmax;
    s.tapes[base..base + s.lmax].iter_mut().for_each(|x| *x = 0);
    s.tapes[base..base + g.len()].copy_from_slice(g);
    *s.hm(at, H_LEN) = len as u32;
    *s.hm(at, H_FLAGS) = FL_OCC | (p.orient.valid[2][0] << ORIENT_SHIFT);
    *s.hm(at, H_E) = 10_000;
}
fn facing_cell(s: &State, p: &Params, at: usize) -> usize {
    exec::nb(s, p, at, p.orient.f[s.orient(at) as usize]).unwrap()
}
/// One slice for `at` with `energy` to spend (flags reset as at an epoch start).
fn slice(s: &mut State, p: &Params, at: usize, energy: u32) {
    *s.hm(at, H_E) = energy;
    *s.hm(at, H_FLAGS) &= !(FL_EXEC | FL_HALT | FL_SLEEPING);
    let flags = exec::flag_snapshot(s);
    exec::exec_slice(s, p, &flags, at, None, None);
}
fn nonzero(s: &State, c: usize) -> usize {
    s.tapes[c * s.lmax..(c + 1) * s.lmax].iter().filter(|&&b| b != 0).count()
}

/// The audit's probe: a 64-byte copier with 256 energy per slice copies 51
/// bytes. Restarting, it recopies the same prefix forever; resuming, it
/// finishes in the second slice.
#[test]
fn resume_finishes_a_copy_that_needs_several_slices() {
    let mut genome = vec![op::CPY, op::H0INC, op::H1INC, 0xC2];
    genome.resize(64, 0x70);
    for resume in [false, true] {
        let mut c = cfg();
        c.individual.resume = resume;
        let p = Params::new(&c);
        let mut s = State::init(&c, &p);
        let a = p.geom.index([4, 4, 0]);
        put(&mut s, &p, a, &genome, 64);
        let b = facing_cell(&s, &p, a);
        put(&mut s, &p, b, &[], 64);
        let mut copied = vec![];
        for _ in 0..3 {
            slice(&mut s, &p, a, 256);
            copied.push(nonzero(&s, b));
        }
        if resume {
            assert!(copied[0] < 64 && copied[1] == 64, "resume {copied:?}");
            assert_eq!(&s.tapes[b * s.lmax..b * s.lmax + 64], &genome[..]);
        } else {
            assert!(copied[0] < 64 && copied.iter().all(|&k| k == copied[0]), "restart {copied:?}");
        }
    }
}

/// The saved context includes the stack (the ancestor's loop counter), and a
/// slice that ends on HALT, a stack fault or SLEEP saves accordingly.
#[test]
fn resume_keeps_the_stack_and_restarts_after_halt_or_faults() {
    let mut c = cfg();
    c.individual.resume = true;
    c.individual.step_budget = 7; // slices end mid-loop
    let p = Params::new(&c);
    let mut s = State::init(&c, &p);
    let a = p.geom.index([4, 4, 0]);
    put(&mut s, &p, a, &ANCESTOR, ANCESTOR.len());
    for _ in 0..40 {
        slice(&mut s, &p, a, 10_000);
    }
    let kids: Vec<usize> = (0..s.n).filter(|&x| x != a && s.tape(x) == ANCESTOR).collect();
    assert!(!kids.is_empty(), "a complete child despite 7-step slices");
    // without resume the same budget never gets past the setup
    let mut c2 = c.clone();
    c2.individual.resume = false;
    let p2 = Params::new(&c2);
    let mut s2 = State::init(&c2, &p2);
    put(&mut s2, &p2, a, &ANCESTOR, ANCESTOR.len());
    for _ in 0..40 {
        slice(&mut s2, &p2, a, 10_000);
    }
    assert!((0..s2.n).all(|x| x == a || s2.tape(x) != ANCESTOR));

    for (prog, ctx, ip) in [
        (vec![0x41, op::HALT, 0x45], false, 0),
        (vec![op::DROP, 0x45], false, 0),
        (vec![0x41, op::SLEEP, 0x45, 0x46], true, 2),
    ] {
        let mut s = State::init(&c, &p);
        put(&mut s, &p, a, &prog, prog.len());
        slice(&mut s, &p, a, 10_000);
        assert_eq!(s.flags(a) & FL_CTX != 0, ctx, "{prog:x?}");
        if ctx {
            assert_eq!(s.h(a, H_CTX) & 0xFFFF, ip);
        }
    }
}

/// Parent -> child -> grandchild under the live mechanisms, each child
/// isolated before it reproduces, with a little energy per epoch.
#[test]
fn ancestor_reproduces_for_three_generations_in_isolation() {
    let mut c = cfg();
    c.world.size = vec![15, 15];
    c.individual.resume = true;
    c.bonds.protect = true;
    c.bonds.construct_window = 64;
    c.bonds.birth_bond = 0.5;
    c.energy.copy_rest = 50;
    c.energy.copy_absorb = 0.25;
    c.energy.init_f = 4000;
    c.energy.absorb_alpha = 0.05;
    c.energy.wake_energy = 40;
    c.energy.dissipation = 0.0;
    c.validate().unwrap();
    let p = Params::new(&c);
    let at = p.geom.index([7, 7, 0]);
    let mut sim = Sim::new(c.clone(), Box::new(CpuExec)).unwrap();
    put(&mut sim.st, &p, at, &ANCESTOR, ANCESTOR.len());
    *sim.st.hm(at, H_E) = 300;
    sim.st.reset_ledger(&p);
    for generation in 0..3 {
        let mut child = None;
        for _ in 0..400 {
            sim.step();
            child = (0..sim.st.n).find(|&x| x != at && sim.st.tape(x) == ANCESTOR);
            if child.is_some() {
                break;
            }
        }
        let child = child.unwrap_or_else(|| panic!("generation {generation}: no complete child"));
        assert!(sim.st.h(at, H_REST) > 0, "the parent is in its rest period");
        assert_eq!(sim.st.ledger.violations, 0);
        let mut isolated = Sim::new(c.clone(), Box::new(CpuExec)).unwrap();
        isolated.st.epoch = sim.st.epoch;
        let l = sim.st.lmax;
        isolated.st.tapes[at * l..(at + 1) * l].copy_from_slice(&sim.st.tapes[child * l..(child + 1) * l]);
        isolated.st.hdr[at * HDR..(at + 1) * HDR].copy_from_slice(&sim.st.hdr[child * HDR..(child + 1) * HDR]);
        isolated.st.hdr[at * HDR + H_FLAGS] &= !BOND_MASK;
        *isolated.st.hm(at, H_E) = 300;
        isolated.st.reset_ledger(&p);
        sim = isolated;
    }
}

/// bonds.protect: a bonded neighbour writes into a young cell but not into
/// one past its construction window; an unbonded writer is unaffected.
#[test]
fn bonded_writes_need_the_construction_window() {
    let copier = [op::CPY, op::H0INC, op::H1INC, 0xC2];
    for (bonded, age, writes) in [(true, 10, true), (true, 300, false), (false, 300, true)] {
        let mut c = cfg();
        c.bonds.protect = true;
        c.bonds.construct_window = 256;
        let p = Params::new(&c);
        let mut s = State::init(&c, &p);
        s.epoch = 1000;
        let a = p.geom.index([4, 4, 0]);
        put(&mut s, &p, a, &copier, 4);
        let b = facing_cell(&s, &p, a);
        put(&mut s, &p, b, &[], 4);
        *s.hm(b, H_BIRTH) = 1000 - age;
        if bonded {
            let d = p.orient.f[s.orient(a) as usize];
            *s.hm(a, H_FLAGS) |= 1 << (BOND_SHIFT + d);
            *s.hm(b, H_FLAGS) |= 1 << (BOND_SHIFT + (d ^ 1));
        }
        slice(&mut s, &p, a, 100);
        assert_eq!(nonzero(&s, b) > 0, writes, "bonded {bonded} age {age}");
    }
}

/// energy.copy_rest: writing into a neighbour starts a period in which the
/// writer absorbs copy_absorb of the free energy a non-copier absorbs.
#[test]
fn reproduction_reduces_absorption_for_the_rest_period() {
    let mut c = cfg();
    c.energy.copy_rest = 3;
    c.energy.copy_absorb = 0.25;
    c.energy.absorb_alpha = 0.5;
    let p = Params::new(&c);
    let mut s = State::init(&c, &p);
    let (a, z) = (p.geom.index([2, 2, 0]), p.geom.index([6, 6, 0]));
    put(&mut s, &p, a, &[op::CPY, op::HALT], 2);
    put(&mut s, &p, z, &[op::HALT], 2);
    let b = facing_cell(&s, &p, a);
    put(&mut s, &p, b, &[], 2);
    slice(&mut s, &p, a, 100);
    slice(&mut s, &p, z, 100);
    assert_eq!((s.h(a, H_REST), s.h(z, H_REST)), (3, 0));
    // compare with the writer's own absorption once the period is over
    // (its neighbour shades it, so an isolated cell is not the baseline)
    let mut gains = vec![];
    for _ in 0..5 {
        *s.fm(a, F_F) = 40_000;
        *s.hm(a, H_E) = 1000;
        phases::absorb_upkeep(&mut s, &p);
        gains.push((s.h(a, H_E) - 1000) as f64);
    }
    assert_eq!(gains[3], gains[4]);
    for (i, g) in gains.iter().enumerate() {
        let want = if i < 3 { 0.25 } else { 1.0 };
        assert!((g / gains[4] - want).abs() < 0.01, "epoch {i}: {gains:?}");
    }
    assert_eq!(s.h(a, H_REST), 0);
}

/// Options added later serialize only when set, so existing configurations
/// (and the hashes their snapshots record) are unchanged; version 3 snapshots
/// keep the saved contexts.
#[test]
fn new_options_keep_old_hashes_and_snapshots_keep_contexts() {
    let base = Config::preset("primordia-2d-sun").unwrap().to_toml();
    for key in ["resume", "protect", "construct_window", "copy_rest", "copy_absorb", "private", "max_age_spread", "predation"] {
        assert!(!base.contains(key), "{key} serialized at its default");
    }
    let mut c = cfg();
    c.individual.resume = true;
    c.energy.copy_rest = 9;
    let t = c.to_toml();
    assert!(t.contains("resume = true") && t.contains("copy_rest = 9"));
    let p = Params::new(&c);
    let mut s = State::init(&c, &p);
    let a = p.geom.index([4, 4, 0]);
    put(&mut s, &p, a, &ANCESTOR, ANCESTOR.len());
    let mut c7 = c.clone();
    c7.individual.step_budget = 7;
    let p7 = Params::new(&c7);
    slice(&mut s, &p7, a, 10_000);
    assert!(s.flags(a) & FL_CTX != 0 && s.h(a, H_REST) == 0);
    s.hdr[a * HDR + H_REST] = 5;
    let path = std::env::temp_dir().join(format!("primordia-{}-v3.psnap", std::process::id()));
    s.write_snapshot(&path, 7).unwrap();
    assert_eq!(State::snapshot_header(&path).unwrap().0, 4);
    let (back, _) = State::read_snapshot(&path, &p).unwrap();
    assert_eq!(back.hdr, s.hdr);
    std::fs::remove_file(path).unwrap();
}

/// The M17 seed events carry exactly the ancestor tested above.
#[test]
fn m17_seed_events_carry_the_tested_ancestor() {
    let seeds = include_str!("../../../runs/seed-m17.jsonl");
    for line in seeds.lines().filter(|l| !l.trim().is_empty()) {
        let ev: primordia_core::events::Event = serde_json::from_str(line).unwrap();
        let g = primordia_core::isa::parse_hex_genome(ev.params["genome"].as_str().unwrap()).unwrap();
        assert_eq!(g, ANCESTOR);
    }
    let c = Config::from_file(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../runs/live-m17.toml"))).unwrap();
    assert!(c.individual.resume && c.bonds.protect && c.energy.copy_rest > 0);
    assert_eq!((c.individual.pair_order, c.individual.random_initial_facing), (PairOrder::SelfFirst, false));
}

// ---- regressions from the 2026-11 code scan

/// An op that pops its operand before paying (ALLOC here) and runs out of
/// energy is retried next slice with the same stack, not the value below.
#[test]
fn failed_payment_is_retried_with_the_same_operand() {
    let mut c = cfg();
    c.individual.resume = true;
    let p = Params::new(&c);
    let mut s = State::init(&c, &p);
    let a = p.geom.index([4, 4, 0]);
    put(&mut s, &p, a, &[0x49, 0x44, op::ALLOC, op::HALT], 4); // PUSH 9, PUSH 4, ALLOC, HALT
    let b = facing_cell(&s, &p, a);
    slice(&mut s, &p, a, 22); // pays both PUSHes, not ALLOC
    assert!(!s.occupied(b) && s.flags(a) & FL_CTX != 0);
    assert_eq!(s.h(a, H_CTX) >> 24, 2, "both operands still on the saved stack");
    slice(&mut s, &p, a, 10_000);
    assert_eq!((s.occupied(b), s.len(b)), (true, 4));
}

/// A slice that ends right after a long jump resumes where the same jump
/// leads within one slice (the saved IP used to be cut to 16 bits).
#[test]
fn a_long_jump_resumes_at_the_same_address() {
    // 7, <<4, +15, <<4, +15, <<4, +15 = 32767; JREL at 10 targets 32777 = 5 (mod 12)
    let prog = [0x47, op::SHL4, 0x4F, op::ADD, op::SHL4, 0x4F, op::ADD, op::SHL4, 0x4F, op::ADD, op::JREL, 0x70];
    let first_ip_after = |budget: u32, split: bool| -> u32 {
        let mut c = cfg();
        c.individual.resume = true;
        c.individual.step_budget = budget;
        let p = Params::new(&c);
        let mut s = State::init(&c, &p);
        let a = p.geom.index([4, 4, 0]);
        put(&mut s, &p, a, &prog, prog.len());
        if split {
            slice(&mut s, &p, a, 10_000);
        }
        *s.hm(a, H_FLAGS) &= !(FL_EXEC | FL_HALT);
        let flags = exec::flag_snapshot(&s);
        let mut tr = vec![];
        exec::exec_slice(&mut s, &p, &flags, a, None, Some(&mut tr));
        tr[if split { 0 } else { 11 }].ip
    };
    assert_eq!(first_ip_after(12, false), 5);
    assert_eq!(first_ip_after(11, true), 5);
}

/// The rest period starts only when a byte is actually written into the
/// other cell (here a write fails for want of monomers).
#[test]
fn a_failed_write_starts_no_rest_period() {
    let mut c = cfg();
    c.matter.enabled = true;
    c.matter.pool_l = 0;
    c.matter.pool_d = 0;
    c.matter.pool_a = 0;
    c.energy.copy_rest = 5;
    let p = Params::new(&c);
    let mut s = State::init(&c, &p);
    let a = p.geom.index([4, 4, 0]);
    put(&mut s, &p, a, &[op::CPY, op::HALT], 2);
    let b = facing_cell(&s, &p, a);
    put(&mut s, &p, b, &[], 2);
    for x in [a, b] {
        for cl in 0..3 {
            *s.fm(x, p.f_pool(cl)) = 0;
        }
    }
    slice(&mut s, &p, a, 100);
    assert_eq!((nonzero(&s, b), s.h(a, H_REST)), (0, 0));
}

/// Inside a fully bonded body, chemistry and monomers diffuse gamma times
/// faster but no faster than the stability bound: a checkerboard decays
/// (kappa * gamma * faces was 2 here, and the pattern grew to 0 / 2000).
#[test]
fn bonded_transport_stays_stable() {
    let mut c = Config::preset("primordia-2d-sun").unwrap();
    c.world.size = vec![6, 6];
    c.world.init_occupancy = 0.0;
    c.chemistry.kappa_c = 0.125;
    c.bonds.gamma = 4.0;
    c.energy.sun.enabled = false;
    c.energy.vents.enabled = false;
    c.validate().unwrap();
    let p = Params::new(&c);
    let mut s = State::init(&c, &p);
    let f = p.f_chem(0, 0);
    for x in 0..s.n {
        s.hdr[x * HDR + H_FLAGS] = FL_OCC | (0xF << BOND_SHIFT);
        let q = p.geom.coords(x);
        s.fields[x * s.nf + f] = if (q[0] + q[1]) % 2 == 0 { 1100 } else { 900 };
    }
    let dev = |s: &State| (0..s.n).map(|x| (s.fields[x * s.nf + f] as i64 - 1000).abs()).max().unwrap();
    let mut last = dev(&s);
    for _ in 0..8 {
        phases::transport(&mut s, &p);
        let d = dev(&s);
        assert!(d <= last, "deviation grew from {last} to {d}");
        last = d;
    }
    assert!(last < 100);
}

/// `--set`-style overrides merge a table value into its section, and a live
/// edit that changes what the ledger measures does not count as a violation.
#[test]
fn table_overrides_merge_and_live_edits_rebase_the_ledger() {
    let base = Config::preset("primordia-2d-sun").unwrap();
    let mut t = toml::Table::new();
    t.insert("share_beta".into(), toml::Value::Float(0.05));
    let c = base.with_override("energy", toml::Value::Table(t)).unwrap();
    assert_eq!((c.energy.share_beta, c.energy.absorb_alpha), (0.05, base.energy.absorb_alpha));

    let mut c = Config::preset("primordia-2d-sun").unwrap();
    c.world.size = vec![9, 9];
    c.world.init_occupancy = 0.5;
    assert!(c.chemistry.enabled && c.chemistry.species.len() == 4);
    let mut sim = Sim::new(c, Box::new(CpuExec)).unwrap();
    sim.push_event(primordia_core::events::Event { epoch: 3, kind: "set".into(), params: serde_json::json!({"key": "chemistry.mass", "value": [2, 2, 4, 4]}) });
    for _ in 0..6 {
        sim.step();
    }
    assert!(sim.rejected.is_empty(), "{:?}", sim.rejected);
    assert_eq!(sim.st.ledger.violations, 0);
}

/// shielding.private: any neighbour may write into a cell under construction,
/// none into an older one; the cell still writes its own tape.
#[test]
fn private_tapes_are_written_only_while_under_construction() {
    let copier = [op::CPY, op::H0INC, op::H1INC, 0xC2];
    for (age, writes) in [(10, true), (300, false)] {
        let mut c = cfg();
        c.shielding.private = true;
        c.bonds.construct_window = 256;
        let p = Params::new(&c);
        let mut s = State::init(&c, &p);
        s.epoch = 1000;
        let a = p.geom.index([4, 4, 0]);
        put(&mut s, &p, a, &copier, 4);
        *s.hm(a, H_BIRTH) = 0; // the writer is old: its own tape stays writable
        let b = facing_cell(&s, &p, a);
        put(&mut s, &p, b, &[], 4);
        *s.hm(b, H_BIRTH) = 1000 - age;
        slice(&mut s, &p, a, 100);
        assert_eq!(nonzero(&s, b) > 0, writes, "age {age}");
    }
    // self-writes: WR into the cell's own tape is never blocked
    let mut c = cfg();
    c.shielding.private = true;
    let p = Params::new(&c);
    let mut s = State::init(&c, &p);
    s.epoch = 1000;
    let a = p.geom.index([4, 4, 0]);
    put(&mut s, &p, a, &[0x45, 0x43, op::SETH1, op::WR, op::HALT, 0, 0, 0], 8); // PUSH5, PUSH3, SETH1 (h1 = 2+3), WR 5 at 5
    slice(&mut s, &p, a, 100);
    assert_eq!(s.tape(a)[5], 5);
}

/// individual.max_age_spread: lifespans are uniform over max_age +- spread,
/// fixed per individual, so a cohort born together dies over a range.
#[test]
fn lifespans_spread_around_max_age() {
    let mut c = cfg();
    c.individual.max_age = 20_000;
    c.individual.max_age_spread = 0.25;
    let p = Params::new(&c);
    assert_eq!(p.age_span, 5_000);
    let lives: Vec<u32> = (0..10_000u32).map(|i| p.lifespan(primordia_core::rng::Rng::new(9, 0, 0).u32(i, 0, 0) | 1)).collect();
    assert!(lives.iter().all(|&l| (15_000..=25_000).contains(&l)));
    let (lo, hi) = (*lives.iter().min().unwrap(), *lives.iter().max().unwrap());
    assert!(lo < 15_100 && hi > 24_900, "range {lo}..{hi}");
    let mean = lives.iter().map(|&l| l as f64).sum::<f64>() / lives.len() as f64;
    assert!((mean - 20_000.0).abs() < 150.0, "mean {mean}");
    c.individual.max_age_spread = 0.0;
    assert_eq!(Params::new(&c).lifespan(12345), 20_000);
}

/// shielding.predation: an established cell can be eaten, but only whole; it
/// then dies at the end of the epoch. Writes into it stay blocked, and a
/// newborn is still digested byte by byte.
#[test]
fn predators_eat_established_cells_whole() {
    let attack = |n: u8, age: u32, predation: bool| -> (usize, bool, u32) {
        let mut c = cfg();
        c.shielding.private = true;
        c.shielding.predation = predation;
        c.matter.enabled = true;
        let p = Params::new(&c);
        let mut s = State::init(&c, &p);
        s.epoch = 1000;
        let a = p.geom.index([4, 4, 0]);
        put(&mut s, &p, a, &[0x40 + n, op::DIGEST, op::HALT], 3);
        *s.hm(a, H_BIRTH) = 0;
        let b = facing_cell(&s, &p, a);
        put(&mut s, &p, b, &[0x20, 0x22, 0x24, 0x26, 0x28, 0x2A], 6); // L monomers
        *s.hm(b, H_BIRTH) = 1000 - age;
        let pool_before = s.f(a, p.f_pool(0));
        slice(&mut s, &p, a, 1000);
        let gained = s.f(a, p.f_pool(0)) - pool_before;
        let left = nonzero(&s, b);
        phases::mutate_decay(&mut s, &p, None);
        assert_eq!(gained as usize, 6 - left);
        (left, s.occupied(b), gained)
    };
    assert_eq!(attack(6, 300, true), (0, false, 6), "eaten whole: dead at the end of the epoch");
    assert_eq!(attack(5, 300, true), (6, true, 0), "too small a bite: nothing happens");
    assert_eq!(attack(6, 300, false), (6, true, 0), "privacy without predation");
    assert_eq!(attack(3, 10, true), (3, true, 3), "a newborn is digested byte by byte");

    // writes into an established cell stay blocked under predation
    let mut c = cfg();
    c.shielding.private = true;
    c.shielding.predation = true;
    let p = Params::new(&c);
    let mut s = State::init(&c, &p);
    s.epoch = 1000;
    let a = p.geom.index([4, 4, 0]);
    put(&mut s, &p, a, &[op::CPY, op::H0INC, op::H1INC, 0xC2], 4);
    *s.hm(a, H_BIRTH) = 0;
    let b = facing_cell(&s, &p, a);
    put(&mut s, &p, b, &[], 4);
    slice(&mut s, &p, a, 100);
    assert_eq!(nonzero(&s, b), 0);
}
