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
    let c = if paper {
        let mut c = Config::preset("paper-0d").unwrap();
        c.world.size = vec![8192];
        c
    } else {
        cfg(dims)
    };
    let p = Params::new(&c);
    let gpu = match GpuExec::new(&p) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping GPU parity: {e:#}");
            return;
        }
    };
    let mut a = Sim::new(c.clone(), Box::new(CpuExec)).unwrap();
    let mut b = Sim::new(c.clone(), Box::new(gpu)).unwrap();
    if !paper {
        seed(&mut a);
        seed(&mut b);
    }
    for e in 0..epochs {
        a.step();
        b.step();
        if let Some(d) = a.st.first_diff(&b.st) {
            panic!("dims {dims}: GPU != CPU at epoch {e}: {:?}", d);
        }
        assert_eq!(a.st.hash(), b.st.hash());
    }
    let occ = (0..a.st.n).filter(|&c| a.st.occupied(c)).count();
    eprintln!("dims {dims}: {epochs} epochs identical, {occ} occupied, hash {:016x}", a.st.hash());
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
