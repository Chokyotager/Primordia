//! T11 (release): one minimal replicator in a random paper-0d soup.
//! Run with: cargo test --release -p primordia-gpu --test takeover -- --ignored --nocapture
//! PRIMORDIA_T11_SEEDS (default 20), PRIMORDIA_T11_EPOCHS (default 128).

use primordia_core::events::Event;
use primordia_core::{Config, Params, Sim, isa};

fn frac_with_replicator(sim: &Sim) -> f64 {
    let r = isa::MINIMAL_REPLICATOR;
    let n = (0..sim.st.n).filter(|&c| sim.st.tape(c).windows(4).any(|w| w == r)).count();
    n as f64 / sim.st.n as f64
}

#[test]
#[ignore]
fn t11_seeded_takeover() {
    let seeds: u64 = std::env::var("PRIMORDIA_T11_SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let epochs: u64 = std::env::var("PRIMORDIA_T11_EPOCHS").ok().and_then(|s| s.parse().ok()).unwrap_or(128);
    let mut takeovers = 0;
    for seed in 1..=seeds {
        let mut c = Config::preset("paper-0d").unwrap();
        c.seed = 1000 + seed;
        let p = Params::new(&c);
        let be = primordia_gpu::backend(&p, true);
        let mut sim = Sim::new(c, be).unwrap();
        sim.push_event(Event { epoch: 0, kind: "seed".into(), params: serde_json::json!({"genome": isa::to_hex(&isa::MINIMAL_REPLICATOR), "center": [0], "radius": 1}) });
        for _ in 0..epochs {
            sim.step();
        }
        let f = frac_with_replicator(&sim);
        let took = f > 0.5;
        takeovers += took as u32;
        eprintln!("seed {seed:>3}: fraction of tapes carrying 12 13 15 c2 after {epochs} epochs = {f:.3} {}", if took { "TAKEOVER" } else { "" });
    }
    let rate = takeovers as f64 / seeds as f64;
    eprintln!("T11: takeover in {takeovers}/{seeds} seeds ({:.0}%), spec expects 10-40% (paper's seeded BFF: 22%)", 100.0 * rate);
}
