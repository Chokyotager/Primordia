//! Primordia core: config, PISA-1 ISA, geometry, Philox RNG and the CPU
//! golden model of every epoch phase.

pub mod config;
pub mod events;
pub mod exec;
pub mod fixed;
pub mod geom;
pub mod isa;
pub mod params;
pub mod phases;
pub mod rng;
pub mod state;
pub mod wgsl;
pub mod world;

pub use config::Config;
pub use params::Params;
pub use state::State;
pub use world::{CpuExec, ExecBackend, Sim};
