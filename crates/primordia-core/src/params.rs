//! Derived integer parameters and tables computed once from a `Config`.
//! Everything the state path needs is integer / fixed-point here.

use crate::config::*;
use crate::fixed::*;
use crate::geom::{Geometry, Orient};
use crate::isa;

pub const NONE: u32 = 0xFFFF_FFFF;

/// Header words per cell.
pub const HDR: usize = 9;
pub const H_LEN: usize = 0;
pub const H_FLAGS: usize = 1;
pub const H_E: usize = 2;
pub const H_SLEEP: usize = 3;
pub const H_LINEAGE: usize = 4;
pub const H_BIRTH: usize = 5;
pub const H_PARENT: usize = 6;
pub const H_DEPTH: usize = 7;
pub const H_ZERO: usize = 8;

pub const FL_OCC: u32 = 1;
pub const FL_HALT: u32 = 2;
pub const FL_EXEC: u32 = 4;
pub const FL_NEWBORN: u32 = 8;
pub const ORIENT_SHIFT: u32 = 8;
pub const ORIENT_MASK: u32 = 0xFF << 8;
pub const BOND_SHIFT: u32 = 16;
pub const BOND_MASK: u32 = 0x3F << 16;

/// Site words per cell (owned by the lattice site, not the individual).
pub const SITE: usize = 4;
pub const S_SPENT: usize = 0;
pub const S_VOTE: usize = 1;
pub const S_WALL: usize = 2;
pub const S_RXN: usize = 3;
pub const WALL_REFLECT: u32 = 1;
pub const WALL_ABSORB: u32 = 2;

#[derive(Clone, Copy, Debug)]
pub struct Reaction {
    pub r: [u32; 2],
    pub p: [u32; 2],
    pub energy: i32,
    /// per-hand energy (Q16) including parity bias
    pub energy_q16: [i64; 2],
    pub catalyzable: bool,
    pub photo: bool,
    pub rate_q32: u64,
    pub arr: usize,
}

#[derive(Clone, Debug)]
pub struct Params {
    pub geom: Geometry,
    pub orient: Orient,
    pub seed: u64,
    pub lmax: usize,
    pub l_init: u32,
    pub k: usize,
    pub s: usize,
    pub nf: usize,
    pub budget: u32,
    pub pair_concat: bool,
    pub pair_random: bool,
    pub random_facing: bool,
    pub dead_after: u32,

    pub energy_on: bool,
    pub chem_on: bool,
    pub matter_on: bool,
    pub sig_on: bool,
    pub shield_on: bool,
    pub bonds_on: bool,
    pub mut_on: bool,
    pub decay_on: bool,
    pub stereo: bool,
    pub stereo_copy: bool,
    pub shield_byte: bool,

    pub costs: [u32; 256],
    pub move_occ_cost: u32,
    /// Q32 write probabilities; u32::MAX means always.
    pub pwrite: [u32; 64],
    pub upkeep_base_q16: u64,
    pub upkeep_byte_q16: u64,
    pub absorb_q16: u64,
    pub beta_q16: i64,
    pub heat_q16: u64,
    pub wake_energy: u32,
    pub e_scale: u32,
    pub f_scale: u32,
    pub kappa_f: i64,
    pub kappa_t: i64,
    pub kappa_c: i64,
    pub kappa_m: i64,
    pub kappa_s: i64,
    pub gamma_q16: i64,
    pub diss_q16: u64,
    pub relax_q16: i64,
    pub t_amb: u32,
    pub sig_decay_q16: u64,
    pub sig_emit: u32,
    pub w_occ_q16: u64,

    pub sun_on: bool,
    pub sun_i0_q16: u64,
    pub sun_grad_q16: i64,
    pub sun_atten_q16: u64,
    pub sun_heat_q16: u64,
    pub day_period: u64,
    pub season_period: u64,
    pub season_amp: f64,
    pub vents_on: bool,
    pub chem_only: bool,

    pub reactions: Vec<Reaction>,
    pub cat: Vec<u32>,
    pub mass: Vec<u32>,
    pub n_cat: u32,
    pub eta_min_q16: u64,
    pub ci_q16: u64,
    pub fold_w: u32,
    pub sigma_min_q16: u64,
    pub rxn_heat_q16: u64,
    pub sink_q16: u64,
    pub rac_q32: u64,
    pub rac_bias_q16: u64,

    /// cumulative copy-error thresholds: sub, sub+ins, sub+ins+del (saturating)
    pub p_sub: u32,
    pub p_ins: u32,
    pub p_del: u32,
    pub mu_bg_q32: u64,
    pub mu_uv_q32: u64,
    pub mu_flip_q32: u64,
    pub decay_q32: u64,
    pub decay_light_q16: u64,
    pub bond_break_q32: u64,
    pub vote_q16: u64,
    pub max_body: u32,

    pub arr: Vec<Vec<u32>>,
    pub arr_mut: usize,
    pub arr_rac: usize,
    pub arr_decay: usize,
    pub arr_bond: usize,
    pub pois: Vec<u32>,
    /// Raw lattice neighbour table, n x 6, NONE where there is no neighbour.
    pub nbr: Vec<u32>,
    pub x4_species: Option<usize>,
    pub vent_cfg: VentCfg,
    pub map_kind: u32,
    pub map_a: f64,
    pub map_r0: f64,
    pub hash_every: u64,
}

fn q16s(x: f64) -> i64 {
    to_q16s(x)
}

impl Params {
    pub fn new(c: &Config) -> Params {
        let geom = Geometry::new(c.world.dims, &c.world.size, &c.world.boundary);
        let k = c.n_species();
        let s = c.n_signals();
        let pair_concat = c.pair_concat();
        let energy_on = c.energy.enabled && !pair_concat;
        let mut costs = [0u32; 256];
        if energy_on {
            for b in 0..=255u8 {
                costs[b as usize] = isa::default_cost(b);
            }
            for (name, v) in &c.costs {
                if let Some(b) = isa::parse_opcode_name(name) {
                    costs[b as usize] = *v;
                }
            }
        }
        let mut pwrite = [u32::MAX; 64];
        if c.shielding.enabled && !pair_concat {
            for b in 0..8 {
                for o in 0..8 {
                    let p = det_powi(1.0 - c.shielding.s_bond, b) * det_powi(1.0 - c.shielding.s_occupied, o);
                    pwrite[(b * 8 + o) as usize] = to_q32(p).min(u32::MAX as u64) as u32;
                }
            }
        }
        let mut arr = Vec::new();
        let t_ref = c.temperature.t_ref;
        let mut add_arr = |a: f64| {
            arr.push(arrhenius_table(a, t_ref));
            arr.len() - 1
        };
        let arr_mut = add_arr(c.mutation.activation);
        let arr_rac = add_arr(c.chirality.racemization.activation);
        let arr_decay = add_arr(c.decay.activation);
        let arr_bond = add_arr(c.bonds.break_activation);
        let mut reactions = Vec::new();
        let mut cat = Vec::new();
        let mut mass = Vec::new();
        if c.chemistry.enabled {
            let sp = |n: &String| c.chemistry.species.iter().position(|x| x == n).unwrap() as u32;
            mass = c.chemistry.mass.clone();
            for r in &c.chemistry.reactions {
                let a = add_arr(r.activation);
                let bias = c.chirality.bias.reaction_energy;
                let rx = Reaction {
                    r: [sp(&r.r[0]), r.r.get(1).map(sp).unwrap_or(NONE)],
                    p: [sp(&r.p[0]), r.p.get(1).map(sp).unwrap_or(NONE)],
                    energy: r.energy,
                    energy_q16: [q16s(r.energy as f64 * (1.0 + bias)), q16s(r.energy as f64)],
                    catalyzable: r.catalyzable,
                    photo: r.kind == ReactionKind::Photolysis,
                    rate_q32: to_q32(r.rate),
                    arr: a,
                };
                if rx.catalyzable && rx.energy > 0 {
                    cat.push(reactions.len() as u32);
                }
                reactions.push(rx);
            }
        }
        let e = &c.energy;
        let m = &c.mutation;
        let cp = |p: f64| to_q32(p).min(u32::MAX as u64) as u32;
        let copy_on = m.enabled && !pair_concat;
        Params {
            orient: Orient::new(),
            seed: c.seed,
            lmax: c.individual.l_max as usize,
            l_init: c.individual.l_init,
            k,
            s,
            nf: 2 + 2 * k + 3 + 2 * s,
            budget: c.individual.step_budget,
            pair_concat,
            pair_random: c.individual.pair_order == PairOrder::Random,
            random_facing: c.individual.random_initial_facing,
            dead_after: c.individual.dead_after,
            energy_on,
            chem_on: c.chemistry.enabled,
            matter_on: c.matter.enabled && !pair_concat,
            sig_on: c.signals.enabled,
            shield_on: c.shielding.enabled && !pair_concat,
            bonds_on: c.bonds.enabled && !pair_concat,
            mut_on: m.enabled,
            decay_on: c.decay.enabled,
            stereo: c.chirality.stereo,
            stereo_copy: c.chirality.stereo_copy && c.matter.enabled && !pair_concat,
            shield_byte: c.shielding.granularity == Granularity::Byte,
            costs,
            move_occ_cost: if energy_on { 2 * costs[isa::op::MOVE as usize] } else { 0 },
            pwrite,
            upkeep_base_q16: to_q16(e.upkeep.base) as u64,
            upkeep_byte_q16: to_q16(e.upkeep.per_byte) as u64,
            absorb_q16: to_q16(e.absorb_alpha) as u64,
            beta_q16: q16s(e.share_beta),
            heat_q16: to_q16(e.heat_per_energy) as u64,
            wake_energy: e.wake_energy,
            e_scale: e.sense_energy_scale.max(1),
            f_scale: e.sense_free_scale.max(1),
            kappa_f: q16s(e.kappa_f),
            kappa_t: q16s(c.temperature.kappa_t),
            kappa_c: q16s(c.chemistry.kappa_c),
            kappa_m: q16s(c.matter.kappa_m),
            kappa_s: q16s(c.signals.kappa_s),
            gamma_q16: q16s(c.bonds.gamma),
            diss_q16: to_q16(e.dissipation) as u64,
            relax_q16: q16s(c.temperature.relax),
            t_amb: (c.temperature.ambient * 100.0) as u32,
            sig_decay_q16: to_q16(c.signals.decay) as u64,
            sig_emit: c.signals.emit,
            w_occ_q16: to_q16(c.shielding.w_occupied) as u64,
            sun_on: e.enabled && e.sun.enabled && e.mode == EnergyMode::Normal,
            sun_i0_q16: to_q16(e.sun.i0) as u64,
            sun_grad_q16: q16s(e.sun.gradient_y),
            sun_atten_q16: to_q16(e.sun.attenuation) as u64,
            sun_heat_q16: to_q16(e.sun.heat) as u64,
            day_period: e.sun.day_period,
            season_period: e.sun.season_period,
            season_amp: e.sun.season_amplitude,
            vents_on: e.vents.enabled,
            chem_only: e.mode == EnergyMode::ChemicalOnly,
            reactions,
            cat,
            mass,
            n_cat: c.chemistry.n_cat,
            eta_min_q16: to_q16(c.chirality.eta_min) as u64,
            ci_q16: to_q16(c.chirality.cross_inhibition) as u64,
            fold_w: c.chirality.fold_window,
            sigma_min_q16: to_q16(c.chirality.sigma_min) as u64,
            rxn_heat_q16: to_q16(c.chemistry.heat_per_energy) as u64,
            sink_q16: to_q16(c.chemistry.global_sink) as u64,
            rac_q32: to_q32(c.chirality.racemization.rate),
            rac_bias_q16: to_q16(1.0 - c.chirality.bias.racemization) as u64,
            p_sub: if copy_on { cp(m.copy.sub) } else { 0 },
            p_ins: if copy_on { cp(m.copy.sub).saturating_add(cp(m.copy.ins)) } else { 0 },
            p_del: if copy_on { cp(m.copy.sub).saturating_add(cp(m.copy.ins)).saturating_add(cp(m.copy.del)) } else { 0 },
            mu_bg_q32: if m.enabled { to_q32(m.background) } else { 0 },
            mu_uv_q32: if m.enabled { to_q32(m.radiation) } else { 0 },
            mu_flip_q32: if m.enabled { to_q32(m.hand_flip) } else { 0 },
            decay_q32: if c.decay.enabled { to_q32(c.decay.rate) } else { 0 },
            decay_light_q16: to_q16(c.decay.light_factor) as u64,
            bond_break_q32: if c.bonds.enabled { to_q32(c.bonds.break_rate) } else { 0 },
            vote_q16: to_q16(c.bonds.vote_threshold) as u64,
            max_body: c.bonds.max_body_move,
            arr,
            arr_mut,
            arr_rac,
            arr_decay,
            arr_bond,
            pois: poisson_exp_table(),
            nbr: (0..geom.n)
                .flat_map(|c| {
                    let g = &geom;
                    (0..6u32).map(move |d| g.neighbor(c, d).map(|t| t as u32).unwrap_or(NONE))
                })
                .collect(),
            x4_species: if c.chemistry.enabled {
                c.chemistry.species.iter().position(|s| s == "X4").or(Some(c.chemistry.species.len().saturating_sub(1)))
            } else {
                None
            },
            vent_cfg: c.energy.vents.clone(),
            map_kind: match c.mutation.map.as_str() {
                "vent_radial" => 1,
                "image" => 2,
                _ => 0,
            },
            map_a: c.mutation.map_a,
            map_r0: c.mutation.map_r0,
            hash_every: c.telemetry.hash_every,
            geom,
        }
    }

    #[inline]
    pub fn f_chem(&self, k: usize, h: usize) -> usize {
        2 + 2 * k + h
    }
    #[inline]
    pub fn f_pool(&self, cl: usize) -> usize {
        2 + 2 * self.k + cl
    }
    #[inline]
    pub fn f_sig(&self, j: usize, h: usize) -> usize {
        2 + 2 * self.k + 3 + 2 * j + h
    }
}

pub const F_F: usize = 0;
pub const F_T: usize = 1;
