//! Config schema, presets (inherit + override by deep merge) and validation.

use crate::geom::Boundary;
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub preset: Option<String>,
    pub seed: u64,
    pub epochs: u64,
    pub world: WorldCfg,
    pub individual: IndividualCfg,
    pub costs: BTreeMap<String, u32>,
    pub energy: EnergyCfg,
    pub temperature: TempCfg,
    pub chemistry: ChemCfg,
    pub matter: MatterCfg,
    pub signals: SignalCfg,
    pub chirality: ChiralCfg,
    pub shielding: ShieldCfg,
    pub bonds: BondCfg,
    pub mutation: MutationCfg,
    pub decay: DecayCfg,
    pub telemetry: TelemetryCfg,
    pub gpu: GpuCfg,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WorldCfg {
    pub dims: u32,
    pub size: Vec<u32>,
    pub boundary: Vec<Boundary>,
    /// Fraction of cells that start with a random individual.
    pub init_occupancy: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairOrder {
    Random,
    SelfFirst,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Interaction {
    Hybrid,
    PairConcat,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IndividualCfg {
    pub l_init: u32,
    pub l_max: u32,
    pub step_budget: u32,
    pub pair_order: PairOrder,
    pub random_initial_facing: bool,
    pub interaction: Interaction,
    pub dead_after: u32,
    /// A cell that stays dormant (out of energy) for this many epochs dies;
    /// 0 = never (dormant cells only die once their tape has decayed to zeros).
    pub starve_after: u32,
    /// A cell dies of old age this many epochs after its birth; 0 = never.
    pub max_age: u32,
    /// `SLEEP n` sleeps for min(n, 255) x this many epochs (1 = as in the spec).
    pub sleep_scale: u32,
    /// A sleeping cell's tape cannot be written or digested by neighbours
    /// (a spore); reading it is still allowed.
    pub sleep_shield: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Upkeep {
    pub base: f64,
    pub per_byte: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnergyMode {
    Normal,
    ChemicalOnly,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SunCfg {
    pub enabled: bool,
    pub i0: f64,
    pub gradient_y: f64,
    /// 3D only: fraction of light absorbed by each occupied cell in a column.
    pub attenuation: f64,
    /// 0 disables the day/night cycle.
    pub day_period: u64,
    pub season_period: u64,
    pub season_amplitude: f64,
    /// Heat (0.01 K) per unit of light per epoch.
    pub heat: f64,
    /// Moonlight: the day/night cycle never falls below this fraction of
    /// full sun (0 = dark nights).
    pub moon: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VentCfg {
    pub enabled: bool,
    pub count: u32,
    /// Free energy injected per epoch.
    pub rate: u32,
    /// Heat (0.01 K) injected per epoch.
    pub heat: u32,
    /// X4 molecules injected per hand per epoch.
    pub x4_rate: u32,
    pub lifetime: u64,
    /// Probability per epoch that a new vent spawns.
    pub spawn_rate: f64,
    /// Probability per epoch that each vent drifts one cell.
    pub drift: f64,
    /// Only vents on the floor (z = 0) in 3D.
    pub floor_only: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EnergyCfg {
    pub enabled: bool,
    pub mode: EnergyMode,
    pub upkeep: Upkeep,
    pub absorb_alpha: f64,
    pub share_beta: f64,
    /// Heat (0.01 K) per energy unit spent.
    pub heat_per_energy: f64,
    pub kappa_f: f64,
    pub dissipation: f64,
    pub init_f: u32,
    pub init_e: u32,
    /// Halted cells wake only once they hold this much energy.
    pub wake_energy: u32,
    pub sense_energy_scale: u32,
    pub sense_free_scale: u32,
    pub sun: SunCfg,
    pub vents: VentCfg,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TempCfg {
    pub ambient: f64,
    pub t_ref: f64,
    pub kappa_t: f64,
    pub relax: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReactionKind {
    /// Rate k * A(T) * product of reactant counts.
    Mass,
    /// Rate k * (light / i0) * reactant count.
    Photolysis,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReactionCfg {
    pub r: Vec<String>,
    pub p: Vec<String>,
    /// >0 releases energy, <0 absorbs it from F.
    pub energy: i32,
    pub catalyzable: bool,
    pub kind: ReactionKind,
    /// Uncatalyzed rate constant (per epoch).
    pub rate: f64,
    /// Arrhenius activation (eV).
    pub activation: f64,
}

impl Default for ReactionCfg {
    fn default() -> Self {
        ReactionCfg {
            r: vec![],
            p: vec![],
            energy: 0,
            catalyzable: false,
            kind: ReactionKind::Mass,
            rate: 0.0,
            activation: 0.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChemCfg {
    pub enabled: bool,
    pub species: Vec<String>,
    pub mass: Vec<u32>,
    /// Initial count per hand per cell, one per species.
    pub init: Vec<u32>,
    pub reactions: Vec<ReactionCfg>,
    pub kappa_c: f64,
    /// Catalysis events per CAT op at eta = 1.
    pub n_cat: u32,
    /// Optional global sink: fraction of each chemical removed per epoch.
    pub global_sink: f64,
    /// Heat (0.01 K) per energy unit released by uncatalyzed reactions.
    pub heat_per_energy: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MatterCfg {
    pub enabled: bool,
    pub kappa_m: f64,
    /// Initial free monomers per cell: L, D, A.
    pub pool_l: u32,
    pub pool_d: u32,
    pub pool_a: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SignalCfg {
    pub enabled: bool,
    pub count: u32,
    pub kappa_s: f64,
    pub decay: f64,
    pub emit: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RacemCfg {
    pub rate: f64,
    pub activation: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct BiasCfg {
    pub racemization: f64,
    pub reaction_energy: f64,
    pub initial_ee: f64,
}

impl BiasCfg {
    pub fn is_zero(&self) -> bool {
        self.racemization == 0.0 && self.reaction_energy == 0.0 && self.initial_ee == 0.0
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChiralCfg {
    pub stereo: bool,
    pub stereo_copy: bool,
    pub fold_window: u32,
    pub eta_min: f64,
    pub sigma_min: f64,
    pub cross_inhibition: f64,
    pub racemization: RacemCfg,
    /// "random", "racemic" or "seeded(x)".
    pub init_ee: String,
    pub bias: BiasCfg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Granularity {
    Slice,
    Byte,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ShieldCfg {
    pub enabled: bool,
    pub s_bond: f64,
    pub s_occupied: f64,
    pub w_occupied: f64,
    pub granularity: Granularity,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BondCfg {
    pub enabled: bool,
    pub gamma: f64,
    pub break_rate: f64,
    pub break_activation: f64,
    pub max_body_move: u32,
    pub vote_threshold: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CopyErr {
    pub sub: f64,
    pub ins: f64,
    pub del: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MutationCfg {
    pub enabled: bool,
    pub background: f64,
    pub activation: f64,
    pub radiation: f64,
    pub copy: CopyErr,
    pub hand_flip: f64,
    /// "uniform", "vent_radial" or "image"
    pub map: String,
    pub map_a: f64,
    pub map_r0: f64,
    pub map_image: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DecayCfg {
    pub enabled: bool,
    pub rate: f64,
    pub activation: f64,
    pub light_factor: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TelemetryCfg {
    pub hash_every: u64,
    pub snapshot_every: u64,
    pub tracers: bool,
    pub ledger_every: u64,
    pub metrics_every: u64,
    pub opcode_every: u64,
    pub chirality_every: u64,
    pub entropy_every: u64,
    pub entropy_samples: u32,
    pub assay_every: u64,
    pub census_every: u64,
    pub ecology_every: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GpuCfg {
    /// Run the execution phase on the GPU (physics stays on the CPU).
    pub exec: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            preset: None,
            seed: 1,
            epochs: 10_000,
            world: WorldCfg::default(),
            individual: IndividualCfg::default(),
            costs: BTreeMap::new(),
            energy: EnergyCfg::default(),
            temperature: TempCfg::default(),
            chemistry: ChemCfg::default(),
            matter: MatterCfg::default(),
            signals: SignalCfg::default(),
            chirality: ChiralCfg::default(),
            shielding: ShieldCfg::default(),
            bonds: BondCfg::default(),
            mutation: MutationCfg::default(),
            decay: DecayCfg::default(),
            telemetry: TelemetryCfg::default(),
            gpu: GpuCfg { exec: true },
        }
    }
}
impl Default for GpuCfg {
    fn default() -> Self {
        GpuCfg { exec: true }
    }
}
impl Default for WorldCfg {
    fn default() -> Self {
        WorldCfg { dims: 2, size: vec![1023, 1023], boundary: vec![], init_occupancy: 0.5 }
    }
}
impl Default for IndividualCfg {
    fn default() -> Self {
        IndividualCfg {
            l_init: 64,
            l_max: 128,
            step_budget: 512,
            pair_order: PairOrder::Random,
            random_initial_facing: true,
            interaction: Interaction::Hybrid,
            dead_after: 64,
            starve_after: 0,
            max_age: 0,
            sleep_scale: 1,
            sleep_shield: false,
        }
    }
}
impl Default for Upkeep {
    fn default() -> Self {
        Upkeep { base: 1.0, per_byte: 0.02 }
    }
}
impl Default for SunCfg {
    fn default() -> Self {
        SunCfg {
            enabled: true,
            i0: 40.0,
            gradient_y: 0.5,
            attenuation: 0.3,
            day_period: 2000,
            season_period: 0,
            season_amplitude: 0.0,
            heat: 0.02,
            moon: 0.0,
        }
    }
}
impl Default for VentCfg {
    fn default() -> Self {
        VentCfg {
            enabled: false,
            count: 8,
            rate: 400,
            heat: 400,
            x4_rate: 20,
            lifetime: 50_000,
            spawn_rate: 1.6e-4,
            drift: 0.001,
            floor_only: true,
        }
    }
}
impl Default for EnergyCfg {
    fn default() -> Self {
        EnergyCfg {
            enabled: true,
            mode: EnergyMode::Normal,
            upkeep: Upkeep::default(),
            absorb_alpha: 0.05,
            share_beta: 0.1,
            heat_per_energy: 0.05,
            kappa_f: 0.08,
            dissipation: 0.002,
            init_f: 500,
            init_e: 600,
            wake_energy: 256,
            sense_energy_scale: 16,
            sense_free_scale: 16,
            sun: SunCfg::default(),
            vents: VentCfg::default(),
        }
    }
}
impl Default for TempCfg {
    fn default() -> Self {
        TempCfg { ambient: 300.0, t_ref: 300.0, kappa_t: 0.08, relax: 0.01 }
    }
}
impl Default for ChemCfg {
    fn default() -> Self {
        let r = |r: &[&str], p: &[&str]| (r.iter().map(|s| s.to_string()).collect(), p.iter().map(|s| s.to_string()).collect());
        let (r0, p0) = r(&["X1", "X2"], &["X3"]);
        let (r1, p1) = r(&["X3"], &["X1", "X2"]);
        let (r2, p2) = r(&["X4"], &["X3"]);
        let (r3, p3) = r(&["X3"], &["X1", "X2"]);
        ChemCfg {
            enabled: true,
            species: vec!["X1".into(), "X2".into(), "X3".into(), "X4".into()],
            mass: vec![1, 1, 2, 2],
            init: vec![40, 40, 20, 0],
            reactions: vec![
                ReactionCfg { r: r0, p: p0, energy: 12, catalyzable: true, kind: ReactionKind::Mass, rate: 1e-6, activation: 0.5 },
                ReactionCfg { r: r1, p: p1, energy: 0, catalyzable: false, kind: ReactionKind::Photolysis, rate: 2e-3, activation: 0.0 },
                ReactionCfg { r: r2, p: p2, energy: 20, catalyzable: true, kind: ReactionKind::Mass, rate: 1e-5, activation: 0.5 },
                ReactionCfg { r: r3, p: p3, energy: -12, catalyzable: false, kind: ReactionKind::Mass, rate: 1e-4, activation: 0.6 },
            ],
            kappa_c: 0.05,
            n_cat: 4,
            global_sink: 0.0,
            heat_per_energy: 0.05,
        }
    }
}
impl Default for MatterCfg {
    fn default() -> Self {
        MatterCfg { enabled: true, kappa_m: 0.02, pool_l: 8, pool_d: 8, pool_a: 24 }
    }
}
impl Default for SignalCfg {
    fn default() -> Self {
        SignalCfg { enabled: true, count: 2, kappa_s: 0.08, decay: 0.1, emit: 64 }
    }
}
impl Default for RacemCfg {
    fn default() -> Self {
        RacemCfg { rate: 1e-4, activation: 0.8 }
    }
}
impl Default for ChiralCfg {
    fn default() -> Self {
        ChiralCfg {
            stereo: true,
            stereo_copy: true,
            fold_window: 8,
            eta_min: 0.1,
            sigma_min: 0.0,
            cross_inhibition: 0.5,
            racemization: RacemCfg::default(),
            init_ee: "random".into(),
            bias: BiasCfg::default(),
        }
    }
}
impl Default for ShieldCfg {
    fn default() -> Self {
        ShieldCfg { enabled: true, s_bond: 0.25, s_occupied: 0.10, w_occupied: 0.5, granularity: Granularity::Slice }
    }
}
impl Default for BondCfg {
    fn default() -> Self {
        BondCfg { enabled: true, gamma: 4.0, break_rate: 1e-4, break_activation: 0.5, max_body_move: 4096, vote_threshold: 0.5 }
    }
}
impl Default for CopyErr {
    fn default() -> Self {
        CopyErr { sub: 1e-3, ins: 2e-4, del: 2e-4 }
    }
}
impl Default for MutationCfg {
    fn default() -> Self {
        MutationCfg {
            enabled: true,
            background: 2.4e-4,
            activation: 0.3,
            radiation: 1e-4,
            copy: CopyErr::default(),
            hand_flip: 1e-5,
            map: "uniform".into(),
            map_a: 4.0,
            map_r0: 8.0,
            map_image: None,
        }
    }
}
impl Default for DecayCfg {
    fn default() -> Self {
        DecayCfg { enabled: true, rate: 2e-5, activation: 0.4, light_factor: 1.0 }
    }
}
impl Default for TelemetryCfg {
    fn default() -> Self {
        TelemetryCfg {
            hash_every: 100,
            snapshot_every: 10_000,
            tracers: false,
            ledger_every: 1,
            metrics_every: 10,
            opcode_every: 10,
            chirality_every: 10,
            entropy_every: 100,
            entropy_samples: 65_536,
            assay_every: 1000,
            census_every: 1000,
            ecology_every: 100,
        }
    }
}

// ---------------------------------------------------------------- presets

pub const PRESETS: &[(&str, &str)] = &[
    (
        "paper-0d",
        r#"
epochs = 16000
[world]
dims = 0
size = [131072]
init_occupancy = 1.0
[individual]
l_init = 64
l_max = 64
step_budget = 8192
interaction = "pair_concat"
random_initial_facing = false
dead_after = 1000000000
[energy]
enabled = false
[chemistry]
enabled = false
[matter]
enabled = false
[signals]
enabled = false
[shielding]
enabled = false
[bonds]
enabled = false
[decay]
enabled = false
[chirality]
stereo = false
stereo_copy = false
[chirality.racemization]
rate = 0.0
[mutation]
background = 2.4e-4
activation = 0.0
radiation = 0.0
hand_flip = 0.0
[mutation.copy]
sub = 0.0
ins = 0.0
del = 0.0
"#,
    ),
    (
        "paper-2d",
        r#"
preset = "paper-0d"
[world]
dims = 2
size = [240, 135]
[individual]
random_initial_facing = true
"#,
    ),
    (
        "primordia-2d-sun",
        r#"
[world]
dims = 2
size = [1023, 1023]
[energy.sun]
enabled = true
[energy.vents]
enabled = false
"#,
    ),
    (
        "primordia-2d-vents",
        r#"
[world]
dims = 2
size = [1023, 1023]
[energy]
mode = "chemical_only"
[energy.sun]
enabled = false
[energy.vents]
enabled = true
count = 64
floor_only = false
x4_rate = 400
[mutation]
map = "vent_radial"
"#,
    ),
    (
        "primordia-3d",
        r#"
[world]
dims = 3
size = [99, 99, 99]
[energy.sun]
enabled = true
i0 = 60
attenuation = 0.2
[energy.vents]
enabled = true
count = 16
floor_only = true
[mutation]
map = "vent_radial"
"#,
    ),
    ("chiral-c0", "preset = \"primordia-2d-sun\"\n[chirality]\nstereo = false\nstereo_copy = false\n"),
    ("chiral-c1", "preset = \"primordia-2d-sun\"\n[chirality]\nstereo = true\ncross_inhibition = 0.5\n"),
    ("chiral-c2", "preset = \"primordia-2d-sun\"\n[matter]\npool_l = 32\npool_d = 32\n[chirality.racemization]\nrate = 1e-3\n"),
    ("chiral-c3", "preset = \"primordia-2d-sun\"\n"),
    ("chiral-c4", "preset = \"primordia-2d-vents\"\n[energy.vents]\nheat = 2000\n"),
    ("chiral-c5", "preset = \"primordia-2d-sun\"\n[chirality.bias]\nracemization = 1e-4\n"),
];

pub fn preset_toml(name: &str) -> Option<&'static str> {
    PRESETS.iter().find(|(n, _)| *n == name).map(|(_, t)| *t)
}

fn merge(base: &mut toml::Value, over: &toml::Value) {
    match (base, over) {
        (toml::Value::Table(b), toml::Value::Table(o)) => {
            for (k, v) in o {
                match b.get_mut(k) {
                    Some(bv) if bv.is_table() && v.is_table() => merge(bv, v),
                    _ => {
                        b.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (b, o) => *b = o.clone(),
    }
}

/// Resolve a TOML document: expand its preset chain on top of the defaults.
pub fn resolve_value(doc: &toml::Value) -> Result<toml::Value> {
    let mut chain = vec![doc.clone()];
    let mut cur = doc.clone();
    let mut guard = 0;
    while let Some(p) = cur.get("preset").and_then(|v| v.as_str()).map(|s| s.to_string()) {
        let t = preset_toml(&p).with_context(|| format!("unknown preset {p}"))?;
        cur = toml::from_str(t)?;
        chain.push(cur.clone());
        guard += 1;
        ensure!(guard < 16, "preset cycle");
    }
    let mut base = toml::Value::try_from(Config::default())?;
    for v in chain.iter().rev() {
        merge(&mut base, v);
    }
    if let Some(p) = doc.get("preset") {
        base.as_table_mut().unwrap().insert("preset".into(), p.clone());
    }
    Ok(base)
}

impl Config {
    pub fn from_toml_str(s: &str) -> Result<Config> {
        let v: toml::Value = toml::from_str(s)?;
        let r = resolve_value(&v)?;
        let c: Config = r.try_into().context("config schema")?;
        c.validate()?;
        Ok(c)
    }
    pub fn from_file(path: &std::path::Path) -> Result<Config> {
        let s = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        Self::from_toml_str(&s)
    }
    pub fn preset(name: &str) -> Result<Config> {
        Self::from_toml_str(&format!("preset = \"{name}\"\n"))
    }
    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).unwrap()
    }
    /// Apply a dotted-key override, e.g. ("energy.absorb_alpha", 0.1).
    pub fn with_override(&self, key: &str, value: toml::Value) -> Result<Config> {
        let mut v = toml::Value::try_from(self)?;
        let mut cur = &mut v;
        let parts: Vec<&str> = key.split('.').collect();
        for (i, p) in parts.iter().enumerate() {
            let t = cur.as_table_mut().context("not a table")?;
            if i + 1 == parts.len() {
                t.insert(p.to_string(), value.clone());
                break;
            }
            cur = t.entry(p.to_string()).or_insert(toml::Value::Table(Default::default()));
        }
        let c: Config = v.try_into()?;
        c.validate()?;
        Ok(c)
    }
    pub fn n_species(&self) -> usize {
        if self.chemistry.enabled { self.chemistry.species.len() } else { 0 }
    }
    pub fn n_signals(&self) -> usize {
        if self.signals.enabled { self.signals.count as usize } else { 0 }
    }
    pub fn faces(&self) -> u32 {
        2 * self.world.dims
    }
    pub fn pair_concat(&self) -> bool {
        self.individual.interaction == Interaction::PairConcat
    }

    pub fn validate(&self) -> Result<()> {
        let w = &self.world;
        ensure!(w.dims <= 3, "dims must be 0..=3");
        let axes = if w.dims == 0 { 1 } else { w.dims as usize };
        ensure!(w.size.len() >= axes, "world.size needs {axes} entries");
        if w.dims > 0 {
            for a in 0..axes {
                ensure!(w.size[a] >= 3 && w.size[a] % 3 == 0, "side length {} must be a multiple of 3", w.size[a]);
            }
        } else {
            ensure!(w.size[0] >= 2, "0D needs at least 2 slots");
        }
        let ind = &self.individual;
        ensure!(ind.l_max >= 1 && ind.l_max <= 256 && ind.l_max % 4 == 0, "l_max must be 4..=256 and a multiple of 4");
        ensure!(ind.l_init <= ind.l_max, "l_init > l_max");
        ensure!((1..=256).contains(&ind.sleep_scale), "individual.sleep_scale must be 1..=256");
        ensure!((0.0..=1.0).contains(&self.energy.sun.moon), "energy.sun.moon must be in 0..=1");
        ensure!(ind.step_budget <= 65535, "step_budget too large");
        let faces = self.faces().max(1) as f64;
        for (name, k) in [
            ("kappa_f", self.energy.kappa_f),
            ("kappa_t", self.temperature.kappa_t),
            ("kappa_c", self.chemistry.kappa_c),
            ("kappa_m", self.matter.kappa_m),
            ("kappa_s", self.signals.kappa_s),
        ] {
            ensure!(k >= 0.0, "{name} negative");
            ensure!(k * faces <= 0.5, "{name} x faces must be <= 2^15/2^16 for stability (got {})", k * faces);
        }
        if self.chemistry.enabled {
            let ch = &self.chemistry;
            ensure!(ch.mass.len() == ch.species.len(), "chemistry.mass length");
            ensure!(ch.init.len() == ch.species.len(), "chemistry.init length");
            ensure!(ch.species.len() <= 8, "at most 8 species");
            for (i, r) in ch.reactions.iter().enumerate() {
                let m = |names: &Vec<String>| -> Result<u32> {
                    let mut s = 0;
                    for n in names {
                        let k = ch.species.iter().position(|x| x == n).with_context(|| format!("unknown species {n}"))?;
                        s += ch.mass[k];
                    }
                    Ok(s)
                };
                ensure!(!r.r.is_empty() && r.r.len() <= 2 && !r.p.is_empty() && r.p.len() <= 2, "reaction {i}: 1-2 reactants and products");
                let (mr, mp) = (m(&r.r)?, m(&r.p)?);
                ensure!(mr == mp, "reaction {i} does not balance mass ({mr} -> {mp})");
                if r.kind == ReactionKind::Photolysis {
                    ensure!(r.r.len() == 1, "photolysis has one reactant");
                }
            }
        }
        ensure!(self.signals.count <= 8, "at most 8 signal species");
        ensure!(self.chirality.fold_window >= 1 && self.chirality.fold_window <= 64, "fold_window 1..=64");
        parse_init_ee(&self.chirality.init_ee)?;
        match self.mutation.map.as_str() {
            "uniform" | "vent_radial" | "image" => {}
            m => bail!("unknown mutation map {m}"),
        }
        for (k, v) in &self.costs {
            let b = crate::isa::parse_opcode_name(k).with_context(|| format!("unknown opcode {k}"))?;
            if crate::isa::is_chiral(b) {
                let m = crate::isa::mirror(b);
                let other = self
                    .costs
                    .iter()
                    .find(|(k2, _)| crate::isa::parse_opcode_name(k2) == Some(m))
                    .map(|(_, v)| *v);
                ensure!(other == Some(*v), "cost override for {k} must be mirrored for its twin");
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InitEe {
    Random,
    Racemic,
    Seeded(f64),
}

pub fn parse_init_ee(s: &str) -> Result<InitEe> {
    let s = s.trim();
    if s == "random" {
        return Ok(InitEe::Random);
    }
    if s == "racemic" {
        return Ok(InitEe::Racemic);
    }
    if let Some(x) = s.strip_prefix("seeded(").and_then(|r| r.strip_suffix(')')) {
        let v: f64 = x.trim().parse()?;
        ensure!((-1.0..=1.0).contains(&v), "seeded ee in [-1,1]");
        return Ok(InitEe::Seeded(v));
    }
    bail!("init_ee must be random, racemic or seeded(x)")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn presets_load() {
        for (n, _) in PRESETS {
            let c = Config::preset(n).unwrap_or_else(|e| panic!("{n}: {e:#}"));
            c.validate().unwrap();
        }
        let p = Config::preset("paper-2d").unwrap();
        assert_eq!(p.world.size, vec![240, 135]);
        assert!(!p.energy.enabled);
    }
}
