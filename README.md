# Primordia

An implementation of `primorida_spec.md`: a deterministic artificial-life simulator in Rust + wgpu.
Random byte programs (PISA-1) on a 0D–3D lattice can become self-replicators under mirror-symmetric
physics with energy, temperature, chemistry, conserved matter, bonding and chirality.

```
cargo build --release
./target/release/primordia presets
./target/release/primordia run primordia-2d-sun                # live viewer
./target/release/primordia run paper-0d --headless --epochs 16000
./target/release/primordia twin primordia-3d --set world.size=[24,24,24] --epochs 10000
./target/release/primordia assay "12 13 15 c2"
./target/release/primordia bench
./target/release/primordia replay runs/<run>
./target/release/primordia branch runs/<run>/snapshots/e0000010000.psnap diff.toml
./target/release/primordia batch sweep.toml
```

`--set key=value` overrides any config key (TOML value syntax), `--cpu` uses the CPU golden model,
`--exec-only` keeps physics on the CPU and only interprets on the GPU, `--events file.jsonl` applies a logged
intervention list, `--autorun` starts the viewer running, and `--web 0.0.0.0:8765` serves the read-only
three.js browser viewer (works with `--headless` too).

### GPU-resident, pipelined (default)

The whole state stays on the GPU: `phys.wgsl` ports every phase except interventions and vent bookkeeping,
and `exec.wgsl` ports the interpreter. Each epoch is one command buffer, and up to 6 epochs are in flight.
Tallies and births come back through a ring of staging buffers mapped asynchronously. Whether the bodies
phase runs is decided on the GPU (indirect dispatch gated by the vote count; union-find labelling rounds
gated by a "changed" flag), so no epoch waits on a read-back. Telemetry, run-directory hashes and snapshots,
the viewers and the web feed run on an observer thread from asynchronous state snapshots.
`gpu/tests/parity.rs` checks it bit for bit against the CPU golden model, including the
pipelined/asynchronous path. `PRIMORDIA_PROFILE=1 primordia bench` breaks GPU time down by phase.

| World (RTX 5070) | interpreter on GPU only | resident | resident + pipelined + kernel work |
| --- | --- | --- | --- |
| 2D 1023² full physics | 1.8 ep/s | 18 | **48** |
| 2D 501² full physics | 8.7 | 41 | **127** |
| 3D 99³ full physics | 1.8 | 9.8 | **34** |
| paper-0d (8192 steps) | 9.7 | 10.6 | **32** |

GPU utilisation during a full run with viewer, telemetry and web feed: 95–98%.

### Viewers

* Native (egui). The default "Colonies" view finds colonies: spatially connected groups of related cells
  whose programs share their opening bytes. Each gets a stable identity, colour and name across frames
  (overlap tracking). The atlas tab lists them (size, growth, copiers, bonding, handedness, typical program);
  click one to fly there, use "follow" to track it, or `[` / `]` to cycle. The map shows name labels, vents
  as beacons, a minimap when zoomed in, bonds and discs at high zoom, and plain-language hover cards.
  Tabs also hold the Chronicle and the Inspector; dashboard cards and trends sit above and below; the
  interventions are collapsed under Tools.
* Browser (three.js, `--web 0.0.0.0:8765`): a read-only observer. It shows a landscape coloured by colony,
  a colony atlas with fly-to, floating name tags, a Tour mode, Overview / Top-down cameras, a spin toggle
  (off by default, remembered), hover cards, a chronicle and trends. The server answers only GET requests.
* `primordia run x --resume <run>/snapshots/e….psnap` continues a run from a snapshot. Closing the viewer
  saves a final snapshot.

## Layout

| Crate | Contents |
| --- | --- |
| `primordia-core` | config + presets + validation, PISA-1 tables, mirror map, rotation group, Philox4x32-10, fixed-point tables, the CPU golden model of every phase, interventions, snapshots, WGSL constant generator |
| `primordia-gpu` | `exec.wgsl` (line-by-line port of `core/src/exec.rs`), wgpu backend for phase 6 |
| `primordia-telemetry` | metrics, high-order entropy (brotli q2), replicator assay, MinHash/LSH species census, lineage, chirality/EE/domains, t_b, Parquet writer |
| `primordia-view` | egui viewer: maps in 12 colour modes, 3D slice panels, 1D space-time, plots, inspector with CPU debugger, all interventions |
| `primordia-cli` | `run`, `batch`, `replay`, `branch`, `twin`, `assay`, `bench` |

A run directory contains `manifest.json`, `config.toml` (fully resolved), `events.jsonl`, `hashes.csv`,
`metrics/chirality/species/lineage/opcodes/ecology/assay.parquet`, `summary.json` and zstd `snapshots/*.psnap`.

## Tests

| Spec | Where | Status |
| --- | --- | --- |
| T1 opcode tests | `core/tests/golden.rs` (`t1_*`) | pass |
| T2 mirror map + every `_L` op on S == `_D` op on M(S) (1D/2D/3D) | `t2_*` | pass |
| T3 minimal replicator in paper-0d sandbox | `t3_*` | pass |
| T4 GPU vs CPU, identical hash every epoch (63² 2D, 21³ 3D, 1D, paper-0d) | `gpu/tests/parity.rs` | pass |
| T5 cross-vendor | only one adapter (NVIDIA RTX 5070) available | not run |
| T6 ledgers exactly zero | `t6_*` (all mechanisms, 1D/2D/3D) + 100k-epoch run | pass |
| T7 twin | `t7_*` + `primordia twin` 10k epochs 2D and 3D | pass |
| T8 resume | `t8_*` | pass |
| T9 replay with interventions | `t9_*` + `primordia replay` | pass |
| T10 paper replication | `primordia run paper-0d` / `batch` | see results |
| T11 seeded takeover | `gpu/tests/takeover.rs` (ignored; release) | see results |
| T12 fuzz | `t12_*` (random configs incl. 0D, walls, pair_concat) | pass |
| T13 performance | `primordia bench` | see results |

`cargo test --release` runs T1–T4, T6–T9 and T12.

## Design decisions and deviations from the spec

* **Physics split.** Phase 6 (the interpreter) runs on the GPU; phases 2–5 and 7–9 run on the CPU
  (rayon, deterministic). The state is uploaded and downloaded every epoch. The CPU golden model
  implements every phase, and the parity test shows GPU == CPU bit for bit.
* **Single source of truth.** Layout constants, parameter indices and rotation tables are generated by
  `primordia_core::wgsl::constants()` and prepended to the shader when the pipeline is created, rather
  than by a `build.rs`.
* **Mirror symmetry of numeric code.** `RD`/`WR` move byte *values* through the stack, and integer
  arithmetic does not commute with `b ^ 1`, so a byte-level physics cannot be exactly mirror-symmetric.
  As the spec does for RNG draws, the twin world reads byte values in the reflected frame:
  `RD` pushes `mirror(b)`, `WR` writes `mirror(v)`, and bit-flip mutations apply `M∘flip∘M`. The twin test
  therefore proves that the *implementation* (geometry, orientations, scheduler, RNG keys, transport
  dithers, chemistry hands, bodies) has no hidden asymmetry. The encoding asymmetry of numeric
  code is a property of PISA-1 itself: L-vs-D results have to be judged statistically across seeds
  (`batch` reports the binomial test), and C0 (`stereo = off`) is the baseline.
* **PITCH_L / PITCH_D** are the same rotation. A rotation about the lateral axis involves only facing and
  up, so it is achiral; making the two different would break the mirror requirement (T2).
* **Conflicts.** Anything that reads beyond the executor's Chebyshev-1 neighbourhood (shielding counts
  the target's other neighbours) reads a flag snapshot taken at the start of the sub-step, so parallel
  executors never race. An individual that moves ends its slice, and each individual executes at most
  once per epoch (`FL_EXEC`).
* **Transport.** The flux is computed once per face from both endpoints with a 16-bit Philox dither; the
  mirror twin complements the dither on x faces, which makes the flux exactly antisymmetric under
  reflection. Outflows are limited proportionally so that no cell goes negative (still exactly
  conservative). Up to 8 fields share one Philox call per face; L/D pairs are never split across batches,
  so hand-swapped dither slots line up in the twin.
* **Mutation sampling** uses a Poisson inverse CDF (series for small λ, a 64k-entry `exp(-λ)` table
  otherwise) instead of a binomial table.
* **Halting.** A halted cell wakes once its energy reaches `energy.wake_energy` (default 256); waking at
  `E > 0` would make a cell restart its slice from IP 0 forever with too little energy to finish anything.
* **paper-2d** uses the checkerboard scheduler with face neighbours, not the paper's radius-2 random-claim
  scheduler. paper-0d/2d use `step_budget = 8192`, as cubff does.
* **Defaults that the spec leaves open:** jumps, `PUSH`, fillers and `JREL` cost 1; `DIGEST` costs 2 per
  byte; `ENRG`/`FREE` scale by 16; `TEMP` = clamp(K − 200, 0, 255); sunlight heats by 0.02 (0.01 K units) per
  unit of light; κ defaults are 0.08 so that κ × faces ≤ ½ also holds in 3D.
* **Organism candidates** are approximated as bonded bodies of size ≥ 2 whose members descend from one
  parent lineage. The census clusters at most 20,000 sampled tapes.
* The viewer renders on the CPU into textures (not straight from GPU buffers). "Step sub-step" runs one
  epoch on the CPU golden model and lets you scrub through the 3^D sub-step frames. Branching from the
  viewer writes a snapshot plus a diff template for `primordia branch`.
