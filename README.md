# Primordia

A deterministic artificial-life simulator in Rust + wgpu, implementing `primorida_spec.md`. Random byte
programs (PISA-1) on a 0D–3D lattice can become self-replicators under mirror-symmetric physics with
energy, temperature, chemistry, conserved matter, bonding and chirality.

It builds on Blaise Agüera y Arcas et al., *Computational Life: How Well-formed, Self-replicating
Programs Emerge from Simple Interaction* ([arXiv:2406.19108](https://arxiv.org/abs/2406.19108); reference
code: [cubff](https://github.com/paradigms-of-intelligence/cubff)), which showed self-replicators
emerging from random BFF, Forth and Z80 program soups with no explicit fitness function. The `paper-0d`
and `paper-2d` presets reduce Primordia to that setting for validation (test T10).

Design decisions, deviations from the spec, the rules added beyond it and the GPU pipeline are described
in [DESIGN.md](DESIGN.md). Licensed under the [MIT License](LICENSE).

## Quick start

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

* `--set key=value` overrides any config key (TOML value syntax).
* `--events file.jsonl` applies a logged intervention list, such as the seed events in `runs/seed-*.jsonl`.
* `--resume <run>/snapshots/e….psnap` continues a run, with its plot history.
* `--autorun` starts the viewer running; `--headless` runs without it.
* `--web 0.0.0.0:8765` serves the read-only browser viewer (also with `--headless`).
* `--cpu` uses the CPU golden model; `--exec-only` keeps physics on the CPU.

### The live world

The current long-running experiment is a 501² world with day and night, seeded with a minimal replicator
and its mirror image:

```
./target/release/primordia run runs/live-m4.toml --events runs/seed-501.jsonl --autorun --web 0.0.0.0:8765
```

It adds weak moonlight, a 3-day lifespan, longer sleeps, and protected sleepers that absorb no light to
the spec's rules
(see [DESIGN.md](DESIGN.md#rules-added-beyond-the-spec)), to find out whether cells and bodies learn to
survive the night.

## Viewers

* **Native (egui).** Colonies (connected groups of related cells) get stable names and colours. The atlas
  lists them with fly-to and follow; the map has labels, vents, a minimap and hover cards. Tabs hold the
  Chronicle and the Inspector, the Trends panel can plot any logged metric, and interventions sit under
  Tools. Closing the viewer saves a final snapshot.
* **Browser (three.js, `--web`).** Read-only: the server answers only GET requests. A 2D map (default) or
  3D landscape coloured by colony, a colony atlas with fly-to, a Tour mode, a chronicle and headline cards
  (population, night survival, sleep, longest-lived cluster). **📈 Charts** (or `C`, or `…/#charts`)
  opens every logged metric as charts, grouped into population, sleep, bodies and clusters, behaviour,
  code, energy and run health. It also has a time-window filter, a table view (`…/#table`), and the
  longest-lived clusters with fly-to.

## Outputs

A run directory holds `manifest.json`, `config.toml` (fully resolved), `events.jsonl`, `hashes.csv`,
`summary.json`, `history.json`, zstd `snapshots/*.psnap`, and Parquet tables:

| Table | Contents |
| --- | --- |
| `metrics` | population, energy, temperature, ledgers, bodies, entropy, species (every 10 epochs) |
| `life` | sunlight; shares asleep, dormant, carrying SLEEP, bonded; cell age |
| `days` | per day: sunset and sunrise population, night survival, births, deaths, state at dawn |
| `clusters` | bodies and kin clusters of 50+ cells: counts, links, nights survived |
| `chirality`, `species`, `lineage`, `opcodes`, `ecology`, `assay` | handedness and domains, species census, births, opcode histogram, spatial blocks, replicator assay |

Run outputs are not committed. The inputs that reproduce them are: `runs/*.toml` (configs) and
`runs/*.jsonl` (seed and intervention lists).

## Layout

| Crate | Contents |
| --- | --- |
| `primordia-core` | config, presets and validation; PISA-1, mirror map, rotation group, Philox4x32-10, fixed point; the CPU golden model of every phase; interventions, snapshots; WGSL constant generator |
| `primordia-gpu` | `exec.wgsl` (interpreter) and `phys.wgsl` (all other phases), GPU-resident pipelined backend |
| `primordia-telemetry` | metrics, entropy, replicator assay, species census, chirality, day/night bookkeeping, cluster persistence tracker, Parquet writer |
| `primordia-view` | egui viewer, observer thread, read-only web server and three.js page |
| `primordia-cli` | `run`, `batch`, `replay`, `branch`, `twin`, `assay`, `bench` |

## Tests

| Spec | Where | Status |
| --- | --- | --- |
| T1 opcode tests | `core/tests/golden.rs` (`t1_*`) | pass |
| T2 mirror map + every `_L` op on S == `_D` op on M(S) (1D/2D/3D) | `t2_*` | pass |
| T3 minimal replicator in paper-0d sandbox | `t3_*` | pass |
| T4 GPU vs CPU, identical hash every epoch | `gpu/tests/parity.rs` | pass |
| T5 cross-vendor | only one adapter (NVIDIA RTX 5070) available | not run |
| T6 ledgers exactly zero | `t6_*` (all mechanisms, 1D/2D/3D) + 100k-epoch run | pass |
| T7 twin | `t7_*` + `primordia twin` 10k epochs 2D and 3D | pass |
| T8 resume | `t8_*` | pass |
| T9 replay with interventions | `t9_*` + `primordia replay` | pass |
| T10 paper replication | `primordia run paper-0d` / `batch` | see results |
| T11 seeded takeover | `gpu/tests/takeover.rs` (ignored; release) | see results |
| T12 fuzz | `t12_*` (random configs incl. 0D, walls, pair_concat) | pass |
| T13 performance | `primordia bench` | see results |

`cargo test --release` runs T1–T4, T6–T9 and T12, plus tests for the added rules.
