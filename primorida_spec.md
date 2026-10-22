# Primordia: Open-Ended Artificial Life Simulator Spec

Oct 22, 2026 · @Hilbert

## 1. Overview

Primordia is a deterministic GPU simulator, written in Rust + wgpu, in which random byte programs on a 0D, 1D, 2D or 3D lattice can turn into self-replicators. They then live under mirror-symmetric physics with energy, temperature, chemistry, conserved matter, bonding and chirality. No fitness function is designed in, and no ancestor is seeded by default.

It extends Agüera y Arcas et al., *Computational Life* ([arXiv 2406.19108](https://arxiv.org/abs/2406.19108), code: [cubff](https://github.com/paradigms-of-intelligence/cubff)). That paper showed replicators emerging from random BFF, Forth and Z80 soups with no explicit fitness, and showed they fail to emerge in SUBLEQ, whose shortest replicator is long. Earlier relatives: Tierra and Avida (both seeded with an ancestor), Coreworld (per-instruction energy), and Fontana's Turing gas.

**Research questions the design must be able to answer**

1. Do replicators still emerge spontaneously once copying costs energy and matter? At what cost threshold do they stop?
2. Do bonded, multi-cell bodies with division of labor appear, and what drives them: shielding, energy sharing or movement?
3. Does one hand dominate (homochirality) under exactly mirror-symmetric physics? If so, is the cause chemistry (stereo-specific catalysis) or only a founder effect?
4. How do dimension (0D/2D/3D), resource geometry (sun, vents, cycles) and temperature shape ecology and diversity?
5. Does complexity keep growing (open-endedness), or does it plateau after the first takeover?

**Goals**

- Reproduce the paper's state transition with a paper-compatible preset before adding anything else.
- Run about 1M cells in real time on one desktop GPU, with bit-exact replay.
- Let every mechanism be switched off, so each result can be checked by ablation.
- Provide a live viewer for watching, inspecting, intervening and branching runs.

**Non-goals**

- No task rewards, scoring or hand-designed fitness. Only physics.
- No neural networks or learned components inside the world.
- No distributed multi-GPU support in v1.

## 2. Design principles

Every later decision in this spec follows from these eight rules. When two of them conflict, the earlier one wins.

1. **Physics, not fitness.** The world has conserved quantities, costs and local rules. Nothing scores programs. Survival means persisting and copying.
2. **Exact mirror symmetry by default.** Reflecting the world on axis 0 and swapping every L for D gives an equally valid world that evolves into the exact mirror image (§9, §11). Any parity bias is an explicit, logged config value.
3. **A short minimal replicator must exist.** SUBLEQ never produced replicators; its shortest one is 60 bytes. The base ISA admits a 4-byte replicator (§5), and every new cost is checked against this.
4. **Locality.** Each rule reads and writes only a cell and its Chebyshev radius-1 neighborhood. This is what makes exact parallel scheduling possible.
5. **Integer state, bit-exact.** All simulation state is integer or fixed-point, and all randomness is counter-based. The same seed, config and intervention log give the same run on any GPU vendor.
6. **Conservation is audited, not assumed.** Matter, chemical atoms and the energy ledger are checked every epoch. Any drift is a bug.
7. **Ablation-first.** Every mechanism has an off switch, and a `paper` preset reduces the system to a Computational Life soup for validation.
8. **Observable.** Anything that can affect dynamics can be shown in the viewer and logged.

## 3. World model

The world is a D-dimensional lattice, with D from 0 to 3, set in config. Each cell holds per-cell fields and either nothing or one individual. The engine is the same in every dimension; only the neighbor tables and the orientation group change.

**Geometry**

| Setting | Values | Default |
| --- | --- | --- |
| Dimension D | 0 (well-mixed soup), 1, 2, 3 | 2 |
| Side length per axis | multiple of 3 (needed by the scheduler, §11) | 1023 in 2D, 99 in 3D |
| Boundary per axis | torus, reflecting wall, absorbing wall | torus |
| Face neighborhood | 2D: 4 faces, 3D: 6 faces | fixed by D |
| Interaction radius | Chebyshev 1 (faces and diagonals readable by sensors; faces used for bonds, moves and partners) | 1 |

At about 1M cells, 2D is 1023² (1.05M) and 3D is 99³ (0.97M). In 0D there is no space. Cells are a list of N slots, a partner is a uniformly random other slot, and every spatial op is a no-op. This mode exists to reproduce the paper.

**Per-cell state**

| Field | Type | Meaning |
| --- | --- | --- |
| `occupied`, `halted` | flag bits | Whether an individual is present, and whether it has stopped executing |
| Tape | u8 × len, len ≤ `L_max` | The individual's program and data (§4) |
| Internal energy `E` | u32 | Energy owned by the individual; pays for instructions |
| Free energy `F` | u32 | Diffusing environmental energy (§7) |
| Temperature `T` | u16 (0.01 K steps) | Diffusing heat field (§7) |
| Chemicals `c[k][h]` | u16 counts, k < K, h ∈ {L, D} | Molecule counts per species and hand (§7, §9) |
| Free matter `m[L]`, `m[D]`, `m[A]` | u16 counts | Monomers not currently in a tape (§7) |
| Signals `s[j][h]` | u16 | Short-lived chiral signal molecules (§8) |
| Bonds | bitmask, one bit per face | Bonds to face neighbors; symmetric (§8) |
| Orientation | u8 index into the proper rotation group | 2D: 4 values, 3D: 24 values (§8) |
| Lineage id, birth epoch | u32, u32 | For telemetry (§13) |

Empty cells still carry fields, so resources flow through empty space. An individual exists only where `occupied` is set. A cell is cleared, and its remaining matter released, when its tape has been all zeros for `dead_after` epochs (default 64).

## 4. Individuals and the virtual CPU

An individual is a variable-length byte tape plus a small CPU. Only the tape is heritable. CPU state is rebuilt at the start of every execution slice, as in the paper.

**Tape**

- Length `len` is between 0 and `L_max` (default 128, configurable up to 256). It starts at `L_init` (default 64).
- The tape grows with `GROW` and `ALLOC` and shrinks with `SHRINK`, `DIVIDE` and `DIGEST` (§5). Insertion and deletion copy errors also change length (§10).
- Storage is a fixed-stride arena of `L_max` bytes per cell, so the GPU never allocates.
- Byte 0x00 is the empty site. It holds no matter and is a no-op when executed. Every other byte value is made of one monomer of class L, D or A, determined by the opcode map (§5, §9).

**Execution address space**

When a cell executes, it sees one flat address space: its own tape, then the tape of its current partner. The partner is the face neighbor it is facing. If that neighbor is empty, the partner tape has length 0. Addresses wrap modulo `len_self + len_partner`; if that sum is 0, the slice ends immediately. Turning or the `TUMBLE` op changes the facing, which swaps the upper half for another neighbor. That gives both interaction styles you picked (§6).

With `pair_order = random` (the default, as in the paper), half of all slices lay out the space as partner then self. The IP still starts at address 0, so the partner's code runs while the executing cell pays the energy. This keeps the paper's ordered A+B vs B+A reaction.

**CPU state**

| Register | Width | Reset value |
| --- | --- | --- |
| `IP` | u16 | 0 |
| `H0` (read head) | u16 | 0 |
| `H1` (write head) | u16 | start of the second tape in the address space |
| Data stack | 16 × i16 | empty; overflow or underflow halts the slice |
| Facing / orientation | rotation index | persistent per cell (stored in the cell, not reset) |
| Step counter | u16 | 0; slice ends at `step_budget` (default 512) |

A slice ends at the step budget, on `HALT`, on a stack error, or when `E` reaches 0. Running out of energy sets `halted`. The tape stays intact but executes nothing until energy arrives through sharing or absorption, and it keeps decaying meanwhile (§7).

## 5. Instruction set (PISA-1)

PISA-1 is a dense, Forth-like stack ISA. All 256 byte values do something or are harmless filler, and a 4-byte self-replicator exists. Half the code space is relative jumps and an eighth is literal pushes, the same proportions that made the paper's Forth variant transition in almost every run within about 1,000 epochs.

**Code-space layout**

| Range | Count | Group | Monomer class |
| --- | --- | --- | --- |
| 0x00 | 1 | Empty site / NOP | none (holds no matter) |
| 0x01–0x0F | 15 | Stack and control | A |
| 0x10–0x1F | 16 | Heads and tape | A |
| 0x20–0x3F | 32 | Chiral pairs: even byte = L, odd byte = D | L or D |
| 0x40–0x5F | 32 | `PUSH 0`…`PUSH 31` | A |
| 0x60–0x6F | 16 | Space, bodies, life cycle | A |
| 0x70–0x7F | 16 | Achiral filler (no-op) | A |
| 0x80–0xBF | 64 | `JMP +n`: IP += 2 + (b & 0x3F) | A |
| 0xC0–0xFF | 64 | `JMP −n`: IP −= 1 + (b & 0x3F) | A |

The mirror map is `mirror(b) = b ^ 1` for 0x20–0x3F, and `b` otherwise. It is an involution, so a single flip of bit 0 swaps the hand of a chiral op (§9, §10).

**Stack and control (0x01–0x0F)**, cost 1 each

| Byte | Op | Effect |
| --- | --- | --- |
| 01 | `DUP` | push top |
| 02 | `DROP` | pop |
| 03 | `SWAP` | swap top two |
| 04 | `OVER` | push second |
| 05 / 06 | `INC` / `DEC` | top ±= 1 |
| 07 / 08 | `ADD` / `SUB` | pop a, pop b, push b+a / b−a |
| 09 | `SHL4` | top <<= 4 (builds large literals with `PUSH`/`ADD`) |
| 0A | `NEG` | top = −top |
| 0B | `AND` | pop a, pop b, push a & b |
| 0C / 0D | `SKIPNZ` / `SKIPZ` | pop c; skip next byte if c ≠ 0 / c = 0 |
| 0E | `JREL` | pop n; IP += n |
| 0F | `JNZ` | pop n, pop c; if c ≠ 0 then IP += n |

**Heads and tape (0x10–0x1F)**

| Byte | Op | Effect | Cost |
| --- | --- | --- | --- |
| 10 | `RD` | push mem\[H0\] | 1 |
| 11 | `WR` | pop v; mem\[H1\] = v (matter exchange, §7) | 2 |
| 12 | `CPY` | mem\[H1\] = mem\[H0\] (matter exchange; copy errors, §10) | 2 |
| 13 / 14 | `H0+` / `H0−` | H0 ±= 1 | 1 |
| 15 / 16 | `H1+` / `H1−` | H1 ±= 1 | 1 |
| 17 / 18 | `SETH0` / `SETH1` | pop v; head = IP + v (position-independent) | 1 |
| 19 / 1A | `GETH0` / `GETH1` | push head − IP | 1 |
| 1B | `XH` | swap H0 and H1 | 1 |
| 1C / 1F | `LEN` / `PLEN` | push own / partner tape length | 1 |
| 1D | `GROW` | append one empty site (0x00) to own tape | 4 |
| 1E | `SHRINK` | remove own last byte, monomer to own pool | 2 |

**Chiral pairs (0x20–0x3F)**: L = even, D = odd. Each pair is exact mirror images (§9).

| Bytes L/D | Op | Effect | Cost |
| --- | --- | --- | --- |
| 20/21 | `YAW_L` / `YAW_D` | rotate facing 90° counter-clockwise / clockwise about the up axis | 1 |
| 22/23 | `PITCH_L` / `PITCH_D` | rotate about the lateral axis (3D only; no-op in 0D–2D) | 1 |
| 24/25 | `ROLL_L` / `ROLL_D` | rotate about the facing axis (3D only) | 1 |
| 26/27 | `BONDS_L` / `BONDS_D` | bond the left / right lateral face | 10 |
| 28/29 | `SENS_L` / `SENS_D` | push left / right lateral face state (0 empty, 1 occupied, 2 bonded) | 1 |
| 2A/2B | `STEP_L` / `STEP_D` | move toward left / right lateral face (as `MOVE`) | 20 |
| 2C/2D | `CAT_L` / `CAT_D` | catalyze reaction (top mod R) on L / D substrates in own cell (§9) | 2 |
| 2E/2F | `TAKE_L` / `TAKE_D` | pop n; pull up to n L / D monomers from faced cell's pool into own | 1 + n/4 |
| 30/31 | `EMIT_L` / `EMIT_D` | pop j; emit signal (j mod S) of that hand | 3 |
| 32/33 | `SIG_L` / `SIG_D` | pop j; push local level of signal (j mod S), that hand | 1 |
| 34/35 | `CHEM_L` / `CHEM_D` | pop k; push own-cell count of species (k mod K), that hand | 1 |
| 36–3F | `FILL_L` / `FILL_D` | chiral filler: no-op, but counts for folding (§9) and uses chiral matter | 1 |

**Space, bodies, life cycle (0x60–0x6F)**

| Byte | Op | Effect | Cost |
| --- | --- | --- | --- |
| 60 | `MOVE` | unbonded: swap with faced cell if it is empty or unbonded. Bonded: vote to move the whole body (§8) | 20 (40 if swapping with an occupied cell) |
| 61 / 62 | `BOND` / `UNBOND` | bond / unbond the faced neighbor | 10 / 2 |
| 63 | `SENSE` | push faced state (0 empty, 1 occupied, 2 bonded) | 1 |
| 64 / 65 | `ENRG` / `TEMP` | push own energy / local temperature, scaled to 0–255 | 1 |
| 66 | `SHARE` | pop n; give n energy to faced bonded neighbor | 1 |
| 67 | `ALLOC` | pop n; faced empty cell becomes a child with n empty sites, same orientation | 30 + n |
| 68 | `DIVIDE` | move own bytes \[H0, H1) into faced empty cell as a child; energy split in half | 30 + n |
| 69 | `DIGEST` | pop n; turn up to n partner bytes from H1 into empty sites, monomers to own pool | 2 per byte |
| 6A | `TUMBLE` | random new orientation (counter-based RNG, mirror-equivariant) | 2 |
| 6B / 6C | `FREE` / `POOL` | push own cell's free energy / A-monomer count, scaled | 1 |
| 6D | `RAND` | push random 0–255 | 1 |
| 6E | `SLEEP` | pop n; skip execution for n epochs at 1/8 of the upkeep cost | 1 |
| 6F | `HALT` | end this slice | 0 |

All costs are defaults and are overridable per opcode in config.

**Minimal replicator (existence proof)**: `12 13 15 C2` = `CPY H0+ H1+ JMP−3`. With equal-length tapes it copies self onto partner byte by byte, about 4 steps per byte, so 64 bytes take 256 steps, inside the 512-step budget. A unit test asserts this replicator works in the `paper` preset.

## 6. Interaction model

Interaction is hybrid. Every slice starts paper-style, paired with one random face neighbor in random order. The program can then retarget its partner by turning. Both styles share one mechanism: the upper half of the address space is whichever neighbor the cell faces.

1. **Slice start.** The scheduler picks the executing cell (§11). Its orientation is kept. If `random_initial_facing` is on (the default), the facing is first re-drawn from the counter-based RNG, which reproduces the paper's random pairing. The pair order (self-first or partner-first) is drawn when `pair_order = random`.
2. **During the slice.** `YAW`, `PITCH`, `ROLL` and `TUMBLE` change the facing. The next memory access then sees the new neighbor's tape. Heads keep their numeric values, so a head pointing past `len_self` now points into the new partner.
3. **Direct effects.** `MOVE`, `BOND`, `SHARE`, `ALLOC`, `DIVIDE`, `DIGEST` and `TAKE` act on the faced or lateral neighbor without going through the address space.
4. **Writes into a partner** are subject to soft geometric shielding (§8) and matter exchange (§7). A failed write still costs its energy.
5. **Who pays.** The executing cell pays every cost, even when the partner's code is running. This leaves room for parasites, which run their code on a host's energy, as in Tierra.

**Paper-compatible mode** (`interaction = pair_concat`): turning ops still change orientation but no longer remap the partner, all costs are 0, matter and shielding are off, and tapes are fixed at 64 bytes. This should reproduce the paper's soup.

## 7. Physics: energy, temperature, chemistry, matter, decay

There are five resource systems. Each can be switched off on its own, and each moves only through local integer fluxes, so totals are conserved exactly except at logged sources and sinks.

### 7.1 Transport (shared by every field)

Every diffusing quantity q (free energy, heat, chemicals, monomers, signals) moves by an antisymmetric per-face flux. Each face's flux is computed once and applied to both cells, so the sum is conserved by construction:

```latex
J_{a\to b} = \left\lfloor \frac{\kappa_q\,(q_a - q_b) + r_{ab}}{2^{16}} \right\rfloor, \qquad q_a \mathrel{-}= J_{a\to b},\; q_b \mathrel{+}= J_{a\to b}
```

Here κ is a Q16 diffusion coefficient, and r is a dither drawn from the face's counter-based RNG in \[0, 2^16), which lets small gradients still move on average. For stability, κ × (number of faces) must be at most 2^15; config validation rejects violations.

### 7.2 Energy

- **Free energy F** (in the environment) is injected by sources, diffuses with κ\_F, and dissipates at rate λ\_F (dissipated energy becomes heat).
- **Absorption.** An occupied cell moves a share of its cell's F into its internal energy E each epoch, scaled by its exposure e ∈ \[0, 1\] (the fraction of unbonded faces, §8). Shielded interior cells absorb less.

```latex
\Delta E = \left\lfloor \alpha \cdot e \cdot F / 2^{16} \right\rfloor
```

- **Upkeep.** Each occupied cell pays u₀ + u₁ · len per epoch for existing. If E can't cover it, the cell halts.
- **Spending.** Every instruction cost and every upkeep payment turns into heat in that cell (§7.3).
- **Bonded sharing.** Bonded neighbors equalize a fraction β of their energy difference each epoch, plus whatever `SHARE` moves.
- **Ledger.** Sources = Δ(ΣF + ΣE) + dissipation + spending − net reaction energy. This is checked exactly every epoch.

**Sources** (any combination, each time-varying):

| Source | Behavior | Key parameters |
| --- | --- | --- |
| Sunlight | 2D: uniform flux with an optional linear gradient along y. 3D: enters through the top face and is attenuated by each occupied cell in the column (self-shading, Beer–Lambert per cell) | `I0`, `gradient`, `attenuation` |
| Vents | Point sources of F, heat and vent mineral X4. Spawn and die as a Poisson process; may drift | `rate`, `lifetime`, `spawn_rate`, `drift` |
| Cycles | Day/night multiplier max(0, sin 2πt/P\_day), seasonal amplitude over P\_season | `P_day`, `P_season`, `amplitude` |
| Chemical-only | Sun and vent F are off; vents emit only X4, so energy comes only from reactions | `mode = chemical_only` |

### 7.3 Temperature

T is a diffusing heat field (κ\_T). It relaxes toward ambient T\_amb at rate λ\_T. Heat comes from vents, absorbed sunlight and spent energy (metabolic heat, coefficient h), so dense active colonies warm their surroundings.

Temperature scales the rates of racemization, uncatalyzed reactions, decay and background mutation through an Arrhenius factor per process. It is precomputed once in f64 on the CPU into a 4096-entry Q16.16 table, so GPUs never evaluate exp:

```latex
A_p(T) = \exp\!\left(-\frac{E_p}{k}\left(\frac{1}{T} - \frac{1}{T_{\mathrm{ref}}}\right)\right)
```

Each process p has its own activation constant E\_p. Setting it to 0 makes that process temperature-independent.

### 7.4 Chemistry

There are K species (default 4), each with an L and a D enantiomer, stored as integer counts. Reactions are declared for L in config; the D versions are generated automatically as exact mirrors. Cross-hand reactions do not exist unless declared. Each species has a mass in atom units, and declared reactions must balance mass (checked at load).

Default network: a light-driven redox cycle plus vent chemistry.

| Reaction | Mass | Energy | Kind |
| --- | --- | --- | --- |
| X1 + X2 → X3 | 1 + 1 → 2 | releases 12 | catalyzable (`CAT`); uncatalyzed rate tiny |
| X3 → X1 + X2 | 2 → 1 + 1 | driven by light | photolysis; rate proportional to local light |
| X4 → X3 | 2 → 2 | releases 20 | catalyzable; X4 is emitted by vents |
| X3 → X1 + X2 (thermal) | 2 → 1 + 1 | absorbs 12 from F | uncatalyzed, Arrhenius |

Uncatalyzed events per cell per epoch = ⌊k · A(T) · Πcounts + dither⌋, capped by available reactants. Catalysis (§9) delivers the energy released to the executing cell's E; uncatalyzed reactions deliver it as heat. Vents are logged chemical sources; absorbing walls and an optional global sink rate are logged sinks.

### 7.5 Matter (tape monomers)

Each cell has three free-monomer pools: m\_L, m\_D and m\_A. A non-zero tape byte contains exactly one monomer of its class (§5). Total matter, Σ pools + non-zero tape bytes, is constant for the whole run.

- **Writing** (`WR`, `CPY`, copy errors, mutation, `INC`-style rewrites) releases the old byte's monomer to the destination cell's pool and takes one of the new byte's class. The destination pool is tried first, then the executing cell's pool. If neither has one, the write fails and the byte is unchanged.
- Writing 0x00 only releases.
- `GROW` needs no monomer, since an empty site holds nothing. `ALLOC` creates empty sites. `TAKE` moves monomers between pools.
- Pools diffuse slowly (κ\_m), and free L and D monomers racemize (§9).
- Initial matter is set by `matter_density`; initial random tapes draw from it.

### 7.6 Decay

Each non-zero tape byte independently becomes an empty site with probability:

```latex
p_{\mathrm{decay}} = \delta_0 \cdot A_{\mathrm{decay}}(T) \cdot \left(1 + \rho \cdot e \cdot I_{\mathrm{local}}\right)
```

The released monomer goes to the cell's pool. Exposure e and local light I make sunlit surfaces decay faster, so shielding pays off. A tape that has been all empty sites for `dead_after` epochs is cleared and the cell becomes empty.

## 8. Bodies: shielding, bonds, movement, sensing, signals

Nothing in the physics defines an organism. It provides three cheap primitives (bonds, shielding that comes from being surrounded, and body movement) and leaves it to evolution whether bonded bodies are worth their cost.

### 8.1 Soft geometric shielding

Protection comes from coverage. The more of a cell's faces are covered by neighbors, especially bonded ones, the harder it is for a foreign program to write into it. Being covered also lowers its exposure to light, radiation and decay. A write from executor W into target X succeeds with probability:

```latex
p_{\mathrm{write}} = (1 - s_b)^{\,b_X} \cdot (1 - s_o)^{\,o_X}
```

- b\_X is the number of X's bonded faces, not counting a face bonded to W.
- o\_X is the number of X's faces that are occupied but unbonded, not counting W's face.
- Defaults are s\_b = 0.25 and s\_o = 0.10. A surface cell bonded on every face except the attacker's resists 76% of foreign writes in 3D (5 bonds) and 58% in 2D (3 bonds). A cell bonded on all faces has no foreign neighbor, so outsiders can't reach it at all.
- If W is bonded to X, then p = 1. A body's own cells can always edit each other.
- `shield_granularity = slice` (the default) draws one pass/fail per partner per slice, so a break-in is all or nothing. The `byte` setting draws per write, which gives leaky, garbled copies.

The same coverage gives exposure e = 1 − (b + w\_o · o) / faces, with w\_o = 0.5 by default. Exposure scales energy absorption (§7.2), radiation mutation (§10) and light-driven decay (§7.6). That makes it a real trade-off: interior cells are safe but starve unless the body shares energy inward.

### 8.2 Bonds

- Bonds are symmetric face bits, set or cleared together on both cells. The executor owns its whole neighborhood during a slice (§11), so this is race-free.
- They are formed by `BOND` (faced) and `BONDS_L`/`BONDS_D` (lateral), and only to an occupied cell.
- They break by `UNBOND`, when either cell is cleared, or thermally with probability b₀ · A\_bond(T) per epoch, so hot places pull bodies apart.
- Across a bonded face, energy equalizes by β per epoch, and chemicals and monomers diffuse γ times faster (default γ = 4). This is a shared cytoplasm.
- Bonded cells can't be swapped by an outsider's `MOVE`.

### 8.3 Movement

- **Unbonded cells** move immediately. `MOVE` and `STEP_L`/`STEP_D` swap the individual with the target cell if that cell is empty or holds an unbonded individual.
- **Bonded cells vote.** The move becomes a vote for a world-frame direction on the body (the connected component of the bond graph, recomputed each epoch with deterministic min-label propagation). After the execution phase, a body moves one step in direction d if its votes for d reach θ · size (default θ = 0.5), d has strictly more votes than any other direction, every destination cell is empty or belongs to the same body, and no other body claims any of those cells. Conflicting claims cancel each other symmetrically.
- Bodies larger than `max_body_move` cells (default 4096) can't move.
- Individuals move; environment fields (F, T, chemicals, pools, signals) stay with the lattice. Moving is therefore also foraging.

### 8.4 Orientation

Orientation is an element of the proper rotation group: 4 values in 2D and 24 in 3D. It holds a facing vector and an up vector. The lateral left face is the facing rotated +90° about up, and right is −90°. Rotation tables are precomputed constants. In 1D, `YAW` reverses facing and lateral ops are no-ops.

### 8.5 Sensing and signals

- Sensors (`SENSE`, `SENS_L`/`D`, `ENRG`, `TEMP`, `FREE`, `POOL`, `CHEM_L`/`D`, `SIG_L`/`D`) cover local resources, neighbor occupancy and bonds, and signals. Neighbor bytes are read through the address space with `RD`.
- **Signals** are S chiral signal species (default 2 × L/D). They are emitted by `EMIT_L`/`D` (σ units per op), diffuse fast (κ\_s) and decay fast (λ\_s). They are not matter; their cost is the energy spent emitting them.

## 9. Chirality

Handedness is carried by instructions, chemicals, monomers and lateral body ops. By default the physics is exactly mirror-symmetric. So if one hand comes to dominate, that is spontaneous symmetry breaking, and the twin test (§11, §16) proves no bug caused it.

### 9.1 What is chiral

| Entity | Chiral forms | Achiral counterpart |
| --- | --- | --- |
| Instructions | 16 L/D pairs, 0x20–0x3F (§5): turns, lateral bonds, sensors and steps, catalysis, monomer uptake, signals, filler | stack, heads, jumps, push, forward-facing ops |
| Chemicals | every species X1…XK as L and D | optional achiral species (config) |
| Monomers | m\_L, m\_D: a chiral op byte must be built from its own hand | m\_A builds every other non-zero byte |
| Signals | S species × L/D | none |
| Bodies | shape chirality emerges from lateral ops `BONDS_L`/`D` and `STEP_L`/`D` | forward `BOND`, `MOVE` |

### 9.2 The mirror transformation M

M reflects the lattice on axis 0 (x → N−1−x) and also:

- maps every tape byte through `mirror(b)`;
- swaps L and D in chemicals, monomer pools and signals;
- reflects every orientation's facing and up vectors. Lateral left and right are derived from them, so the effect of each `_L` op becomes that of its `_D` twin.

**Requirement:** step(M(S)) = M(step(S)), bit for bit, for every state S when all bias parameters are 0. To achieve this, every source of asymmetry in the implementation must be equivariant:

- **RNG.** Draws are keyed by canonical cell coordinates and per-face ids. The twin uses reflected keys, and any draw that picks a direction is read in the reflected frame.
- **Scheduler.** The color order of sub-steps is permuted with the reflected mapping (§11).
- **Tie-breaks.** Never "lowest index wins" on a spatial choice. Conflicts cancel symmetrically (§8.3), and labels are only identifiers.
- **Fields.** Gradients and sources are defined in mirror-covariant terms (sun gradient on y, vent positions drawn through the RNG).

### 9.3 Stereo-specific catalysis and folding

This is the built-in reason for hand to matter, modeled on enzymes: a catalyst works only on its own hand's substrate, and only if its own chain is mostly of one hand.

- `CAT_h` runs reaction (top mod R) using only hand-h reactants in the executing cell.
- **Folding.** The active site is the w bytes before the `CAT` op on the executing tape (default w = 8). Let n\_same and n\_opp be the chiral bytes in that window of the same and opposite hand as the `CAT` op. Achiral bytes are neutral.

```latex
\eta = \eta_{\min} + (1-\eta_{\min})\,\frac{\max(0,\; n_{\mathrm{same}} - n_{\mathrm{opp}})}{w}
```

- **Cross-inhibition.** The opposite enantiomer of the first reactant poisons the site (Frank-model mutual antagonism): η is multiplied by 1 − κ · c\_opp / (c\_same + c\_opp).
- **Events per op** = ⌊n\_cat · η + dither⌋, capped by the available reactants. Released energy goes to the executor's E. Defaults: η\_min = 0.1, n\_cat = 4, κ = 0.5.
- **Control switch** `stereo = off`: `CAT` uses either hand's reactants and η = 1. This baseline separates a chemistry-driven win from a pure founder effect.

### 9.4 Chiral matter: a built-in counter-force

This follows real chemistry. A residue's hand is the hand of the monomer it was built from, so an L amino acid can only become an L residue unless it racemizes first. The model adds the second thing believed to happen in nature: copying is not automatically stereoselective.

- **Matching monomers (kept).** A chiral byte uses a monomer of its own hand. A population dominated by L depletes local m\_L and makes D cheaper to build, which favors the rarer hand and pushes toward racemic coexistence.
- **Wrong-hand incorporation (new).** In prebiotic, non-enzymatic copying, opposite-hand monomers get built into chains and poison them. This is enantiomeric cross-inhibition, reported by Joyce et al. (1984) for template-directed nucleotide copying. In PISA-1, when `CPY` or `WR` writes a chiral byte, it writes mirror(b) instead with probability (1 − σ) · m\_opp / (m\_same + m\_opp).
- **Selectivity depends on the copier (new).** Stereoselectivity σ is computed like folding (§9.3), over the w bytes before the copy op: σ = σ\_min + (1 − σ\_min) · max(0, n\_same − n\_opp) / w. Here 'same' means the majority hand of that window. Racemic copiers are sloppy, as prebiotic chemistry is; homochiral copiers become stereoselective, as modern polymerases are. Default σ\_min = 0, with `stereo_copy = off` as an ablation.

Nature thus supplies two opposing forces: depletion favors the rarer hand, while cross-inhibition and selective copying favor whichever hand already leads. Which one wins, and at what monomer density and racemization rate, is one of the main experiments (§13).

### 9.5 Racemization

- Free monomers and chemicals switch L ↔ D at rate k\_rac · A\_rac(T) per unit per epoch, the same in both directions by default. Vents and warm regions racemize faster.
- Code changes hand through mutation: the hand-flip channel and bit-0 flips (§10). Bytes inside tapes do not racemize on their own.
- k\_rac = 0 freezes free material's hand. High k\_rac keeps the environment racemic, so only code can carry an enantiomeric excess.

### 9.6 Parity-violation bias (optional, default 0)

| Parameter | Effect |
| --- | --- |
| `bias.racemization` ε | L→D runs at (1−ε) of the D→L rate |
| `bias.reaction_energy` ε | L reactions release (1+ε) times the energy |
| `bias.initial_ee` ε | adds ε to the initial enantiomeric excess of all material |

Any non-zero bias is written into the run record. The twin check is then disabled, unless the twin is given the mirrored bias.

### 9.7 Initial hand balance

`init_ee` takes one of three values:

- `random` (the default): whatever the random tapes and pools give, as in nature;
- `racemic`: after generation, deterministic RNG-chosen hand flips make global L and D counts equal for each material type, with every cell within ±1;
- `seeded(x)`: set the excess to x.

## 10. Mutation

There are three independent mutation channels plus a hand-flip channel. All of them go through matter exchange (§7.5), so a mutation that needs a monomer the cell's pools don't have is skipped. The paper found emergence happens even with zero mutation, so every rate may be 0.

| Channel | What happens | Rate per byte per epoch (or per copy) | Default |
| --- | --- | --- | --- |
| Background | flip one uniformly chosen bit of a non-empty tape byte | μ\_bg · A\_mut(T) · M(x) | μ\_bg = 2.4×10⁻⁴ per epoch (the paper's 0.024%) |
| Radiation | same bit flip, on exposed bytes in light | μ\_uv · e · I\_local | 1×10⁻⁴ at full sun |
| Copy errors | on `CPY`/`WR`: substitute (bit flip), insert a duplicate byte after H1, or delete the byte at H1 | p\_sub, p\_ins, p\_del per copy | 1×10⁻³, 2×10⁻⁴, 2×10⁻⁴ |
| Hand flip | chiral op b → mirror(b) | μ\_flip · A\_rac(T) | 1×10⁻⁵ |

- **Location dependence.** M(x) is a spatial multiplier map: `uniform`, `vent_radial` (1 + a · exp(−r/r₀) around active vents) or `image` (a grayscale PNG at lattice resolution). Combined with the temperature and light terms, mutation is location-dependent in three ways, as requested.
- **Bit-0 flips.** A bit flip at bit 0 of a chiral op is automatically a hand flip, about 1/8 of single-bit flips on chiral bytes. The explicit hand-flip channel adds racemization of code that can be tuned on its own.
- **Insertions and deletions** change length. Inserting into a tape at `L_max` is skipped, and a deletion releases the byte's monomer.
- **Sampling.** Per-byte Bernoulli trials are too expensive at about 128M bytes. Each epoch, the number of mutations per cell is drawn from a binomial distribution approximated by inverse-CDF lookup on the counter-based RNG, then the positions are drawn. The table is precomputed in f64 for each rate, so the result is deterministic and matches the binomial to within table resolution (2⁻³²).

## 11. Scheduling and bit-exact determinism

An epoch is a fixed pipeline of phases. Execution is split into 3^D color classes, so cells that run at the same time never touch each other's neighborhoods. All randomness comes from a counter-based RNG keyed by position. Together these make every run bit-exact and every twin run an exact mirror.

### 11.1 Epoch pipeline

1. **Interventions.** Apply any viewer or script events stamped for this epoch, from the event log (§14).
2. **Sources.** Sunlight (including the 3D column attenuation scan), vents, vent spawn/death, cycles.
3. **Transport.** Per-face fluxes for F, T, chemicals, monomers and signals, with bonded faces getting the γ multiplier. Then dissipation, heat relaxation and signal decay.
4. **Chemistry.** Uncatalyzed reactions, photolysis, racemization of chemicals and monomers.
5. **Absorption and upkeep.** F → E by exposure, upkeep paid, bonded energy equalization, halting.
6. **Execution.** 3^D sub-steps (9 in 2D, 27 in 3D). Their order is permuted each epoch by the RNG. Sub-step c runs every occupied, awake, non-halted cell whose coordinates mod 3 equal c. Cells three apart on every axis have disjoint Chebyshev-1 neighborhoods, so each executor exclusively owns its partner, lateral cells and their pools.
7. **Body movement.** Recompute bond components, tally votes, resolve claims, translate bodies.
8. **Mutation and decay.** All channels from §10 and decay from §7.6. Dead-cell clearing.
9. **Bookkeeping.** Energy, matter and atom ledger checks; telemetry sampling (§13); snapshot if due.

In 0D the execution phase is the paper's: a random permutation pairs all N slots and every pair executes once (2^17 slots by default).

### 11.2 Randomness

- **Generator.** Philox4x32-10, a counter-based generator: no state is stored, and every draw is a pure function of (key, counter).
- **Key.** (run seed, epoch, phase id).
- **Counter.** (canonical cell index or face id, stream id, draw index). In twin mode, canonical coordinates are reflected and direction-valued draws are read in the reflected frame (§9.2).
- **Seeds.** No global sequential RNG exists anywhere in the simulation. A per-run seed and the config fully determine the run.

### 11.3 Determinism rules

- **No floating point in state updates.** Floats are allowed only in the viewer, in telemetry derived from state, and in tables precomputed on the CPU (Arrhenius, binomial inverse CDFs, rotations), which are uploaded as integers.
- **Fixed-point arithmetic.** Q16.16 for rates; integer counts for quantities; WGSL wrapping integer semantics.
- **No order-dependent atomics on state.** Atomics are used only for integer statistic counters, where addition commutes. Each phase writes either cell-owned data or face-owned flux buffers.
- **No subgroup operations in the state path.** Subgroup sizes differ by vendor.
- **Run identity.** A run is fully defined by config + seed + event log. Its state hash is a 64-bit FNV-1a over all buffers in a fixed order, logged every `hash_every` epochs (default 100).

### 11.4 Twin mode

`primordia twin` runs world A and world B = M(A) in lockstep with all bias = 0. Every `hash_every` epochs it checks hash(M(state\_A)) == hash(state\_B). On a mismatch it bisects to the first diverging epoch and phase, then dumps that cell's neighborhood. Like ledger checks, this is a correctness tool. A twin divergence blocks any chirality claim.

## 12. Architecture: Rust + wgpu

The simulation lives in WGSL compute kernels over structure-of-arrays buffers. A CPU reference interpreter in Rust shares the same generated opcode tables and serves as the golden model. A 1M-cell 2D world fits in about 250 MB of GPU memory without tracers.

**Crates**

| Crate | Responsibility |
| --- | --- |
| `primordia-core` | Config schema and validation, ISA tables, mirror map, rotation tables, Philox RNG, CPU reference implementation of every phase (golden model) |
| `primordia-gpu` | wgpu device setup, buffers, WGSL kernels, epoch pipeline, snapshot upload/download |
| `primordia-telemetry` | Ledgers, complexity, chirality and species metrics, lineage log, Parquet writer |
| `primordia-view` | egui + wgpu viewer, inspector, intervention tools (§14) |
| `primordia-cli` | `run`, `batch`, `replay`, `branch`, `twin`, `assay`, `bench` (§15) |

A `build.rs` generates the WGSL constant includes (opcode table, costs, mirror map, rotation tables) from `primordia-core`. CPU and GPU can't disagree on semantics because there is one source of truth.

**GPU memory at 1M cells, L\_max = 128**

| Buffer | Per cell | Total |
| --- | --- | --- |
| Tape arena (u8 × L\_max) | 128 B | 128 MB |
| Cell header: len, flags, orientation, bonds, E, lineage, birth, sleep | 24 B | 24 MB |
| Fields, ping-pong: F, T, 4 species × 2 hands, 3 pools, 2 signals × 2 hands | 2 × 36 B | 72 MB |
| Face flux scratch (reused per field) | 3 × 4 B | 12 MB |
| Body labels and votes | 16 B | 16 MB |
| Tracer tokens (optional, u32 per tape byte) | 512 B | 512 MB, so tracers are limited to ≤ 256k cells by default |

**Kernels**

| Kernel | Phase | Thread per |
| --- | --- | --- |
| `sources`, `sun_column_scan` | 2 | cell / column |
| `face_flux`, `apply_flux` (per field group) | 3 | face / cell |
| `chem_react`, `racemize` | 4 | cell |
| `absorb_upkeep_share` | 5 | cell |
| `exec_color` (× 3^D dispatches) | 6 | executing cell |
| `bond_label` (iterate to convergence), `body_vote`, `body_claim`, `body_move` | 7 | cell |
| `mutate_decay` | 8 | cell |
| `ledger_reduce`, `stats_reduce`, `state_hash` | 9 | workgroup tree |

**Interpreter kernel.** One thread per executing cell. Each thread copies its own and partner tapes (up to 2 × L\_max bytes) into workgroup memory (64 threads × 256 B = 16 KB), runs up to `step_budget` steps through a switch on the opcode, and writes back. Divergence is accepted; the paper's cubff shows this style is fast enough. Partner retargeting reloads only the upper half.

**Performance targets** (one RTX 4080-class GPU, 2D, step\_budget 512), to be confirmed by `primordia bench` in phase 1:

- at least 1 epoch/s at 1M cells;
- at least 8 epochs/s at 250k cells;
- 0D paper preset (2^17 × 64 B) at least 20 epochs/s.

## 13. Telemetry and analysis

Every run writes columnar logs (Parquet) of ledgers, complexity, chirality, species and lineage, plus periodic snapshots. These are enough to answer the five research questions in §1 without re-running.

### 13.1 Metrics

| Metric | Definition | Cadence |
| --- | --- | --- |
| Ledgers | energy, matter (L/D/A), chemical atoms; logged residual must be 0 | every epoch |
| Population | occupied, halted, sleeping cells; mean and histogram of tape length; births, deaths | every epoch |
| Opcode histogram | 256-bin count over all tape bytes | every 10 epochs |
| High-order entropy | Shannon entropy of bytes minus Brotli (quality 2) compressed bits per byte, as in the paper; computed on CPU over 65,536 sampled tapes | every 100 epochs |
| Unique tracer tokens | count of distinct (epoch, position, char) tokens, as in the paper (tracer mode only) | every 100 epochs |
| Bodies | number of bonded components; size distribution; mean exposure; moved bodies | every 10 epochs |
| Ecology | mean F, T and chemical counts by region (16^D coarse blocks) | every 100 epochs |
| State hash | FNV-1a over all buffers | every `hash_every` |

### 13.2 Replicators, species, lineage

- **Replicator assay.** Every 1,000 epochs, the 100 most common genomes are run in a sandbox: genome + empty partner, all physics off, the paper's test. Those whose copy reaches ≥ 90% byte identity are logged as replicators, with their minimal functional substring (found by ablating bytes one at a time). The first-replicator epoch is a headline result.
- **Species census.** Genomes are clustered by MinHash over byte 4-grams with LSH; same species means Jaccard ≥ 0.8. For each species: abundance, spatial centroid and extent, mean length, hand (sign of code EE), first and last seen.
- **Lineage.** `ALLOC` and `DIVIDE` create a child with a new lineage id. Each event (parent, child, epoch, position, genome hash) is appended to `lineage.parquet`. Pairwise copying doesn't create ids, so tracer tokens cover that case as in the paper.
- **Organism candidates.** A body of size ≥ 2 whose cells share a lineage ancestor within the last `k` divisions, and that has itself produced a child body. This is the operational definition of an emergent multicellular organism.

### 13.3 Chirality

- **Enantiomeric excess**, EE = (N\_L − N\_D) / (N\_L + N\_D). It is computed separately for chiral code bytes, each chemical species, free monomers and signals, globally and per 16^D block, every 10 epochs.
- **Domain maps.** The per-block code EE is smoothed and thresholded at |EE| > 0.3. The viewer overlay and logs record domain count, mean domain size, total interface length and interface velocity.
- **Organism hand.** The sign of each tape's code EE. For bodies, a geometric chirality index: 1 if the body is not superimposable on its mirror image by any rotation (checked by canonical-form comparison, for bodies of ≤ 64 cells).
- **Symmetry-breaking time.** t\_b is the first epoch at which |EE\_code| > 0.5 and stays there for 1,000 epochs. The winning hand is the sign at t\_b.
- **Twin runs**: see §11.4.
- **Breaking statistics** (`primordia batch` across seeds): the winning hand's frequency with a binomial test against 50/50, the distribution of t\_b, and final |EE| for code, chemicals and monomers.

### 13.4 Built-in experiment presets for homochirality

| Experiment | Varies | Question |
| --- | --- | --- |
| C0 founder baseline | `stereo = off` | Does code EE reach ±1 just because one replicator lineage takes over? |
| C1 stereo on | `stereo = on`, κ sweep | Does catalysis make the environment's chemicals homochiral too, not just code? |
| C2 monomer counter-force | `matter_density` × k\_rac sweep | Where does frequency-dependent selection (§9.4) keep the world racemic? |
| C3 dimension | 0D vs 2D vs 3D | Does space allow stable L and D domains to coexist? |
| C4 temperature | vent count and heat | Do hot regions stay racemic while cold ones break symmetry? |
| C5 parity bias | `bias.*` ε from 10⁻⁶ to 10⁻² | How small a bias decides the winner more often than chance? |

### 13.5 Snapshots

Snapshots are full state dumps (all buffers plus config hash and epoch), zstd-compressed, written every `snapshot_every` epochs (default 10,000) and on demand. Resuming from a snapshot must give the same hash as an uninterrupted run (§16). A branch is a snapshot plus a config diff, recorded in the new run's manifest with its parent run id.

## 14. Viewer, inspection and interventions

The viewer is a native egui + wgpu app that renders straight from the simulation buffers. Every intervention it makes is written to the run's event log, so a run you poked by hand still replays bit-exactly.

### 14.1 Views

- **2D map**, with these color modes:
  - dominant opcode group;
  - code hand (red = L, blue = D, saturation = |EE|);
  - species (hash to hue);
  - lineage depth;
  - energy E or free energy F;
  - temperature;
  - one chemical's count or its EE;
  - monomer pools;
  - signals;
  - exposure.

  Bond overlay (lines between bonded cells), vent markers and chirality domain outlines can be toggled.
- **3D**: three orthogonal slice panels with draggable slice indices, plus an optional raymarched volume of one scalar field (phase 5).
- **Plots panel**: live time series of any logged metric. The default panel shows complexity, population, EE (code, chemicals, monomers) and ledger residuals.
- **Playback**: run, pause, step one epoch, step one execution sub-step, and run headless with redraws every N epochs.

### 14.2 Inspector

Click a cell to see:

- tape hex dump with disassembly, with chiral ops colored by hand and the active-site window of each `CAT` highlighted with its η;
- CPU debugger, which re-runs the cell's next slice on the CPU golden model, step by step, showing registers, stack, heads, partner and energy (read-only: it doesn't change the world);
- energy, temperature, pools and chemicals;
- bonds and body id;
- species and lineage chain back to the first ancestor on record.

### 14.3 Interventions

Every intervention is a logged event `{epoch, kind, params}`, applied at phase 1 of that epoch.

| Tool | Effect |
| --- | --- |
| Edit physics | change any config value live (rates, costs, sources, bias). The validator runs first; rejected edits are not applied |
| Paint resources | brush F, heat, chemicals (by species and hand) or monomers into a region |
| Place vents | add, move or remove vents |
| Seed programs | paint genomes from a library (minimal replicator, saved species, hand-written) into cells; optional mirror-image placement |
| Clear / kill | empty a region; matter returns to pools so the ledger still balances |
| Mirror region | apply M to a region, to test whether mirror invaders behave identically |
| Walls | paint absorbing or reflecting obstacles |
| Branch | fork from the current state or any snapshot into a new run, with a config diff |

Every intervention that adds energy or chemicals is a logged ledger source, and painting monomers is a logged matter source.

## 15. Configuration, CLI and file formats

A run is one TOML config, which may inherit from a preset, plus a seed. Every mechanism has an `enabled` flag. Output is a self-describing run directory.

### 15.1 Config (excerpt)

```toml
preset = "primordia-2d-sun"   # inherit, then override
seed = 42
epochs = 1_000_000

[world]
dims = 2
size = [1023, 1023]
boundary = ["torus", "torus"]

[individual]
l_init = 64
l_max = 128
step_budget = 512
pair_order = "random"
random_initial_facing = true
interaction = "hybrid"        # or "pair_concat"

[energy]
enabled = true
upkeep = { base = 1, per_byte = 0.02 }
absorb_alpha = 0.05
sun = { enabled = true, i0 = 40, gradient_y = 0.5, day_period = 2000 }
vents = { enabled = true, count = 8, rate = 400, lifetime = 50_000 }

[chemistry]
enabled = true
species = ["X1", "X2", "X3", "X4"]
reactions = [
  { r = ["X1", "X2"], p = ["X3"], energy = 12, catalyzable = true },
  { r = ["X4"], p = ["X3"], energy = 20, catalyzable = true },
]

[chirality]
stereo = true
fold_window = 8
eta_min = 0.1
cross_inhibition = 0.5
racemization = { rate = 1e-4, activation = 0.8 }
init_ee = "random"
bias = { racemization = 0.0, reaction_energy = 0.0, initial_ee = 0.0 }

[shielding]
enabled = true
s_bond = 0.25
s_occupied = 0.10
granularity = "slice"

[mutation]
background = 2.4e-4
radiation = 1e-4
copy = { sub = 1e-3, ins = 2e-4, del = 2e-4 }
hand_flip = 1e-5
map = "vent_radial"

[telemetry]
hash_every = 100
snapshot_every = 10_000
tracers = false
```

### 15.2 Presets

| Preset | Purpose |
| --- | --- |
| `paper-0d` | 2^17 × 64-byte tapes, pair\_concat, all physics off. Validation against Computational Life |
| `paper-2d` | the paper's 240 × 135 grid (both sides are multiples of 3), physics off. Uses the paper's pairing radius 2 and its random-claim scheduler instead of the checkerboard |
| `primordia-2d-sun` | full physics, sunlight with day/night, no vents |
| `primordia-2d-vents` | chemical-only energy from drifting vents |
| `primordia-3d` | 99³, sunlight from above with self-shading, vents on the floor |
| `chiral-c0` … `chiral-c5` | the homochirality experiments of §13.4 |

### 15.3 CLI

| Command | Does |
| --- | --- |
| `primordia run <config> [--seed N] [--headless]` | run one world, with the viewer unless headless |
| `primordia batch <sweep.toml>` | grid or random sweep over config keys × seeds, one run directory each, summary table at the end |
| `primordia replay <run_dir> [--to EPOCH]` | re-run from config + seed + events and verify state hashes |
| `primordia branch <snapshot> <diff.toml>` | new run from a snapshot with changes |
| `primordia twin <config>` | mirror-twin check (§11.4) |
| `primordia assay <genome.hex>` | run a genome through the replicator assay |
| `primordia bench` | performance benchmark per kernel and per epoch |

### 15.4 Run directory

| File | Format | Contents |
| --- | --- | --- |
| `manifest.json` | JSON | run id, parent run (for branches), git commit, GPU adapter, config hash, seed |
| `config.toml` | TOML | fully resolved config, presets expanded |
| `events.jsonl` | JSON lines | interventions `{epoch, kind, params}` |
| `metrics.parquet` | Parquet | per-epoch ledgers, population and summaries |
| `chirality.parquet` | Parquet | EE series and domain statistics |
| `species.parquet`, `lineage.parquet` | Parquet | census and birth events |
| `hashes.csv` | CSV | epoch, state hash |
| `snapshots/e{epoch:010}.psnap` | zstd binary | magic `PSNP`, format version, config hash, epoch, then each buffer in fixed order with length prefixes |

## 16. Test and validation plan

Three properties must hold before any scientific result is trusted: GPU matches the CPU golden model bit for bit, ledgers balance exactly, and twin worlds stay exact mirrors. Paper replication then checks the dynamics themselves.

| # | Test | Pass criterion | Runs |
| --- | --- | --- | --- |
| T1 | Opcode unit tests on the CPU golden model, one or more cases per opcode, including edge cases (wraps, empty partner, stack limits, failed matter exchange) | all pass | every commit |
| T2 | Mirror map: mirror(mirror(b)) = b; class(mirror(b)) = swap(class(b)); each `_L` op on state S equals the `_D` op on M(S), reflected | all 256 bytes, all ops | every commit |
| T3 | Minimal replicator `12 13 15 C2` copies itself in the `paper-0d` sandbox | exact copy within 512 steps | every commit |
| T4 | GPU vs CPU: 63² 2D and 21³ 3D worlds, full physics, 2,000 epochs | identical state hash every epoch | every commit (small), nightly (large) |
| T5 | Cross-vendor: T4 on NVIDIA, AMD, Apple and Intel adapters | identical hashes across vendors | weekly |
| T6 | Ledgers: energy, matter and atoms with all mechanisms on, 100k epochs | residual exactly 0 | nightly |
| T7 | Twin: 2D and 3D worlds, bias 0, 10k epochs | hash(M(A)) = hash(B) every check | nightly |
| T8 | Resume: run 2N epochs vs N + snapshot + N | identical final hash | nightly |
| T9 | Replay: replay a run with interventions from its event log | identical hashes | nightly |
| T10 | Paper replication: 200 seeds of `paper-0d` for 16k epochs | state transition (high-order entropy > 1) in ≥ 50% of seeds; the paper reports \~40% for BFF and \~100% for its Forth | release |
| T11 | Seeded takeover: one minimal replicator in a random `paper-0d` soup, 128 epochs | takeover in 10–40% of seeds (the paper's seeded BFF case: 22%) | release |
| T12 | Fuzz: random valid configs × random seeds | no panics, no out-of-bounds, ledgers balanced | continuous |
| T13 | Performance: `primordia bench` | meets §12 targets | release |

T1–T4 gate every merge. A failure in T6 or T7 blocks publishing any result from the affected build.

## 17. Roadmap

The build runs in six phases of about 24 weeks for one full-time developer. Each phase ends at a gate from §16, and nothing new is added until the previous gate passes.

&#91;embedded content: roadmap · 6 phases, each closed by a test gate\]

The order is deliberate. Exactness (P1) and conservation (P2) are proven before chirality (P3), so any homochirality result rests on a verified mirror-symmetric engine. Bodies and 3D come last because they are the most expensive and are not needed for the first chirality answer.

## 18. Risks and open questions

The biggest scientific risk is that resource costs stop life from starting at all, as happened with SUBLEQ. The biggest interpretive risk is mistaking a founder effect for real homochirality. Both have built-in mitigations.

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Costs suppress the origin of life | Each cost makes the first replicator rarer (§2, rule 3) | Phase 2 gate. Sweep costs to find the threshold. Allow a staged start (physics ramps on after the first transition) or seeding from evolved replicators, always recorded as such |
| A trivial replicator freezes complexity | The 4-byte copier may win early and nothing more complex can compete, as the paper saw with Forth's trivial replicator | Watch high-order entropy and species counts after the transition. Raise `CPY` cost or add upkeep per byte; vents and cycles create varied niches |
| Founder effect looks like homochirality | One replicator lineage taking over makes code EE ±1 with no chemistry involved | Experiment C0 (`stereo = off`) as the baseline. Report chemical and monomer EE, not only code EE |
| Twin divergence from a subtle asymmetry | Any one-sided tie-break or RNG use breaks the mirror proof | T7 every night, bisecting to epoch and phase; a lint that forbids raw RNG calls outside equivariant helpers |
| Interpreter divergence makes the GPU slow | Branchy code on SIMD hardware | Accept for v1. If bench misses targets, sort executors by opcode mix or use a persistent-thread queue |
| Cross-vendor drift | wgpu backends could differ | Integer-only state path; T5 weekly on four vendors |
| Memory at large L\_max | 256-byte tapes × 1M cells and tracers exceed consumer GPUs | Default L\_max 128; tracers only on ≤ 256k cells or a sampled region |
| 3D orientation bugs | 24-element rotation group, reflected frames | Generated tables, exhaustive unit tests of group closure and mirror conjugation |

**Open questions for later versions**

- **Unify monomers and chemistry**, so tape monomers are built from reaction products instead of a separate pool. This adds metabolism in the strong sense.
- **Variable step budget by energy** (`Energy-weighted` scheduling), as an alternative to the fixed budget.
- **Achiral chemical species** as an extra knob for chirality experiments.
- **Multi-GPU** domain decomposition for worlds over 10M cells.
- **Physics-level ancestor library**: whether to ship hand-written replicators for every preset, or only the minimal one.

