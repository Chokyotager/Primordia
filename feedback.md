# Feedback: autonomous replication and differentiated cells


## Fix status

The confirmed implementation defects listed below have been addressed. The historical analysis below
refers to commit `5829061`; its line-number references and reproduction results describe that revision.
The model-design hypotheses remain experiments to pursue, rather than claims that differentiated
multicellular reproduction is now solved.

| Audit finding | Implemented change |
| --- | --- |
| DIVIDE endpoint; short child allocation | CPU/GPU accept the one-past-end boundary; all tracked seed templates and viewer seeds allocate 15 bytes. |
| D/right mutation bias | Ordinary mutations and copy errors use an equal mixture of a bit-flip kernel and its mirror conjugate, with exhaustive ordinary-kernel symmetry tests. |
| Poisson overflow/underflow | Stable deterministic CDF sampling for higher means; explicit 64-event cap; mean, variance, and CDF regressions through λ=255. |
| Sleep duration; STEP costs | Full future sleep rounds, including final-round shielding; each STEP uses its configured cost. |
| Live edits | Refresh GPU neighbors/tables/transport buffers and cached image maps; reject incompatible layouts; snapshots carry their effective rules to the observer. |
| Resume/replay | Effective configuration sidecars, recovery of prior configuration events, and local replay-origin snapshots for resume and branch. |
| Snapshot accounting/parser | Preserve epoch-specific ledger/tallies, residuals, violation history, and light in v2; read v1 with rebased accounting; validate every block and state-array size. |
| Configuration/backend limits | Validate unsafe fractions and reaction counts; handle empty chemistry; dynamic transport/birth capacity; aggregate all vents without truncation. |
| Region mirroring | Move walls/environment/vents with cells and reflect pending adhesion requests and movement votes. |
| Telemetry | Retest trimmed genomes; include NOP-prefixed kin; unique inherited cluster IDs after splits; epoch-bucketed births and independent day-boundary samples. |
| Run artifacts/inputs | Reject nonempty output directories and missing explicit event files before creating a run. |

Regression coverage also includes a hand-authored allocating copier whose isolated offspring reproduces
for three generations under the same physics, and CLI continuation/replay after the parent directory
has moved. The optional takeover test now synchronizes resident GPU state before measuring its result.

Compatibility: these corrections change trajectories and old recorded hashes. Legacy snapshots can be
continued under the corrected rules, while exact historical replay requires the original revision.
The new sampler interpolates adjacent Poisson CDFs on a 1/64 grid above λ=4 (extra uncapped variance at
most 1/16384); this numerical approximation and the 64-event cap are explicit. Numeric RD/WR program
semantics can still introduce handedness effects beyond the corrected mutation kernel.

Only versioned seed templates were corrected; existing experiment output directories were not rewritten.
The audit's original validation record is retained at the end of this file.

### Implementation validation

`RAYON_NUM_THREADS=4 cargo test --offline --release --workspace -- --nocapture --test-threads=1`
completed successfully: **75 passed, 0 failed, 1 ignored**. This includes 7 core unit tests, 28 existing
core integration tests, 14 new core regressions, 12 GPU parity/regression tests, 11 telemetry tests,
2 CLI continuation tests, and the viewer rendering/inspection test. The GPU tests ran on the RTX 3050;
they were not skipped. The ignored test is the optional long takeover experiment, which was not run.

The mutation-only experiment starts 100,000 cells with the achiral four-byte copier, applies one epoch
of background mutation at rate 0.1 without execution, and measures `(L-D)/(L+D)`:

| Seed | L bytes | D bytes | Handedness imbalance |
| --- | ---: | ---: | ---: |
| 1 | 1,782 | 1,840 | −0.01601 |
| 2 | 1,746 | 1,831 | −0.02376 |
| 3 | 1,841 | 1,810 | +0.00849 |

These results and the exhaustive transition-law test address the mutation bias, not every possible
source of handedness in executing populations. The isolated copier regression verifies three successive
generations under controlled unchanged physics; it does not demonstrate evolved differentiated bodies.

GPU regression coverage includes 16,500 births in one epoch, 4,097 vents, maximum supported field counts,
large mutation means, live boundary/reaction edits, and epoch-specific snapshot accounting. CLI tests
compare continuous and resumed state hashes, recover configuration without a sidecar, and replay new
resume/branch directories after the parent directory has moved. `git diff --check` also passed.

**Unresolved verification issue:** an earlier parallel workspace run passed, but a subsequent parallel
rerun stalled with GPU utilization at zero and multiple test threads waiting. That stalled test process
was terminated; the complete serial run above passed. The cause of the concurrent stall was not
established, so the serial result should not be taken as proof that concurrent GPU test execution is
reliable. No long evolutionary run was performed.

## Historical audit

Review of commit `5829061`, including the CPU interpreter and physics, corresponding GPU execution paths, configuration defaults, reproduction tests, telemetry, experiment notes, and saved M15/M16 results. At audit time, only this report was added to the repository. Diagnostic programs and build outputs were placed under `/tmp`.

**My main conclusion:** Primordia provides strong routes for programs to spread by overwriting other cells, but much weaker routes for an individual to preserve its identity, finish an offspring, and reproduce a differentiated organization. The observed short copiers, mixed mats, and unstable bodies are plausible outcomes of those rules. Increasing genome capacity, chemical reserves, or run duration alone does not remove these pressures.

The follow-up audit also confirmed an encoding-induced D/right mutation bias, a faulty Poisson sampler, and defects in live configuration changes, resume/replay, asynchronous snapshots, and telemetry. These need separating from evolutionary explanations. Fixing them would make the experiments more trustworthy, but would not by itself remove the lifecycle obstacles described below. Several of those obstacles are deliberate semantics inherited from the program-soup model.

## What the existing runs show

I read the saved histories and inspected these snapshots with the current core code:

| Measurement | M15, epoch 649,000 | M16, epoch 11,000 |
| --- | ---: | ---: |
| Occupied cells | 238,290 | 226,420 |
| Mean allocated tape length | 28.93 | 57.14 |
| Mean nonempty bytes per occupied cell | 15.57 | 26.07 |
| Cells carrying the halted flag | 187,735 (78.8%) | 145,536 (64.3%) |
| Occupied cells with zero geometric exposure | 115,466 (48.5%) | 33,350 (14.7%) |
| Largest bonded body, from saved history | 182,125 | 42,623 |

Sources: `runs/live-m15-s2/snapshots/e0000649000.psnap`, `runs/live-m16-s2/snapshots/e0000011000.psnap`, and their respective `history.json` files. These are individual observations, not averages over a day or independent experimental replicates. A halted flag does not necessarily mean zero stored energy: a cell can be accumulating energy below its wake threshold.

The most frequent exact tapes included:

- M15: `12 64 67 13` (639 copies), `12 64 67 13 15 c4` (480 copies).
- M16: `12 47 67 13 15 c4 12` (472 copies), `12 4f 67 13 15 c4 12` (333 copies).

These are dominated by copying, allocation, and head movement. The M15 history reports about 15.8 prefix-defined types per counted body at the final sample, but only one kin body with two or more such types. That is consistent with a large mixed mat; it does not establish functional division of labour. The available M16 history ends at 11,000 epochs, only 5.5 day periods, so it cannot establish M16's long-term outcome.

## 1. A cell does not reliably execute its own program

**Confirmed semantics; high-priority design obstacle.**

The resolved M16 configuration inherits both:

```toml
[individual]
pair_order = "random"
random_initial_facing = true
```

In `exec_slice`, the interpreter starts at IP 0 and randomly chooses whether the concatenated address space starts with self or partner. With an occupied partner, half the slices therefore start by executing the neighbor's program, while applying operations and charging energy to the executing cell. The facing is also redrawn on every slice, despite the setting's name suggesting initialization only.

References: [defaults](crates/primordia-core/src/config.rs#L430), [address resolution](crates/primordia-core/src/exec.rs#L121), [slice initialization](crates/primordia-core/src/exec.rs#L903), [GPU equivalent](crates/primordia-gpu/src/exec.wgsl#L994). This behavior is explicitly described in spec sections 4 and 6.

A controlled probe with an executor whose first instruction was `HALT` and a neighbor beginning with `PUSH 7` executed the neighbor's first instruction in **51 of 100 slices**. This confirms that the effect applies to execution itself, not just to copy direction.

This weakens the connection between a cell's genome and its behavior. A putative metabolic specialist can spend its next slice running a neighbor's copying program. A parent cannot rely on facing its unfinished child next epoch. Persistent polarity and coordinated development must repeatedly reconstruct information that the scheduler discards.

**Suggested first experiment:** compare the four combinations of self-first/random pair order and persistent/random facing, using the same seeds and physics. Self-first does not prevent later execution from wrapping into partner code, but it is an inexpensive first isolation of the problem. Keep the existing paper mode as a separate control.

## 2. Execution progress is lost whenever a slice ends

**Confirmed semantics; a direct bottleneck for longer genomes.**

Every slice resets IP, both heads, and the stack. `SLEEP` ends the slice; waking restarts at the beginning rather than after `SLEEP`. Energy exhaustion and the step budget also discard progress. Persistent information exists in the tape, energy, bonds, orientation, sleep counter, and environmental fields, but there is no separate persistent working memory for a developmental state or copy cursor.

References: [slice reset](crates/primordia-core/src/exec.rs#L915), [sleep and halt](crates/primordia-core/src/exec.rs#L839), [persistent header fields](crates/primordia-core/src/params.rs#L11).

The minimal copy loop costs approximately **5 energy per copied byte**: CPY costs 2, and two increments plus a jump cost 3. The default wake threshold is 256. In a controlled probe with a 64-byte tape, a blank allocated partner, no copy errors, and 256 energy supplied at the beginning of each slice:

```text
After slice 1: 51 nonempty bytes copied
After slice 2: 51 nonempty bytes copied
After slice 3: 51 nonempty bytes copied
```

Each restart recopied the prefix. This is not a claim that every natural slice has exactly 256 energy; rich sites and explicit energy-saving programs can do better. It demonstrates why increasing `l_max` does not automatically make longer programs heritable.

M16's 1,024-step budget leaves almost no room beyond a bare 256-byte copy loop, before allocation, sensing, metabolism, or coordination. A program must either finish in one slice or evolve a way to store and recover its progress on the same tape that is being executed and overwritten.

**Suggested direction:** experimentally separate the effects of persistent IP/heads/stack, a persistent copy cursor, and additional slice energy. A small persistent state that does not require altering executable instructions could also support cell states. These are alternative design experiments, not a recommendation to change all three at once.

## 3. Allocation, copying, and viable reproduction are different events

**Confirmed lifecycle gap, plus a confirmed boundary defect.**

`ALLOC` creates an occupied child with a blank tape, zero energy, a lineage ID, and a birth time immediately. It prevents execution in that epoch, but there is no explicit construction state or completion condition. Subsequent slices can expose a partial program to execution and foreign writes. A blank tape can also reach `dead_after` before a slow construction finishes.

`DIVIDE` transfers a segment from the parent and splits energy. It does not duplicate the segment or ensure that either remaining tape is a complete reproducer. These are reasonable low-level primitives, but the full lifecycle must be demonstrated separately.

References: [new child](crates/primordia-core/src/exec.rs#L298), [ALLOC and DIVIDE](crates/primordia-core/src/exec.rs#L527), [death criteria](crates/primordia-core/src/phases.rs#L1116).

### Confirmed defect: division cannot transfer a suffix ending at the tape boundary

The code reduces both segment endpoints modulo `len_self + len_partner` before checking `so < eo`. A successful division requires an empty destination, so `len_partner` is zero. Consequently an endpoint equal to `len_self` becomes zero. The usual interval `[start, len_self)` cannot be transferred.

References: [CPU endpoints](crates/primordia-core/src/exec.rs#L548), [GPU endpoints](crates/primordia-gpu/src/exec.wgsl#L577).

With fixed facing, self-first order, an empty neighbor, and physics costs/errors disabled, I tested these eight-byte tapes:

```text
41 17 42 18 68 77 78 6f   # H0=2, H1=5: child length 3, parent length 5
41 17 45 18 68 77 78 6f   # H0=2, H1=8: no child, parent remains length 8
```

The existing division test covers an internal interval, so it passes while missing this boundary case. A future correction should preserve the one-past-end meaning of a slice endpoint and explicitly test the intended circular-address semantics on both CPU and GPU.

### The seeded ancestor is not an exact-length birth baseline

All 400 seed events in `runs/seed-501.jsonl` use this 15-byte genome, with mirror variants selected by the event flag:

```text
36 36 36 36 36 36 36 4d 67 40 2c 12 13 15 c4
```

`4d 67` means `PUSH 13; ALLOC`, so it allocates a **13-byte child for a 15-byte parent**. Its copy loop is unbounded; ordinary CPY writes do not grow the child's tape. Without insertion errors or another mechanism, that child cannot hold an exact copy of the parent, and continued head movement wraps into the combined address space. This does not prove the seed cannot spread. It does mean its success is not evidence of faithful autonomous reproduction.

**Suggested prerequisite:** establish a hand-authored ancestor that creates a complete child in an empty neighborhood, survives the process, and whose isolated child creates a grandchild under the same physics. Repeat across starting energy, orientation, and neighbor conditions. A population-expansion test alone is weaker.

## 4. Bonding helps neighbors overwrite one another

**Confirmed mechanism; likely conflict with stable differentiation.**

`shield_ok` grants an ordinary write unconditionally when the target is bonded to the writer. The same permission path is used by DIGEST. M16 disables the additional sleep shield. Thus bonding combines protection from outsiders and energy sharing with unrestricted genetic access by partners.

References: [write shielding](crates/primordia-core/src/exec.rs#L150), [digestion](crates/primordia-core/src/exec.rs#L575), [bond rules](crates/primordia-core/src/exec.rs#L858), [M16 settings](runs/live-m16.toml#L19).

That can aid parental construction, but it also lets a copier overwrite a useful specialist after construction. Mutual adhesion checks that both participants requested a bond; it does not establish cooperative behavior. The kin rule checks only the first 16 slots when forming an ordinary bond. It does not protect the remaining genome or recheck an existing bond after its members are overwritten. Birth bonds bypass the kin check.

There is also a tension between the kin rule and the differentiation metric: for fully filled 16-byte prefixes, ordinary bonds permit at most four mismatches, while the type metric merges prefixes within four mismatches. Genotypic differences large enough to count as distinct types can therefore obstruct direct adhesion. Birth bonds and chains of intermediate relatives make this a bias rather than an impossibility.

**Suggested experiment:** separate adhesion/energy sharing from genome-write permission, with a narrowly defined parental construction exception if needed. Measure whether differentiated programs survive their own relatives, not just whether bonded population increases.

## 5. Metabolic complexity is optional, while efficient copying pays immediately

**Mechanisms are confirmed; the strength of the evolutionary explanation needs controlled experiments.**

Sunlight and vents supply free energy independently of a cell's program. Every occupied cell absorbs from the local field automatically. CAT can add energy, but no specialized metabolic program is required for basic absorption, tape construction, or use of shared energy. Increasing chemical stocks improves an opportunity; it does not make complementary cell roles necessary or advantageous.

References: [sunlight source](crates/primordia-core/src/phases.rs#L130), [absorption and sharing](crates/primordia-core/src/phases.rs#L676), [catalysis](crates/primordia-core/src/exec.rs#L393).

Chemical species, monomer pools, and signals are site fields. Individual and body movement carry headers and tapes, not those fields. A site can have persistent chemistry, but a moving organism does not carry a private chemical inventory or selectively bounded interior. Catalysis also does not manufacture the tape monomers: the chemistry and tape-matter resource systems are separate.

References: [cell movement](crates/primordia-core/src/exec.rs#L320), [body translation](crates/primordia-core/src/phases.rs#L889), [field layout](crates/primordia-core/src/params.rs#L384), [monomer acquisition](crates/primordia-core/src/exec.rs#L193).

The present rules provide some advantages to cooperation, but a single cell can execute the same CAT operations as a specialist and neighboring copiers can receive shared energy. There is no demonstrated task for which a reproducible combination of roles outperforms the best generalist or overwriter.

**Suggested experiment:** identify one measurable benefit that genuinely requires complementary activities, then compare a manually constructed cooperating pair/body with the same number of independent copiers. Use offspring production per supplied resource as the outcome. A chemical-only condition already exists and is useful as an ablation, but removing sunlight alone would not establish an advantage for differentiation.

## 6. Damage, shading, and starvation still favor short or inactive programs

M16 lowers background damage and decay, but those remain **per byte, per epoch**, not just per birth. At approximately reference temperature, background `1e-4` plus base decay `5e-5` gives a rough rate of `1.5e-4` before radiation, hand flips, or additional temperature effects. For an unmaintained 64-nonempty-byte tape over a 1,000-epoch night, a constant-rate estimate gives 9.6 events and a no-event probability near `exp(-9.6) ≈ 0.000068`.

That is an illustration of maintenance pressure, not a measured viability probability: bytes may be repaired, some changes are harmless, decay changes the number of eligible sites, and local conditions differ. Crucially, full enclosure suppresses radiation and the light-dependent part of decay but does not remove background mutation or base decay.

References: [damage rates](crates/primordia-core/src/phases.rs#L976), [mutation targets](crates/primordia-core/src/phases.rs#L1011), [M16 damage settings](runs/live-m16.toml#L58).

The independent copy-error defaults also remain `sub=0.001`, `ins=0.0002`, `del=0.0002`. For 256 write attempts, the probability of no sampled copy-error event is approximately `(1-0.0014)^256 ≈ 0.70`, before environmental damage or stereochemical effects. Lowering background mutation did not lower these errors.

Two inherited M16 settings deserve explicit attention:

- **`shielding.light_cover = 1.0`:** four bonds in 2D give zero exposure and hence zero automatic absorption. This attenuates absorption from the free-energy field, including vent energy, not just light. Interior cells depend on sharing or catalysis. The snapshot counts above show that this is a substantial population, especially in M15.
- **`individual.dormant_idle = false`:** a nonempty cell that ends every epoch halted increments its starvation counter even if it executed that epoch. The implementation supports excluding those active epochs, but M16 does not enable it. The configured 800-epoch starvation interval is shorter than the 1,000-epoch night.

References: [exposure](crates/primordia-core/src/phases.rs#L31), [absorption](crates/primordia-core/src/phases.rs#L706), [starvation bookkeeping](crates/primordia-core/src/phases.rs#L1064), [defaults](crates/primordia-core/src/config.rs#L430).

**Suggested measurements:** successful complete copies per cell lifetime; damage between completed copies; time spent constructing offspring; and absorption, upkeep, execution spending, and CAT income by role and exposure. Genome capacity and average tape length do not answer these questions.

## 7. Bodies can persist without reproducing their organization

The body mechanism identifies connected bonds and supports collective translation. It does not provide a body birth event, a propagule lifecycle, or a criterion that offspring reconstruct the parent's roles. Cells could evolve these behaviors from the primitives, but connectivity and longevity alone do not select directly for such a lifecycle.

Large mats also become physically stationary: M16 inherits `max_body_move = 4096`, far below the observed largest components. Smaller bodies need enough agreeing movement votes and an unobstructed destination. Attachment can therefore yield protection and shared resources while preventing dispersal.

References: [body movement constraints](crates/primordia-core/src/phases.rs#L829), [bond defaults](crates/primordia-core/src/config.rs#L575).

**Suggested criterion:** an isolated body or propagule must produce a separate descendant that reconstructs the relevant roles, then repeat this for several generations. Compare role proportions and function across descendants. A body that merely expands, merges with strangers, or leaves disconnected remnants has not yet passed that test.

## 8. Current telemetry cannot establish the requested outcome

These limitations matter both for false positives and for missing a real early success:

| Measure | What it actually establishes | Consequence |
| --- | --- | --- |
| Replicator assay | One slice copying into an already occupied, equal-length blank partner in paper-0d, with physics off; at least 90% identity on original nonzero sites | Does not test allocation, energy/matter sufficiency, parent survival, or offspring fertility. It also disables spatial behavior that a real replicator may need. |
| Cell types | First 16 slots, clustered by a four-slot mismatch threshold; at least five members per type in bodies of at least ten cells | Misses different behaviors expressed by the same genome and small differentiated pairs; can count unrelated strains or damaged copies as types. |
| Roles | Sets of behavior opcodes present anywhere on the tape | Unreachable instructions count. Different programs can execute the same function; identical genomes can express different roles. |
| Executed repertoire | A slice in a physics-free sandbox with an identical partner | Measures visited instruction kinds, not successful in-world contribution. |
| Organisms | A simplified immediate-parent relationship among bonded members | Does not require the body to have reproduced or reconstruct a differentiated offspring. |
| Cluster persistence | At least 25% overlap in cells with matching site, birth time, and lineage; only clusters of 50+ cells | Movement and turnover can break tracking; splitting can pass the same historical ID to multiple descendants. It is not a body genealogy. |
| Night survival | Sunrise population divided by sunset population | Includes replacement births; it is not survival of the sunset cohort. |

References: [assay](crates/primordia-telemetry/src/analysis.rs#L175), [repertoire](crates/primordia-telemetry/src/analysis.rs#L117), [types and roles](crates/primordia-telemetry/src/lib.rs#L770), [organisms](crates/primordia-telemetry/src/lib.rs#L905), [cluster tracking](crates/primordia-telemetry/src/clusters.rs#L162), [night metric](crates/primordia-telemetry/src/lib.rs#L547).

Lineage deserves particular care: ALLOC/DIVIDE assign parent IDs before later copying finishes. CPY/WR can subsequently replace the entire genotype without changing those IDs. The spec anticipates byte tracers for that distinction, but I found only the `telemetry.tracers` configuration field/default, not an implemented tracer path. Consequently cell-container ancestry is not reliable genome ancestry in this world.

For differentiation, record actual successful in-world actions and resource contributions over time, including the source of executed instructions. Then disable a proposed role and measure whether the body's reproduction suffers. This would distinguish functional specialization from opcode diversity much more directly.

## 9. The negative/right-handed drift has a real encoding bias behind it

**Confirmed by isolated mutation experiments. This is a mutation-law asymmetry, not evidence of a reversed browser label or a geometry error.**

The reported code handedness is `(L − D) / (L + D)`, so a negative value correctly means excess D/right bytes. Only `0x20..0x3f` are chiral; even bytes are L and odd bytes are D. Ordinary background/radiation mutations flip a uniformly chosen bit of a nonzero byte. That operation does not respect the L/D symmetry of this instruction encoding.

Several common, initially achiral copying instructions have these one-bit routes into the chiral range:

| Original instruction | Chiral result reachable by one bit flip | Hand |
| --- | --- | --- |
| `12` CPY | `32` SIG_L | L |
| `13` H0INC | `33` SIG_D | D |
| `15` H1INC | `35` CHEM_D | D |
| `67` ALLOC | `27` BONDS_D | D |

There is no corresponding one-bit route from ALLOC to BONDS_L. Even the achiral minimal copier `12 13 15 c2` therefore has twice as many routes into D as into L. Another asymmetry is `20` YAW_L → `00` empty versus `21` YAW_D → `01` DUP: empty slots are excluded from subsequent background/radiation mutation, so these losses do not have equivalent return paths.

**Controlled reproduction:** 100,000 identical copies of `12 13 15 c2`, one mutation-only phase, background rate 0.1, activation/radiation/hand-flip rates zero, and energy, chemistry, matter, decay, and execution absent. The rate was deliberately elevated to expose the effect quickly.

| Seed | L bytes afterward | D bytes afterward | Code EE |
| --- | ---: | ---: | ---: |
| 1 | 1,240 | 2,382 | −0.31530 |
| 2 | 1,220 | 2,357 | −0.31786 |
| 3 | 1,181 | 2,470 | −0.35305 |

This produces a substantial rightward bias without selection, competition, spatial interactions, or differentiated cells. It is a plausible contributor when copying instructions dominate the population. It does **not** quantify its contribution in the user's live run or prove that every ordinary run must finish D-dominant: saved histories include both signs.

### Why the mirror-twin tests do not rule this out

When `st.mirrored` is true, mutation changes from `flip(v)` to `M(flip(M(v)))`. For example, the same achiral `13` changes to D byte `33` in an ordinary world, but to L byte `32` in the mirrored world. My mutation-only probes reversed all the counts exactly when this flag was enabled.

The twin therefore compares a process with its conjugated mutation law. Passing establishes consistency under that transformation; it does not establish L/D neutrality of the ordinary mutation law. Numeric RD/WR also use the reflected frame. `DESIGN.md` acknowledges encoding asymmetry, but the experimental interpretation still needs this distinction. Setting explicit chirality bias to zero, or disabling stereochemical interactions, does not remove the bit-encoding mechanism.

References: [byte classes and mirror](crates/primordia-core/src/isa.rs#L73), [EE calculation](crates/primordia-telemetry/src/analysis.rs#L495), [mutation frame and eligible bytes](crates/primordia-core/src/phases.rs#L1001), [copy-error mutation](crates/primordia-core/src/exec.rs#L235), [design discussion](DESIGN.md#L35).

**Suggested correction/experiment:** first measure mutation-only drift on the actual dominant tapes. If ordinary dynamics are intended to be neutral, define and test a mutation transition law satisfying `K(Ma, Mb) = K(a, b)`, including empty-byte transitions. A stochastic mixture of a kernel and its mirror conjugate is one possible construction; it changes the model and should be evaluated explicitly. Keep the exact twin test, but add neutrality checks with both worlds using ordinary runtime semantics. Averaging seeds alone does not remove a systematic encoding bias.

## 10. The Poisson sampler is wrong well below its documented limit

**High priority; confirmed numerical defect shared by CPU and GPU.**

The recurrence computes the equivalent of:

```rust
p = (((p as u64 * lam24 as u64) >> 24) as u32) / k;
```

It truncates the intermediate `p × lambda` to 32 bits **before** dividing by `k`. The intermediate can overflow even when the final probability would fit. Separately, the Q32 representation of `exp(-lambda)` becomes zero at sufficiently large lambda; the loop then returns one event immediately instead of sampling the distribution.

Using the current sampler, its normal cap of 64, and 100,000 deterministic Philox draws per row:

| Requested mean λ | Observed sample mean |
| ---: | ---: |
| 1 | 0.9975 |
| 3 | 2.9954 |
| 5 | 4.9937 |
| 8 | **17.7530** |
| 10 | **23.1795** |
| 20 | **41.3316** |
| 22 | **44.5420** |
| 23 | **1.0000** |
| 30 | **1.0000** |

The existing mean test stops at λ = 3. `DESIGN.md` says accuracy extends to about 20; the overflow counterexample at 8 disproves that assurance. This affects background mutation, radiation, hand-flip, and decay event counts when their per-cell means enter the bad range. Long genomes, high rates, temperature multipliers, or dormant rot can make such ranges relevant. I have not established that the inspected M15/M16 snapshots encounter them often.

References: [CPU sampler](crates/primordia-core/src/fixed.rs#L277), [existing test](crates/primordia-core/src/fixed.rs#L181), [GPU recurrence](crates/primordia-gpu/src/phys.wgsl#L1040), [call site](crates/primordia-core/src/phases.rs#L1029), [documented limitation](DESIGN.md#L174).

**Suggested fix:** retain adequate precision through division and use a sampling method that remains valid over the supported mean range. Explicitly define the behavior above the event cap. Test distributions around both the intermediate-overflow and exponential-underflow thresholds, not only CPU/GPU equality; both backends currently share the error.

## 11. `SLEEP n` blocks only n−1 subsequent execution rounds

**Confirmed timing behavior; an apparent off-by-one against the documented sleep duration.**

SLEEP sets the counter during execution. At the start of the next epoch, absorption/upkeep decrements it before the execution scheduler tests whether the cell is asleep. Consequently `PUSH 1; SLEEP` can run again every epoch. It receives the sleeping absorption/upkeep treatment during that next epoch, then executes immediately. More generally, a counter of N prevents only N−1 future execution rounds, although sleeping resource treatment lasts N rounds.

A probe running `41 6e 6f` through absorption/upkeep and execution recorded two executed instructions and a final sleep counter of one on each of three consecutive epochs. This can favor short sleep calls for resource discounts without the advertised loss of future execution opportunities. If counting the issuing epoch is intentional, the documentation and metrics should make that convention explicit; otherwise the decrement/eligibility order needs correcting.

References: [SLEEP assignment](crates/primordia-core/src/exec.rs#L839), [early decrement](crates/primordia-core/src/phases.rs#L676), [scheduler check](crates/primordia-core/src/exec.rs#L903), [GPU decrement](crates/primordia-gpu/src/phys.wgsl#L668), [documented duration](crates/primordia-core/src/config.rs#L69).

## 12. Live configuration changes leave derived state stale

**High priority when using live edits or `set` events; multiple distinct defects.**

### GPU neighbor tables are not refreshed

The GPU allocation cache is keyed only by `(cell_count, l_max, field_count)`. `nbr` is uploaded only when allocation happens. A boundary change preserves that key, so the CPU switches to the new geometry while the GPU continues to use the old neighbors.

**Reproduction:** in a 9×9 world, first verify CPU/GPU agreement, then change the x boundary from toroidal to reflecting. Place free energy at x=0. Both backends accept the edit. After one epoch the opposite edge x=8 has **CPU F=0, GPU F=899**. The GPU still transports across the removed wraparound edge.

The layout guard also accepts `[9,6] → [6,9]` because the cell count is unchanged; I confirmed that acceptance on CPU. On GPU this also leaves stale neighbors and fails to resize the row buffer. Increasing the number of reaction tables similarly does not change the allocation key even though the table buffer's required size changes. These latter buffer cases were identified in source, not separately run to a GPU validation failure.

References: [live-edit layout guard](crates/primordia-core/src/world.rs#L417), [allocation key and neighbor upload](crates/primordia-gpu/src/lib.rs#L241), [table and row allocations](crates/primordia-gpu/src/lib.rs#L283), [reconfigure](crates/primordia-gpu/src/lib.rs#L650).

### The observer retains the startup configuration forever

`Observer::spawn` captures `cfg`, `Params`, and telemetry cadence. Its message enum has no configuration update; receiving an event only logs it. Telemetry, assays, rendering, sunlight/day calculations, snapshot hashes of the configuration, and checkpoint scheduling consequently continue using the startup values after a successful `set`.

**Reproduction:** set `telemetry.hash_every` from 1 to 2 at epoch zero. The simulation accepts it, but the CLI still writes checkpoints at **every epoch 1 through 10**. Changes to day period or assay budget can likewise make the reported analysis describe different rules from the running simulation.

References: [captured configuration](crates/primordia-view/src/observer.rs#L76), [fixed cadence](crates/primordia-view/src/observer.rs#L92), [event and state handling](crates/primordia-view/src/observer.rs#L157).

### Image mutation maps are not reloaded on edits

The image is loaded in `Sim::from_state`; `apply_set` replaces parameters without reloading or validating the cached image. Switching to image mode can therefore use no map, while changing `map_image` or its amplitude can retain the previous image-derived values. This is a source-confirmed cache invalidation gap; I did not run a separate image-file reproduction.

References: [initial load](crates/primordia-core/src/world.rs#L131), [edit path](crates/primordia-core/src/world.rs#L417), [image transformation](crates/primordia-core/src/world.rs#L454), [cached map use](crates/primordia-core/src/phases.rs#L911).

**Suggested fix:** define which edits are supported, reject layout changes that cannot be safely migrated, refresh all dependent GPU buffers/tables and image data, and deliver the effective configuration to the observer alongside the corresponding state epoch.

## 13. Resume can change the rules, and resumed runs lose replay provenance

**High priority; both defects reproduced through the freshly compiled CLI.**

### Earlier configuration events are discarded

`config.toml` is written at run creation. Resume loads that initial configuration, then keeps only events whose epoch is at or after the snapshot. A `set` that already took effect before the snapshot is therefore lost. Branch also derives its base configuration from the initial file. The snapshot contains a configuration hash, not the effective configuration, and the loader's returned hash is ignored by these callers.

**Reproduction:** run ten CPU epochs with initial dissipation 0.1 and an epoch-zero event changing it to 0.5; save at epoch five. Resume that snapshot for five epochs. The uninterrupted and resumed epoch-ten hashes differ:

```text
continuous: 642ef39b486293b3
resumed:    58b697cfab9173d6
```

The parent run itself replayed all ten checkpoints correctly. The defect is in reconstructing the configuration at the resume point, not nondeterminism in the original run.

### `run --resume` does not record its starting snapshot

Unlike `branch`, resume creates a manifest with `parent: null` and no `branch.json`. Replay therefore initializes a new epoch-zero world and only applies the resumed directory's remaining events.

**Independent reproduction with no configuration changes:** seed a cell at epoch zero, run ten epochs, then resume the epoch-five snapshot into a new directory. The continuation matches the parent's epoch-ten hash exactly. Replaying the resumed directory nevertheless fails at epoch six, its first checkpoint, because the seed event and starting snapshot are absent from replay's origin information.

References: [resume loading and event filtering](crates/primordia-cli/src/main.rs#L311), [branch origin metadata](crates/primordia-cli/src/main.rs#L380), [replay initialization](crates/primordia-cli/src/main.rs#L433), [snapshot format](crates/primordia-core/src/state.rs#L457).

**Suggested fix:** reconstruct or persist the effective configuration at each checkpoint, distinguish already-applied configuration events from future interventions, and give resumed runs an explicit origin snapshot that replay uses. Add CLI-level continuation/replay tests with prior `set` and seed events; the passing core snapshot test does not exercise this orchestration.

## 14. GPU snapshots can contain a ledger from the wrong epoch

**Confirmed asynchronous bookkeeping defect.**

`take_snapshots` first collects completed GPU epoch results, advancing the live ledger, then attaches `self.st.ledger.clone()` to every returned snapshot. An older queued snapshot therefore receives the latest collected ledger rather than its own epoch's ledger.

**Reproduction:** request a snapshot after epoch one, submit through epoch five, then collect it. With ledger checks every epoch:

```text
snapshot epoch:                1
energy in snapshot arrays:     72,900
correct epoch-one energy_prev: 72,900
attached energy_prev:          47,824  (epoch five's value)
```

The underlying CPU/GPU state arrays still matched. Hash parity does not catch this because the state hash omits the ledger. Historical ledger residuals/violation counts can consequently be associated with the wrong epoch, and the serialized checkpoint baseline can disagree with its arrays. The ordinary CLI resume path resets the ledger, which mitigates that baseline mismatch for continuation but does not repair the historical record.

There is a separate persistence gap: snapshot serialization omits ledger residuals, accumulated violation count, and pending tallies. A round trip of `violations = 123` returns **0**. Any claim that the violation indicator remains sticky across resume therefore needs qualification. Direct API continuation from a checkpoint between ledger checks also needs a policy for pending tallies, rather than silently dropping them.

References: [snapshot assembly](crates/primordia-core/src/world.rs#L194), [serialized ledger subset](crates/primordia-core/src/state.rs#L482), [restore](crates/primordia-core/src/state.rs#L526).

**Suggested fix:** retain the ledger for each requested snapshot epoch, serialize the state needed for the promised continuation semantics, and compare ledger contents in delayed-snapshot tests as well as hashes.

## 15. Validation accepts configurations the implementation cannot safely execute

The following are concrete mismatches, not just requests for more defensive checks:

| Accepted configuration or load | Result | Evidence |
| --- | --- | --- |
| 8 chemistry species and 8 signal species | Generates 5 transport batches; GPU `write_static` asserts that there are at most 3 | Validation and batch count reproduced; assertion inspected |
| 17 copies of a valid catalytic reaction | CPU parameters retain 17; GPU packs only 16 reactions and 16 catalysts | Validation and parameter truncation reproduced |
| Chemistry enabled with empty species/mass/init/reaction arrays, vents enabled | Chooses `x4_species = Some(0)`, writes into the wrong field slots, then panics indexing empty `mass` | Runtime reproduction |
| `energy.absorb_alpha = 2.0` | From field F=100, absorbs 200 and wraps remaining F to **4,294,967,196** in the optimized build | Validation and runtime reproduction |
| More than 4,096 active vents | GPU uploads only the first 4,096 while CPU-side source accounting counts all vents | Source-confirmed; not stress-tested |

The chemistry/reaction limits can directly undermine experiments intended to support richer metabolic differentiation. The absorption case can manufacture enormous free-energy values. Invalid values should fail with a useful message; supported values should not silently change meaning between CPU and GPU.

References: [validation](crates/primordia-core/src/config.rs#L832), [GPU transport limit](crates/primordia-gpu/src/lib.rs#L360), [catalyst truncation](crates/primordia-core/src/wgsl.rs#L129), [reaction truncation](crates/primordia-core/src/wgsl.rs#L447), [empty-species fallback](crates/primordia-core/src/params.rs#L366), [vent injection](crates/primordia-core/src/phases.rs#L154), [absorption subtraction](crates/primordia-core/src/phases.rs#L718), [vent upload cap](crates/primordia-gpu/src/lib.rs#L717).

**Suggested fix:** validate backend limits explicitly or allocate dynamically, reject unsupported fractions/rates, handle zero chemistry species consistently, and keep source tallies aligned with injections actually performed. Check arithmetic bounds across field additions and sinks as well as the already-validated diffusion coefficients.

## 16. Region mirroring does not preserve spatial invariants

**Confirmed cell-in-wall defect; additional directional-state omission found in source.**

`mirror_region` reflects tapes, headers, and fields but leaves the site/wall array in place. In a box spanning x=3..5, with a cell at x=3 and a reflecting wall at x=5, the operation produces **an occupied cell inside the wall at x=5**. Wall creation and ordinary movement otherwise prevent this combination.

The same function reflects orientation and existing bond bits, but leaves directional bond-request bits unchanged. A pending +x adhesion request consequently remains +x after a reflection that should turn it into −x. The full-world `State::mirror` handles more state than this region operation, so the passing global twin test does not cover the intervention.

Reference: [region reflection](crates/primordia-core/src/events.rs#L230).

**Suggested fix:** define whether walls and environmental sources are part of a region reflection, enforce occupancy/wall invariants under that choice, and transform all directional state, including pending requests. Test asymmetric wall layouts and mutual-bond handshakes.

## 17. Additional telemetry bugs can misclassify reproduction and persistence

### The returned “minimal replicator” can fail the replicator assay

**Confirmed.** Minimization zeroes bytes while testing the padded tape, then strips leading/trailing zeros without retesting. Trimming changes length, addressing, and wraparound. With a 256-instruction assay budget:

```text
Original: 12 80 40 80 6f 15 13 12 13 15 c2 01 18 13 18
Identity: 1.000000
Reported minimal: 80 00 15 13 12 13 15 c2
Identity when that minimal is assayed: 0.857143  (not a replicator)
```

The ablation comment also says it preserves the original's essential bytes, but each identity test compares against the current ablated tape and excludes its zeroed positions. That is a different criterion. A minimized result should be independently revalidated after every representation-changing step, and the intended identity target should be explicit.

Reference: [minimization](crates/primordia-telemetry/src/analysis.rs#L214).

### Kin clusters disappear when their first byte is empty

**Confirmed.** `kin_labels` skips every cell whose first tape byte is zero, even if it has a valid, nonempty genome and satisfies the actual kin predicate. In a fully occupied 9×9 world containing identical `00 12 13 15 c2` tapes, pairwise kinship is true but the tracker reports **zero kin clusters**. Changing the initial byte to `12` produces one cluster. NOP-prefixed copiers are therefore systematically undercounted.

Reference: [first-byte shortcut](crates/primordia-telemetry/src/clusters.rs#L79).

### Daily births depend on delivery order rather than birth epoch

**Confirmed recorder behavior; the asynchronous caller can trigger it.** `record_births` increments one accumulator using the vector length, ignoring the epoch stored in each birth record. At sunrise, `record_state` assigns the entire accumulator to the day being closed. Delivering an epoch-three birth after the epoch-four sunrise snapshot records zero births for day one and one for day two.

The headless loop takes births before `take_snapshots` collects additional completed GPU results; at shutdown it explicitly delivers snapshots before sending the final births. Thus the required ordering is not guaranteed. Daily death estimates derived from births and population inherit the error. Also, sunrise/sunset bookkeeping only runs inside periodic metrics samples: if the metrics interval does not hit those boundaries, day records are skipped or combine multiple days.

There is a further high-volume limit: the GPU birth log retains at most 16,384 birth records per epoch. The total tally can remain larger, but lineage output and daily counts based on the truncated vector lose births. `Sim` queues a warning when truncation occurs; that does not correct the daily counts. This cap was identified in source, not exercised in a mass-birth stress test.

References: [birth accumulator](crates/primordia-telemetry/src/lib.rs#L324), [day closure](crates/primordia-telemetry/src/lib.rs#L547), [delivery order](crates/primordia-cli/src/main.rs#L255), [birth-log cap](crates/primordia-gpu/src/lib.rs#L498).

**Suggested fixes:** bucket births by their recorded epoch, close a day only once its events are complete, schedule day boundaries independently of metrics sampling, and make overflowed lineage data visibly incomplete or retain it. Remove the first-byte kin shortcut and retest every exported minimized genome.

## 18. STEP cost overrides are ignored for ordinary spatial movement

**Confirmed in CPU execution; matching GPU code.** The movement planner always uses the MOVE base cost, even when executing STEP_L or STEP_D. A valid configuration with MOVE=20 and STEP_L=STEP_D=1 charged **20** for a single STEP_L into an empty neighbor. This can invalidate comparisons intended to change the cost of chiral movement independently.

References: [CPU movement planning](crates/primordia-core/src/exec.rs#L347), [GPU base cost](crates/primordia-gpu/src/exec.wgsl#L642).

**Suggested fix:** pass the executing opcode's cost into movement planning, or reject/document unsupported overrides if all three instructions intentionally share one cost.

## 19. Run-output and input handling can silently invalidate an experiment

**Source-confirmed persistence/robustness issues; existing run directories were not altered during this review.**

- **Reusing an output directory overwrites some history and appends other history.** `RunDir::create` overwrites `config.toml`, `manifest.json`, and `hashes.csv`; telemetry opens Parquet files with `File::create`; events are appended, and existing snapshots remain. An accidental repeated command, or resume directed into the old directory, can leave artifacts from different runs mixed together. Require a fresh directory for a new run, or implement an explicit consistent append/resume policy. References: [run-directory creation](crates/primordia-cli/src/main.rs#L168), [table creation](crates/primordia-telemetry/src/table.rs#L54).
- **A nonexistent explicit events file is silently treated as no events.** `read_events` returns an empty vector for every missing path, and `run --events` uses it directly. A typo in a seed/intervention filename therefore launches an unseeded or otherwise different experiment without reporting the missing input. Distinguish an optional absent run log from a file explicitly requested by the user. Reference: [event loading](crates/primordia-core/src/events.rs#L298).
- **Malformed snapshots can panic instead of returning a load error.** `read_snapshot` indexes the header and length-prefixed blocks before checking their bounds; it also checks tape/field sizes but not header/site sizes. Truncated or inconsistent input can therefore panic during parsing or later use. Validate lengths before slicing and check every state array. Reference: [snapshot reader](crates/primordia-core/src/state.rs#L493).

## Recommended order of investigation

Before interpreting additional evolutionary runs, correct or isolate the numerical sampler, ordinary-world handedness baseline, and configuration/resume/measurement defects above. Prioritize paths actually used by an experiment: live editing and resume can silently change its rules, while the high-rate sampler and larger chemistry limits affect particular parameter regimes. Then proceed with the lifecycle experiments:

1. **Prove a complete single-cell lifecycle.** Resolve and test the DIVIDE endpoint semantics, audit the 15-byte seed/13-byte allocation mismatch, and establish isolated parent → child → grandchild reproduction. This should be a separate acceptance test from paper-mode copying.
2. **Isolate execution ownership.** Run the pair-order/facing factorial with controlled seeds. Include both a hand-authored viable ancestor and random-soup conditions so a change in discoverability is distinguishable from a change in viability.
3. **Measure and remove the copy-progress bottleneck experimentally.** Compare larger energy reserves with retained execution/copy state. Keep track of complete viable offspring rather than allocated cells.
4. **Establish stable cooperation in a small example.** Separate bond benefits from unrestricted genome writes, demonstrate an advantage for complementary roles, and test reconstruction after dispersal. Do this before scaling to giant populations.
5. **Sweep environmental pressure around that working lifecycle.** Vary damage, shading, starvation accounting, chemical availability, and copy errors separately across several seeds. Then ask whether an unseeded population discovers the capability.

This ordering separates three different questions: whether the rules permit the desired lifecycle, whether the lifecycle can beat simple copiers, and whether evolution can discover it. The current experiments largely combine all three.

## Validation performed and limits

- Compiled the current core, GPU, telemetry, viewer, and CLI source into `/tmp`, reusing existing dependency artifacts. Repository source, configuration, and saved run outputs were left unchanged; only this report was written.
- Ran **all 7 core unit tests, all 28 core integration tests, and all 8 GPU parity integration tests: 43 passed, 0 failed**. GPU tests actually ran; they were not skipped for lack of a device. They covered 0D/1D/2D/3D, both GPU execution modes, ageing, dormancy, randomized configurations, and pipelined snapshots.
- Ran isolated probes for DIVIDE endpoints, random partner execution, repeated copying, mutation-only handedness drift, Poisson sampling, sleep duration, configuration limits/overflow, region mirroring, snapshot persistence, STEP costs, minimization, kin tracking, and birth-day assignment.
- Built and ran actual temporary CLI parent/resume/replay cases, and a CPU/GPU comparison of live boundary changes and delayed snapshot ledgers. All generated runs and binaries were under `/tmp/primordia-review-cvbk5dqk`.
- Loaded the resolved M16 configuration and inspected the two existing snapshots and saved histories cited above. Reviewed corresponding CPU/GPU execution and physics paths, event/configuration handling, CLI persistence, observer delivery, and telemetry.
- Findings described as source-confirmed were not separately driven to every proposed failure mode. No long evolutionary experiment, high-volume birth/vent stress test, or full takeover experiment was run.

The proposed causal explanations are hypotheses supported by these mechanisms and observations, not results of the recommended ablations. In particular, this review does not prove that autonomous differentiated reproduction is impossible under the existing rules.
