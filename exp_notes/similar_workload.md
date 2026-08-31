# Experiment: Efficient sample collection on a *similar* workload for warm-start tuning

## Scenario / motivation
We are about to launch an application. We know *roughly* what the production workload will
look like but cannot reproduce it exactly, so before go-live we run a **testing workload**
that we believe imitates the target. We have a limited compute window to collect samples on
that testing workload, and we want those samples to let us recommend a good configuration
for the *real* target workload in **few tuning iterations (no cold start)** via transfer
learning (`workload_map` / OtterTune).

## Central question
**How similar does the testing (source) workload have to be to the target for its collected
samples to still warm-start target tuning well?** I.e. as the source diverges from the
target, at what point — and along which mismatch dimensions — does the warm-start advantage
collapse to (or below) a cold start? Equivalently: characterize warm-start quality as a
function of source→target dissimilarity.

Secondary question: **how should we spend the offline sampling budget** on the testing
workload (which sampling strategy) so the collected data warm-starts best at fixed cost?

## Goal
1. Map warm-start quality vs source/target dissimilarity across several realistic, singly-
   varied mismatch dimensions (the **S0–S4** ladder below) → answer the central question.
2. Compare offline sampling strategies (sweep / LHS / random / LlamaTune) at equal budget.

## Hypotheses
- H1 (similarity threshold — central): warm-start quality decreases monotonically with
  source→target dissimilarity; there is a dimension- and magnitude-dependent point beyond
  which a mismatched source gives no benefit over cold start. Some mismatch dimensions
  (e.g. concurrency, data scale) degrade transfer faster than others (e.g. skew, modest mix
  error).
- H2: a perfectly-imitated source (S0) is the upper bound; each S1–S4 gap quantifies the
  cost of that specific imperfection in the testing workload.
- H3: non-adaptive space-filling / projection strategies (LHS, LlamaTune) collect data that
  warm-starts better than a naive one-knob-at-a-time sweep at equal budget.
- H4: some sampling strategies are more robust to source/target mismatch than others (i.e.
  strategy × similarity interaction).

## Independent variables
1. **Sampling strategy** (4 levels) — how the source configurations are chosen:
   - `sweep` — 1-knob-at-a-time, values from LlamaTune's binning (`sweep_levels`=5), all
     other knobs at default (naive-DBA baseline).
   - `lhs` — maximin Latin Hypercube over the full knob space.
   - `random` — uniform random over the full knob space.
   - `llama` — LlamaTune BO (HeSBO low-dim projection, VLDB'22).
2. **Source/target similarity (S0–S4 ladder, 5 levels)** — each level changes **exactly one**
   workload dimension of the *source*; everything else equals the target, so each isolates
   one kind of imitation error. Ordered by how realistic the mismatch is pre-launch:

   | level | mismatch dimension | source setting | target setting | scenario |
   |-------|--------------------|----------------|----------------|----------|
   | **S0** | none (oracle) | identical | — | "we imitated the target perfectly" → upper bound |
   | **S1** | client concurrency | 32 threads | 128 threads | "production traffic exceeded our load test" (client count is the hardest thing to predict pre-launch) |
   | **S2** | access skew | uniform | zipfian exp 0.7 | "load generator left on uniform defaults; real traffic is hot-spotted" |
   | **S3** | data scale | 80k rows/table | 800k rows/table (÷10) | "test env doesn't have production data volume" |
   | **S4** | read/write mix | read-heavier (≈89% read) | default rw (≈78% read) | "we overestimated the read share" |

   The dissimilarity is *ordinal* by construction (S0 < others), but we also **quantify** it
   per cell with a measured source→target distance at the default config (e.g. OtterTune
   binned internal-metric distance, or the DML embedding distance) so warm-start quality can
   be plotted against a continuous dissimilarity axis, not just the categorical level.

→ full design = 4 strategies × 5 sources = **20 cells**.

### Current collection scope
S0 and S1 are collected first (8 cells = 4 strategies × {S0,S1}); S2–S4 follow. This phases
the campaign and validates the pipeline + the most realistic mismatch (concurrency) before
spending budget on the rest.

## Dependent variables (measured in the later warm-start evaluation, not yet run)
- **Primary:** best TPS on the **target** workload after a small fixed number of warm-started
  tuning iterations (e.g. 5–10).
- **Secondary:** iterations-to-reach-X%-of-known-target-optimum; regret vs a cold-start
  baseline.
- Reported across ≥3 seeds per (strategy × source) cell to average out run-to-run variance.

## Controls / held fixed
- **Target workload** (identical for all cells): sysbench `oltp_read_write`, 150 tables ×
  800k rows, **128 threads**, zipfian skew exp **0.7**, MySQL 8.0.
- **Knob space:** `scripts/experiment/gen_knobs/mysql_perf_8.0.json` — 117 curated
  performance-relevant knobs (conservative curation, InnoDB-only; see
  [`claude_memory/KNOB_CURATION.md`](../claude_memory/KNOB_CURATION.md)). Same file for
  collection and for the later target tuning (exact-config reconstruction).
- **Budget:** fixed **N = 451 evaluations per cell** (the sweep's natural size at 5 levels);
  LHS/random/llama matched to the same N. No adaptive stopping — required for a fair
  strategy comparison.
- **Every strategy's schedule starts with the default config** (iteration 0) — required for
  OtterTune's exact-config matching at the start of target tuning.
- Same hardware (Miyabi-C node class), MySQL build, warmup (60 s) / measurement (120 s)
  window, `rand_seed = 42`, offline knob application (my.cnf + restart per iteration),
  internal-metric collection via Prometheus (needed by `workload_map`).

## Method
1. **Collect** (this phase): run each of the 8 cells through DBTune's `Sampler` /`LlamaTune`
   optimizers, producing `scripts/DBTune_history/history_miyabic_150-800000-<128|32>-oltp_read_write-0.7-<strategy>.json`.
2. **Build source pools:** per-cell `data_repo` dirs (one source dataset each).
3. **Warm-start eval** (later): tune the **target** with `transfer_framework = workload_map`
   (OtterTune mapping) seeded from each cell's data; ≥3 seeds; record the dependent vars.
   Budget sensitivity comes free by replaying truncated history prefixes (N = 50/100/200/…).

## Scope notes
The 20-cell S0–S4 × strategy design is the full experiment. S0–S1 are collected first
(phasing, see "Current collection scope"); S2–S4 reuse the same machinery — each is one
changed `[database]` setting in the cell ini (`sysbench_rand_type`, `sysbench_table_size`,
sysbench mix args) plus a matching prepared dataset for S3.

## Results (final, 2026-07-06 — all 20 cells collected and transplant-evaluated)

Reference points: target default ≈ 120–240 tps (single-measurement noise is large);
target best-known ≈ **19,940 tps**. "kept %" = best transplanted tps / 19,940.
Raw numbers: `scripts/report_eval.py` + `DBTune_history/history_eval_replay_*.json`.

### Headline: best transplant per cell (% of target best)

| source (mismatch) | sweep | lhs | random | llama |
|---|---|---|---|---|
| **S0** none (oracle) | 4% | **97%** | 84% | 80% |
| **S1** concurrency ÷4 | 4% | 91% | **101%** | 61%¹ |
| **S2** skew (uniform → zipf 0.7) | — | 62% | 79% | **96%** |
| **S3** data scale ÷10 | — | 51% | 55% | **88%** |
| **S4** mix (≈89% vs 78% read) | — | **99%** | 84% | 97% |

¹ S1-llama's pool is from collection round 1 (38 usable samples of 451 attempts, before
the chunked-datadir fix) — underpowered; every other llama pool has 100 usable samples.

### Full transplant detail (top-3 source configs, each re-evaluated once on the target)

| cell | #1 src→tgt | #2 src→tgt | #3 src→tgt | best kept |
|------|-----------|-----------|-----------|-----------|
| S0 sweep | 3392→854 | 1327→631 | 709→381 | 854 (4%) |
| S0 lhs | 19500→17231 | 19378→**19283** | 17902→FAIL | 19283 (97%) |
| S0 random | 19940→9605 | 17371→**16744** | 16087→14055 | 16744 (84%) |
| S0 llama | 17446→**15969** | 17173→13552 | 16723→15560 | 15969 (80%) |
| S1 sweep | 2410→812 | 2085→686 | 2081→698 | 812 (4%) |
| S1 lhs | 15500→15955 | 13894→**18236** | 13737→16557 | 18236 (91%) |
| S1 random | 15781→**20180** | 14208→19514 | 13620→17066 | 20180 (101%) |
| S1 llama | 12963→**12109** | 12266→11256 | 9845→7800 | 12109 (61%) |
| S2 random | 15350→11237 | 9364→**15786** | 9033→11316 | 15786 (79%) |
| S2 lhs | 8078→12071 | 8051→10022 | 7957→**12315** | 12315 (62%) |
| S2 llama | 15959→**19081** | 13111→9402 | 12785→17499 | 19081 (96%) |
| S3 random | 25664→7334 | 25291→**11062** | 24747→6059 | 11062 (55%) |
| S3 lhs | 26984→9488 | 26023→3895 | 25519→**10175** | 10175 (51%) |
| S3 llama | 32095→12845 | 31665→15149 | 31602→**17598** | 17598 (88%) |
| S4 random | 11962→**16772** | 10532→14704 | 9041→10050 | 16772 (84%) |
| S4 lhs | 12704→**19772** | 10018→13912 | 9422→11890 | 19772 (99%) |
| S4 llama | 15415→5632 | 14704→**19294** | 14436→17684 | 19294 (97%) |

(Bold = each cell's best transplant. Note how often it is NOT the source's #1 config.)

### Collection yield: usable samples per cell (and why round 2 was so much healthier)

The two collection rounds ran under different protocols, and the yields differ sharply:

| round | protocol | cell | lhs | random | llama | sweep |
|-------|----------|------|-----|--------|-------|-------|
| 1 | fixed 451 attempts, one datadir per 48 h wave | S0 | 118/451 (26%) | 133/451 (29%) | **80/451 (18%)** | 449/451 |
| 1 | 〃 | S1 | 112/451 (25%) | 136/451 (30%) | **38/451 (8%)** | 451/451 |
| 2 | stop at 100 successes (cap 300), fresh datadir every 4 h chunk | S2 | 100/122 (82%) | 100/134 (75%) | 100/159 (63%) | — |
| 2 | 〃 | S3 | 100/101 (99%) | 100/102 (98%) | 100/119 (84%) | — |
| 2 | 〃 | S4 | 100/171 (58%) | 100/108 (93%) | 100/225 (44%) | — |

(x/y = usable SUCCESS samples / attempts. Sweep barely fails because it perturbs one knob
at a time; it was dropped in round 2 after F2 showed its data is worthless as payload.)

**Why round 1 failed so often — and round 2 didn't.** Most round-1 failures were not
caused by the sampled configs. Certain configs leave the datadir in a sticky bad state
(e.g. huge redo logs forcing a crash-recovery longer than the 600 s connection wait on
every subsequent start), after which *every* following attempt fails regardless of its
own merits — S0-lhs had 64 successes in its first 75 attempts, then ~240 consecutive
failures that consumed the rest of the 48 h wave. Round 2's runner (`collect2_node_run.sh`)
replaces the datadir with a fresh snapshot copy every 4 h chunk, so a poison event costs
at most one chunk tail. Same config generators, same knob file: overall success went from
~30% to 44–99%. The residual round-2 failures (S4-llama's 44% is the worst) are the true
config-caused rate — genuinely startup-breaking knob values, tolerated deliberately since
failures are cheap and only good configs matter for transplant.

**Correction (2026-08-29) — one bogus FAILED row per chunk restart.** `DBEnv.step_GP` ran
`ensure_default_config(strong=True)` on the first step of every *process*, assuming it is the
default configuration; on a resumed chunk that step is an arbitrary suggestion, the check
raised, and the never-evaluated config was stored as FAILED. Every round-2 cell shows exactly
(chunks − 1) such rows (S2: 3/4/3, S3: 1/2/1, S4: 4/7/2 for lhs/llama/random; S4-llama's 125
failures include 7 of these). Pools are unaffected (FAILED rows are filtered) and the
"config-caused" failure rates above are overstated by ≤ 6 %; LlamaTune's inner BO received one
corrupt "failed" label per chunk. Fixed in `dbenv.py` (`_knobs_are_default` guard) on
2026-08-29; runs before that date carry the artifact.

**Why LlamaTune improved the most in S2–S4 (an effect of the fix, not of LlamaTune).**
LlamaTune was hit hardest by round 1's streaks, for a strategy-specific reason: lhs and
random play back schedules fixed in advance, so environmental failures only cost them
samples — but LlamaTune's inner BO *trains on the outcomes*. During a streak, every
config is recorded as FAILED with a worst-case objective even though the config was not
at fault, so the BO learns "this whole region is terrible" from environmental noise and
steers away from it. That is why the round-1 llama pools are both the thinnest (S1: 38
usable) and the weakest (S1-llama 61% transplant, the outlier in the headline matrix).
In round 2 the chunked runner fed the BO mostly honest labels, and llama promptly
produced the best and most mismatch-robust configs of the study (S2/S3/S4: 96/88/97%
best-kept). Interpretation caveat that follows: **round-1 llama cells (S0, S1) understate
LlamaTune** — cross-round comparisons of llama should account for the label-corruption
handicap, and the S1-llama cell should not be used as evidence against BO collection.

### Findings

**F1 — concurrency mismatch is ~free.** A source at 4× lower client count transfers
losslessly: S1-random hit 20,180 tps at 128 threads — above anything found on the target
itself. Good configs are concurrency-robust across this gap.

**F2 — the sweep is useless as warm-start payload.** One-knob-at-a-time never finds
strong configs (≤3.4k source-side) and even those do not reproduce (→ ~700–850): good
configs need jointly-set knobs.

**F3 — winner's curse + rank inversion → always transplant top-k, never top-1.** The
source's #1 config is frequently not the best transplant (12 of 17 non-sweep cells),
either from source-side measurement noise (S0-random 19940→9605) or genuine
mismatch-induced re-ranking (S2-llama #1 19081 vs #2 9402; S4-llama #1 5632 vs #2 19294).

**F4 — surrogate-level transfer added nothing.** Warm-started BO (SMAC + OtterTune
mapping, 10 iters, 3 seeds) means: warm 5.4–7.7k vs cold 7.3k — no separation, seed
variance dominates (seed 43 reaches ~14k in nearly every arm including cold). One
transplanted evaluation beats the mean outcome of 10 BO iterations ~3×. The effective
warm-start mechanism is config replay, not the surrogate.

**F5 — mismatch-dimension ranking (the central question).** For space-filling pools:
`S0 ≈ S1 (concurrency) ≈ S4 (mix) > S2 (skew, −20–40%) ≫ S3 (scale, −50%)`.
Data volume is the one property a testing workload must reproduce: configs tuned on a
working set that fits in RAM mis-size memory/IO knobs for 10× data, and scale mismatch
also scrambles the config ranking, so more samples cannot fix it. Concurrency and mix —
the hardest things to predict pre-launch — are nearly free; skew is second-order.

**F6 — strategy × similarity interaction (H4 confirmed).** LlamaTune(BO)-collected
configs transplant robustly under EVERY mismatch (S2/S3/S4: 96/88/97%), while
space-filling collapses exactly where the mismatch bites (S3: 51–55%, S2: 62–79%).
BO concentrates on high-performing regions whose configs are good for robust reasons;
space-filling "bests" are more often lucky extremes that do not generalize. Collection
strategy matters most precisely when the testing workload is imperfect.

### Bottom-line recipe for the pre-launch scenario

1. Load **production-scale data** in the test environment (the one non-negotiable);
   approximate concurrency/mix as convenient, keep skew roughly realistic.
2. Collect ~**100 successful samples with LlamaTune** (≈ half a node-day with the
   chunked runner; success rates 59–98% after the sticky-datadir fix vs ~30% before).
3. Warm-start production tuning by **replaying the source's top-3 configs**
   (1–3 evaluations ≈ 88–101% of the known optimum), then refine with BO from those
   incumbents. Skip OtterTune-mapping surrogate transfer at this horizon (F4).

## Results — transfer-method comparison under a 1 h deadline (2026-07-07)

Arms (all SMAC downstream, target workload, `max_runs` unbinding, **hard `timeout 3600`**,
3 seeds, fresh session each, per-iteration `update_time` stamps persisted in history):
**ottertune** = workload_map metric-binning selection; **rgpe** = RGPE ensemble
(ResTune-style meta-learner); **opadviser** = `space_transfer=True` (compact space +
initial design seeded with source-best configs); **cold** = no transfer. Shared source
repo `pool_ALL`: 15 contexts (S0–S4 × lhs/random/llama, SUCCESS-only, 1,517 rows; sweep
excluded). Scripts: `transfer_{gen_task,node_run,launch,smoke}` in `scripts/lab/`.

| arm | best@60min per seed (42/43/44) | mean | mechanism observed |
|-----|-------------------------------|------|--------------------|
| **opadviser** | 19508 / 19413 / 19110 | **19,344** | iteration 1 already ≈19.4k on every seed — init replays pool-best configs (built-in transplant) |
| ottertune | 1933 / 13716 / 3096 | 6,249 | picked S2/S4-llama sources, never the S0 oracle; surrogate route unexploited |
| rgpe | 1925 / 4266 / 3547 | 3,246 | ≈ cold |
| cold | 3183 / 5146 / 1568 | 3,299 | baseline |

- ~9–12 iterations/hour in every arm; optimizer overhead negligible at this pool size
  (ottertune 2 s, rgpe 5 s, opadviser 17 s per iteration vs ~350 s evaluations) — the
  budget went almost entirely to evaluations, so arms differed only in *which* configs
  they evaluated.
- opadviser's five init evaluations were pool-best configs (~19.4k, 9.8k, 18.3k, 8.7k,
  19.4k) — top-k source replay by design; that alone locked in ~97% of the known optimum
  within ~35 min on all three seeds.
- ottertune's metric binning failed the oracle test: with four identical-workload S0
  pools available it matched S2-uniform-llama / S4-ps30-llama instead. (Per F4, even a
  perfect pick would not have been exploited by the EI route.)

**F7 — under a wall-clock budget, the only transfer mechanism that mattered was
evaluating source-best configs directly** (OpAdviser's init ≙ our replay protocol:
19.3k vs 3.3k cold, ~6×). Surrogate-level transfer (OtterTune mapping, RGPE) stayed
indistinguishable from cold start even with a multi-source repository and real source
selection in play.

**F7a — provenance of OpAdviser's replay init (sub-agent audit of paper + official repo).**
The *published* OpAdviser method (PVLDB 17(3) p539) contains NO config-replay warm start:
its two components are compact-space construction from similar tasks (§5) and optimizer
recommendation (§6); the only "warm" language concerns "warm[ing] up the space
recommendations" (§5.3). The replay behavior lives only in the **released implementation**:
`get_max_distence_best()` collects each source task's incumbent (best) config and
`iterate()` evaluates them directly for the first `init_num` iterations, bypassing the
optimizer (OpAdviser repo `pipleline.py:118-119, :264-267, :429-431`; identical code in
this repo, inherited — introduced upstream in commit `8ca5a99` "space transfer:use best
source config to init" by the OpAdviser first author, Feb 2023; their shipped config
defaults to `initial_runs=10`, i.e. default + 9 source-best replays). During the replayed
iterations the compact space has no effect on which configs run, so in our ~10-iteration
1 h sessions **essentially the entire measured "OpAdviser" win is the undocumented
warm-start, not the paper's space-construction mechanism**. Note also: the paper's own
"w/o-Space" ablation removes the warm start *together with* the space constructor (both
hang off `space_transfer`), so the paper's space-construction gains conflate the two.
→ In any writeup, label the arm "OpAdviser (released implementation)" and attribute the
win to source-config replay explicitly.

**F7b — ablation: OpAdviser without the replay (2026-07-08).** New arm `opadviser_ns`
(`space_transfer=True, space_transfer_replay=False, initial_runs=1` — identical to cold
except for the paper's compact-space mechanism; switch added in `pipleline.py`):
best@60min = 6700 / **21640** / 6800 (mean 11,713; median 6,800) vs cold 3,299 and full
OpAdviser 19,344. Reading: **the paper's space construction alone does help (~2× cold
at the median)** — constraining proposals to the sources' promising regions lifts every
seed above its cold counterpart, and seed 43 landed **21,640 tps at iteration 8, the
highest configuration ever observed on this target** (previous best 20,180). But it is
high-variance and slow (typical seed ~6.7–6.8k after an hour), whereas replay delivers a
deterministic ~19.3k floor by minute 7. Combined (full OpAdviser) = replay's floor;
the space+BO tail never exceeded the replayed incumbents within the hour, though s43's
21.6k here suggests it eventually could with a longer budget. Mechanism split at 1 h:
replay ≈ guaranteed 97% of optimum; space-only ≈ 34% median with a long upside tail.

## Results — reproduction run of the 1 h-deadline comparison (run 2, 2026-08-27/28)

Re-ran four arms (ottertune / rgpe / opadviser_ns / opadviser; cold omitted) with the recipe
in [`claude_memory/REPRO_1H_DEADLINE.md`](../claude_memory/REPRO_1H_DEADLINE.md): same scripts,
byte-identical inis, same `pool_ALL`, snapshot and seeds. PBS job 3088190, nodes mc057–mc060,
22:25–01:40 JST, ~13 node-h. The July histories/logs/inis are preserved under
`*/eval2_run1_2026-07/`; `scripts/report_eval2.py --hist-dir` reads them.

| arm | best@60min per seed (42/43/44) | mean | median | July mean |
|-----|-------------------------------|------|--------|-----------|
| **opadviser** | 14840 / 16218 / 15649 | **15,569** | 15,649 | 19,344 |
| ottertune | 1676 / 9523 / 1596 | 4,265 | 1,676 | 6,249 |
| rgpe | 1686 / 7633 / 1587 | 3,635 | 1,686 | 3,246 |
| opadviser_ns | 1631 / 1693 / 1844 | 1,723 | 1,693 | 11,713 |

**F7c — the failure reproduces; two caveats from the critique are now observed directly.**

- Same shape as July: the three surrogate-route arms end the hour at a 1.6–1.7k *median*,
  ~9× below the replay arm, which again exceeded 10k on its first replayed configuration
  (iteration 1, ~minute 7) on every seed. The seed still decides the outcome: OT
  1.6k / 9.5k / 1.6k, RGPE 1.6k / 7.6k / 1.6k. OT again selected S4-lhs (`ps30-0.7-lhs`)
  over the S0 pools (now ranked #2/#3, distance 18.2 vs 17.4) — oracle test failed again.
- **Shared seeds: the first 3 iterations are byte-identical across ottertune / rgpe /
  opadviser_ns in every seed.** The seed-43 lift of OT and RGPE comes from the same shared
  candidate stream, not from either transfer method. The arms are not independent
  samples — never count the design as 3 seeds × 4 arms = 12 observations.
- **Cluster-wide throughput was 22–60 % lower than on Jul 7 for identical configurations.**
  Default config: 138–245 → 76–112 tps on all four nodes. OpAdviser's replayed source-best
  configs — iterations 1–4 identical to July's on all three seeds — 19.5k → 12.1k / 10.8k /
  14.8k (one config, three measurements). This is why every arm, including the replay
  floor (19.3k → 15.6k), sits below July. **Absolute tps is not comparable across
  runs/days; compare arms within a run, or normalise by a within-run reference config.**
- July's opadviser_ns 21,640 outlier did not recur: the space-only ablation stayed in the
  1.6–1.8k band on all seeds (its July mean of 11.7k was one lucky seed).
- FAILED iterations: 1 per arm (July 0–5); 7–10 iterations per session (July 7–12).

## Results — same comparison with the clean-shutdown harness (eval4, 2026-08-29 15:29–18:35)

Same four arms, seeds, knobs and `pool_ALL`, but between trials MySQL is now shut down cleanly
when the estimated dirty-page flush is cheap (default since commit `6a88e30`; details in
[`claude_memory/REPRO_1H_DEADLINE.md`](../claude_memory/REPRO_1H_DEADLINE.md) §10–11). Job
3208151, nodes under head mc103, ~12.4 node-h.

| arm | best@60min per seed (42/43/44) | mean | median | iterations/session | run-2 mean (kill -9) |
|-----|-------------------------------|------|--------|--------------------|----------------------|
| **opadviser** | 20004 / 20464 / 20065 | **20,178** | 20,065 | 13 / 15 / 13 | 15,569 |
| rgpe | 5629 / 12358 / 3289 | 7,092 | 5,629 | 14 / 13 / 11 | 3,635 |
| opadviser_ns | 1723 / 3943 / 13942 | 6,536 | 3,943 | 13 / 14 / 12 | 1,723 |
| ottertune | 2264 / 11157 / 3451 | 5,624 | 3,451 | 13 / 13 / 11 | 4,265 |

**F7d — 1.6× more evaluations per hour does not change the picture.** Dead time per iteration
fell from 140 s median / 196 s mean (run 2, kill -9) to **43 s median / 92 s mean** (158 clean
shutdowns, median 15 s, p90 205 s, max 405 s; 4 kill -9 decisions by the rule; 0 budget
fallbacks), giving 11–15 evaluations per session instead of 7–12. The surrogate-route arms still
end the hour at a 3.5–5.6k median with seed-dominated spread (each arm's best seed is a
different one: OT s43, RGPE s43, space-only s44), 3–6× below the replay arm, which again exceeded
20k on every seed by minute 14–21 on its first replayed configuration. The first 3 / 6 / 3
iterations are byte-identical across the three surrogate arms (shared seed). Absolute numbers are
back at July level (replay floor 20.2k vs 19.3k July, 15.6k on Aug 27) — the cross-day offset is
real and unrelated to the harness change (F7c). Failures: 2 startup-breaking configs in 158
evaluations; no tps ≤ 0 rows.

## Status (2026-06-28)
- Implementation complete; offline + smoke + pilot all green. First phase (S0+S1) ran:
  **all 8 cells reached 451/451** mechanically. S2–S4 not yet collected.
- **Open problem found before the eval phase:** in offline application mode every config is
  written to my.cnf and mysqld is restarted, so a single startup-breaking value fails the
  whole config. The sweep cells are clean (0–2 FAILED) because they perturb one knob at a
  time, but **lhs/random/llama failed 70–91%** of trials (`mysql FAILED to start`) because
  they set all 117 knobs at once and oversized values (e.g. `read_buffer_size`≈1.4 GB,
  `innodb_log_buffer_size`≈3.5 GB) break startup. Usable (successful) configs per cell:
  sweep ~449–451, lhs ~112–118, random ~133–136, llama ~38–80. FAILED trials carry empty
  internal metrics, so they cannot feed OtterTune mapping → the strategy comparison is
  currently confounded by very different yields.
- **Next:** clamp the startup-breaking knob ranges (diagnose with `mysqld --validate-config`
  on saved FAILED configs), keep offline mode, re-run lhs/random/llama (sweep stays). Then
  proceed to the warm-start evaluation. See [`claude_memory/COLLECTION.md`](../claude_memory/COLLECTION.md)
  for the operational details (scripts, how to run, resume).
