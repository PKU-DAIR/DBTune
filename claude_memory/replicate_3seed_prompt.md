# Replicate the 3-seed RF-vs-OT experiment — paste this into a new Claude Code session

Copy everything below the `---` line and paste it as the first prompt of a fresh session.

---

You are picking up a project mid-stream. Before doing anything else, read these three docs in this order:

1. `claude_memory/top_percent.md` — the source-selection methods (`ot`, `rf`, `top1`) and how the parallel infrastructure dispatches them.
2. `claude_memory/MIYABI.md` §3 (PBS) — the `pbsdsh` / `select=N` job-packing pattern.
3. `claude_memory/SETUP.md` §5 (Running the experiment) — the per-task isolated mysqld + datadir copy flow.

## What you're replicating

**Job 2098610**: a 3-seed comparison of OT-binning (`mapping_method=ottertune`, live) vs runtime RF (`mapping_method=rf`, live), across 3 workloads (`read`, `rw50`, `write`), on the Miyabi 112c125g target hardware.

- 18 tasks total: 2 methods × 3 workloads × 3 seeds.
- `workload_time = 90 s` (already set in the configs).
- 100k sources are excluded from both training pool and runtime candidate pool (already configured).
- Run all 18 in parallel on one `select=18` PBS job (`medium-c` queue routes there automatically).

## ⚠ Known issue: argmax randomness needs to be removed before submitting

The downstream `_recommend_argmax_mean` in [autotune/optimizer/bo_optimizer.py:399–426](autotune/optimizer/bo_optimizer.py#L399-L426) samples `REC_N_RANDOM=2000` random candidates per task and picks the GP-argmax-mean over that pool. The random sampler is seeded by `DBTUNE_SEED`, so different seeds get a different 2000-random pool — meaning the iter-1 config can vary across seeds **even when the source selected is identical**.

For a multi-seed run where the goal is to measure source-selection variance, this is noise we want to remove. **Implement option 2** from the prior session's analysis:

In `_recommend_argmax_mean`, just before `self.sample_random_configs(n_rand)`, reseed the ConfigSpace with a fixed seed independent of `DBTUNE_SEED`:

```python
n_rand = int(os.environ.get('REC_N_RANDOM', '2000'))
if n_rand > 0:
    rec_seed = int(os.environ.get('REC_RANDOM_SEED', '0'))
    self.config_space.seed(rec_seed)
    cands.extend(self.sample_random_configs(n_rand))
```

This way all 18 tasks draw the **same** 2000 random candidates, so the GP-argmax-mean for a given source is deterministic. `DBTUNE_SEED` still controls source-selection randomness (RF is deterministic, OT binning is deterministic given live iter-0 metrics), so multi-seed variance you observe will reflect **only** real sources of randomness (iter-0 metric noise across compute nodes, GP hyperparameter optimisation restarts) — not candidate-pool resampling.

## Steps

1. **Apply the argmax fix** (the 3-line change above to `_recommend_argmax_mean` in `autotune/optimizer/bo_optimizer.py`).
2. **Verify the existing artifacts are present** (don't retrain unless missing):
   - `autotune/optimizer/rf_models_loo_{read,rw50,write}/rf_model.joblib`
   - `scripts/config_sysbench_{ot,rf}_{read,rw50,write}.ini` should have `workload_time = 90` and `exclude_contexts` containing `64-100000-4-oltp_`
3. **Clear any prior history for the seeds you'll submit** (seeds 43/44/45 from the prior job — overwrite is fine):
   ```bash
   for t in ot_read_s43 ot_read_s44 ot_read_s45 ot_rw50_s43 ot_rw50_s44 ot_rw50_s45 \
            ot_write_s43 ot_write_s44 ot_write_s45 rf_read_s43 rf_read_s44 rf_read_s45 \
            rf_rw50_s43 rf_rw50_s44 rf_rw50_s45 rf_write_s43 rf_write_s44 rf_write_s45; do
       rm -f scripts/DBTune_history/history_${t}.json parallel/${t}/.rc
   done
   ```
4. **Submit the 18-task job:**
   ```bash
   TASKS="ot:read:43;ot:read:44;ot:read:45;ot:rw50:43;ot:rw50:44;ot:rw50:45;\
   ot:write:43;ot:write:44;ot:write:45;rf:read:43;rf:read:44;rf:read:45;\
   rf:rw50:43;rf:rw50:44;rf:rw50:45;rf:write:43;rf:write:44;rf:write:45"
   qsub -l select=18 -v "TASKS_OVERRIDE=${TASKS}" scripts/lab/par_launch.sh
   ```
5. **Monitor**: `qstat <jobid>.opbs` until "No unfinished job found", then check `parallel/<tag>/.rc` is `0` for all 18.
6. **Report** the per-task source picked, iter-1 config, tps0, tps1, and improvement, then per-workload-per-method means/std over the 3 seeds.

## Expected outcome (vs prior 2098610 run *without* the argmax fix)

Prior run showed RF tps1 std ≈ OT tps1 std on every workload, with one rw50 seed at 2614 TPS and another at 265 TPS — most of that swing came from candidate-pool resampling. With the fix, you should see RF and OT std collapse closer to single-seed noise (~iter-0 measurement variance per node), making the method comparison cleaner.

Compare your results to the prior run's per-task table (in `claude_memory/top_percent.md` §1 or `parallel/*/history*_s4{3,4,5}.json`).
