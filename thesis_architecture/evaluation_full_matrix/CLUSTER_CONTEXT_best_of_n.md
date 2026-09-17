# Context: Best-of-N Cluster Runs (prepared 2026-09-18)

Everything needed to run the two prepared best-of-30 jobs from the machine
that has cluster access. Written because this was prepared on a different
machine with no cluster access at all.

## Background (short version)

`evaluation_full_matrix/` is a comparison matrix of trajectory-generation
methods (CFM under several tuned acquisition strategies, a GUI-style
heuristic, a linear-waypoint baseline, lawnmower, random walk) against
holdout target shapes, evaluated against several metrics. This session
added:

1. Three new metrics wired into all `metric_bars` plots: `smoothness_energy`
   (solver's own smoothness term, resampled to a fixed point count for a
   fair cross-variant comparison), `steps_to_full_coverage` (+
   `_reached` flag; normalised arclength at which the swept target mass
   first reaches 99%, forced to 0 for `ground_truth`).
2. Two new baseline families driven by the same tuned strategies as CFM:
   `heuristic_tuned` (the GUI's own TSP+serpentine initializer, extracted
   into `init_baselines.gui_heuristic_path` so GUI and eval matrix share one
   implementation) and `linear_waypoints_tuned` (same initial path, refined
   on dense waypoints instead of B-spline control points).
3. A "best of N" mode (`run_best_of_n_matrix.py`): generate `--n_candidates`
   candidates per test case for CFM and random walk, keep the best
   candidate per metric panel (direction-aware), plus a mean/std overlay
   marker. Two selection modes, see below.
4. Row-level checkpoint/resume: every row is saved to disk the instant it's
   computed; a SIGTERM handler (`--signal=SIGTERM@120`) finalises tables/
   plots cleanly before a 24h job is killed; re-running the *same* command
   skips every already-cached `(shape, knowledge_condition, variant)`
   combination and only computes what's missing. Verified locally: a
   2-shape run followed by re-running with a 3rd shape added skipped the
   first two in 0.1s each and only computed the new one.

## The two prepared jobs

Both live in `thesis_architecture/evaluation_full_matrix/`, both use
`run_best_of_n_matrix.py`, same scope (9 tuned strategies -- the ones
already used in `results/full_run_with_optuna_ideal_v2_20260916`, i.e.
excluding `mi_optuna_gross`, which that run's own code comment flags as
coming from a non-representative, 98%-crashed hyperparameter search -- x 2
representations x 2 replan schemes x 30 candidates x 4 knowledge conditions
x 25 holdout shapes, plus the heuristic/linear-waypoint variants), differing
only in `--selection`:

| Script | `--selection` | What it means | Estimated time |
|---|---|---|---|
| `run_job_best_of_n_full.bash` | `post_svgd` | 30 fully independent candidates per decision point, SVGD-refine *every one*, keep the best finished trajectory. The literal "30x completely independent" Philipp asked for. | **~75-77h** (two independent local measurements on this exact code agree within a few hours) |
| `run_job_best_of_n_full_presvgd.bash` | `pre_svgd` | One batched network forward pass per decision point, rank the 30 raw (unrefined) candidates against the truth, SVGD-refine only the winner. | **~33-34h** (measured ~2.25x speedup over post_svgd, applied to the number above) |

**Note on "10 hours":** Philipp asked for a second version "die 10 Stunden
dauert" for the pre_svgd mode. The calibrated number for this exact scope is
~33-34h, not 10h -- flagged rather than silently relabelled. If a single
job under 24h is wanted, drop `spectral` from `--representations` in
`run_job_best_of_n_full_presvgd.bash` (particles-only pre_svgd estimate:
~16-17h) before submitting, or ask Claude to help pick a smaller scope
(fewer strategies, fewer candidates) -- both are scope decisions that
weren't made unilaterally here.

Both scripts are otherwise ready to submit as-is; no other changes needed.

## Steps on the cluster-access machine

1. **Pull this repo's latest commit** (the commit that added these files --
   check `git log --oneline -5` in `thesis_architecture/evaluation_full_matrix/`
   for a commit mentioning best-of-N / smoothness_energy / steps_to_full_coverage).

2. **Sync the updated code to the cluster** (same file list as always,
   `rsync` push is pre-approved -- see `CLAUDE.md`'s workflow section):
   ```bash
   rsync -av --progress \
     thesis_architecture/evaluation_full_matrix/*.py \
     thesis_architecture/evaluation_full_matrix/*.bash \
     stud_schonfeld@mn.ias.informatik.tu-darmstadt.de:Master_thesis/thesis_architecture/evaluation_full_matrix/
   ```

3. **Make sure both checkpoints exist on the cluster** (the particle one is
   almost certainly already there from earlier runs; the spectral one is
   new to this comparison -- neither prepared job scope has ever included
   `spectral` before, so double check):
   ```bash
   rsync -av --progress \
     transfer/netz2d_startpunkt.pt \
     stud_schonfeld@mn.ias.informatik.tu-darmstadt.de:Master_thesis/transfer/
   rsync -av --progress \
     thesis_architecture/checkpoints/cond_spectral_crossattn_ep900.pt \
     stud_schonfeld@mn.ias.informatik.tu-darmstadt.de:Master_thesis/thesis_architecture/checkpoints/
   ```

4. **Submit the long version as 3 chained jobs** (per Philipp's request --
   ~75-77h needed vs. 3x24h=72h budget: likely needs a 4th identical
   submission afterwards to finish the remainder; safe to do since it's
   fully resumable, not a restart):
   ```bash
   cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix
   JOB1=$(sbatch run_job_best_of_n_full.bash | awk '{print $4}')
   JOB2=$(sbatch --dependency=afterany:$JOB1 run_job_best_of_n_full.bash | awk '{print $4}')
   JOB3=$(sbatch --dependency=afterany:$JOB2 run_job_best_of_n_full.bash | awk '{print $4}')
   echo "long: $JOB1 -> $JOB2 -> $JOB3"
   ```

5. **Submit the short (pre_svgd) version.** At ~33-34h it needs at least 2
   chained jobs; submit a 3rd for safety margin (harmless no-op if the first
   two already finished everything -- resumable, so a job that finds
   nothing left to do just exits quickly after the row-existence checks):
   ```bash
   JOBP1=$(sbatch run_job_best_of_n_full_presvgd.bash | awk '{print $4}')
   JOBP2=$(sbatch --dependency=afterany:$JOBP1 run_job_best_of_n_full_presvgd.bash | awk '{print $4}')
   JOBP3=$(sbatch --dependency=afterany:$JOBP2 run_job_best_of_n_full_presvgd.bash | awk '{print $4}')
   echo "short: $JOBP1 -> $JOBP2 -> $JOBP3"
   ```
   Both job chains can run concurrently (they write to different
   `--out_tag` folders: `best_of_30_full_20260918` vs.
   `best_of_30_full_presvgd_20260918`) -- no need to wait for the long one
   to finish before starting the short one, as long as the cluster has
   capacity for 2 concurrent 1-GPU jobs under the `stud` partition's rules
   (check `CLAUDE.md`'s cluster section for current limits before assuming
   this).

6. **Monitor.** `squeue -u stud_schonfeld`, and each job's own
   `best_of_n_full-<jobid>.out`/`.err` in
   `thesis_architecture/evaluation_full_matrix/` (the Python script prints
   `[i/25] <shape> fertig, ...s seit Start, N Zeilen neu, M uebersprungen`
   after every shape -- a quick way to see progress without waiting for the
   whole job). No W&B logging in this script (it's a batch evaluation, not
   a training run).

7. **Pull results back** once satisfied with progress (partial pulls are
   fine at any time -- every row is saved incrementally):
   ```bash
   rsync -av --progress \
     stud_schonfeld@mn.ias.informatik.tu-darmstadt.de:Master_thesis/thesis_architecture/evaluation_full_matrix/results/best_of_30_full_20260918 \
     thesis_architecture/evaluation_full_matrix/results/
   rsync -av --progress \
     stud_schonfeld@mn.ias.informatik.tu-darmstadt.de:Master_thesis/thesis_architecture/evaluation_full_matrix/results/best_of_30_full_presvgd_20260918 \
     thesis_architecture/evaluation_full_matrix/results/
   ```

## If a job needs to be re-run partially or extended later

Both scripts are safe to re-submit with the *same* `--out_tag` at any time
(finished rows are skipped automatically). To later extend either folder
with a different scope (more strategies, dropped/added representation),
see the "Extending a folder later" section in `run_best_of_n_matrix.py`'s
own module docstring -- summarise/plots/holdout panels always rebuild from
everything cached under `raw/`, not just what one particular invocation
generated, so an extension merges instead of overwriting.

## Known caveats to carry into any write-up of these results

- `spectral` representation has never been run at this evaluation scope
  before (the existing `full_run_with_optuna_ideal_v2_20260916` only ever
  used `particles`) -- treat spectral-arm numbers as also testing
  out-of-distribution generalisation, not a clean representation
  comparison (see `spectral_planner.py`'s module docstring for the full
  reasoning: different, smaller training shape set).
- `mi_optuna_gross` is deliberately excluded from both jobs' `--strategies`
  list -- it comes from a hyperparameter search where 98% of trials
  crashed, see `variant_runner.STRATEGIES`'s comment.
- `pre_svgd` selection trades a small, unverified risk (candidate ranking
  might shift after SVGD refinement) for ~2.25x speed -- both jobs exist
  precisely so the two can be compared once both have results.
- The 99%-coverage threshold behind `steps_to_full_coverage` is strict:
  local calibration on a single shape showed most variants never reach it
  (`steps_to_full_coverage_reached` stays near 0 except for `ground_truth`,
  which reaches it by construction) -- the `_reached` panel (the reach
  *rate*) is the more informative one at this threshold, not the step count
  itself.
