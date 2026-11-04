#!/bin/bash
#SBATCH -J db_refine
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx&vram24gb'
#SBATCH --mem=16G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# DB improvement plan, all four phases in one job (November 2026):
#
#   gate     test_db_refine.py + test_3d_backbones.py, then a smoke pass of
#            refine_db3d.py / diagnose_db3d.py on a few entries
#   phase 0  diagnose_db3d.py: where the teacher's Sinkhorn error comes from
#            (floor, B-spline fit, footprint, standoff, reachable by refinement)
#   phase 1-3  refine_db3d.py: every path moved 0.12 along the surface normal
#            and refined on the training objective (Sinkhorn footprint
#            lambda 630 + orientation, standoff 0.12), start held; train
#            entries above p98 Sinkhorn or off the standoff band dropped
#            -> ergodic_dataset_3d_lm_ref.db (+ results/db_refine/)
#   phase 4  the same DiT-S (D384/12/6), same arguments as run_job_3d_lm.bash,
#            150 epochs on the old and on the refined DB, then
#            eval_lm3d_compare.py of both + the DiT-B lambda-630 run on the
#            holdout of the OLD DB (identical particles/starts for all)
#            -> results/dbcmp_compare/
#
# Every stage is skipped when its result exists and the trainings resume from
# their latest checkpoint, so after a time-out simply resubmit:
#
#   sbatch run_job_db_refine.bash
#
# Timing (estimates): gate ~10 min, phase 0 ~30 min, phase 1-3 ~1-2 h (not
# measured yet -- printed as ms/entry + ETA), DiT-S 0.255 s/step x 328 steps
# + overhead ~95 s/epoch -> ~4 h per training, eval ~45 min. Total ~11 h.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/3D_ergodic_learning
export MPLBACKEND=Agg

SRC=ergodic_dataset_3d_lm.db
DST=ergodic_dataset_3d_lm_ref.db
EPOCHS=150
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)  start: $(date)"

echo "=== gate: tests ==="
for T in test_db_refine.py test_3d_backbones.py; do
    python -u $T > test_out.$SLURM_JOB_ID.txt 2>&1
    RC=$?
    tail -2 test_out.$SLURM_JOB_ID.txt
    if [ $RC -ne 0 ]; then cat test_out.$SLURM_JOB_ID.txt; echo "FAIL: $T"; exit 1; fi
done
rm -f test_out.$SLURM_JOB_ID.txt

if [ ! -f results/db_diagnose/diagnose_summary.md ] || [ ! -f "$DST" ]; then
    echo "=== gate: smoke pass on real entries ==="
    rm -rf results/db_smoke && mkdir -p results/db_smoke
    srun --unbuffered python refine_db3d.py --src $SRC --dst results/db_smoke/smoke.db \
        --limit 64 --iters 20 --report_dir results/db_smoke/refine || exit 1
    srun --unbuffered python diagnose_db3d.py --db3d $SRC --n_train 16 --per_holdout 8 \
        --iters 20 --out_dir results/db_smoke/diagnose || exit 1
    rm -f results/db_smoke/smoke.db
fi

echo "=== phase 0: diagnosis  $(date) ==="
if [ -f results/db_diagnose/diagnose_summary.md ]; then
    echo "done before -- skipped"
else
    srun --unbuffered python diagnose_db3d.py --db3d $SRC --out_dir results/db_diagnose || exit 1
fi

echo "=== phase 1-3: refinement, standoff 0.12, filter  $(date) ==="
srun --unbuffered python refine_db3d.py --src $SRC --dst $DST \
    --report_dir results/db_refine || exit 1

echo "=== phase 4: DiT-S on the old and the refined DB  $(date) ==="
train() {   # $1 = run tag, $2 = DB
    local PAT="checkpoints/dbcmp_flow3d_particle_ergodic_$1_*_ep*.pt"
    local LATEST=$(ls -t $PAT 2>/dev/null | head -1)
    local START_ARG=""
    if [ -f "$LATEST" ]; then
        local DONE=$(echo "$LATEST" | sed -E 's/.*_ep0*([0-9]+)\.pt$/\1/')
        if [ "$DONE" -ge "$EPOCHS" ]; then echo "$1 finished ($LATEST)"; return 0; fi
        echo "$1: resuming from $LATEST"; START_ARG="--resume $LATEST"
    fi
    srun --unbuffered python flow_matching_runner_particles.py $START_ARG \
        --backbone dit --D 384 --dit_depth 12 --dit_heads 6 \
        --db3d "$2" --run_tag "$1" --save_model checkpoints/dbcmp.pt \
        --orientation --frame_mode lookat --rot_full \
        --start_cond --p_drop_start 0 --start_inpaint --target_cps \
        --erg_on footprint --lambda_erg 630 --erg_metric sinkhorn --sinkhorn_blur 0.05 \
        --erg_t_power 2.0 --erg_pts 128 \
        --lambda_ori 0.012 --w_cfm_rot 0.5 --w_point 0.1 --w_standoff 300 --w_angsmooth 2.0 \
        --standoff_target 0.12 --standoff_band 0.03 \
        --n_particles 512 --grid_res 64 --copies_per_char 1 --p_flip 0.0 --parts_on_gpu \
        --lr 1e-4 --mini_batch 256 --epochs $EPOCHS \
        --save_every 15 --viz_every 50 --n_holdout_viz 10 \
        --keep_checkpoints 1 --use_wandb --wandb_project flow3d-surfaces
}
train dbcmpold $SRC || exit 1
train dbcmpref $DST || exit 1

echo "=== phase 4: holdout comparison  $(date) ==="
OLD=$(ls -t checkpoints/dbcmp_*dbcmpold_*_ep$(printf %04d $EPOCHS).pt | head -1)
REF=$(ls -t checkpoints/dbcmp_*dbcmpref_*_ep$(printf %04d $EPOCHS).pt | head -1)
BIG=$(ls checkpoints/lm3d_*lm630dit_*_ep0750.pt | head -1)
echo "old: $OLD"; echo "ref: $REF"; echo "big: $BIG"
srun --unbuffered python eval_lm3d_compare.py --db3d $SRC \
    --models "DiT-S, refined DB=$REF" "DiT-S, old DB=$OLD" "DiT-B 750 ep, old DB=$BIG" \
    --per_split 100 --n_gen 4 --steps 100 --n_examples 3 --out_dir results/dbcmp_compare
echo "end: $(date)"
