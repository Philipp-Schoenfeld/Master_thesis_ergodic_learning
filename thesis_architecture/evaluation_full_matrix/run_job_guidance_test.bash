#!/bin/bash
#SBATCH -J guidance_test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 04:00:00
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=8G
#SBATCH -c 3

# Test of the GP-belief guidance of the CFM planner at inference time
# (method cfm_guided, belief_guidance.py) in the active-sensing missions:
# identical setup to run_job_active_sensing_full.bash (25 holdout shapes,
# none_known + ten_samples, UCB kappa 1.0 -> 0.05 over 5 rounds, full
# execution, TSVEC-SVGD 30 x 1500), beliefs RBF and AK (the NP has no
# closed-form covariance). Two configurations:
#   beta = 1  expected post-measurement ergodic error (ergodic + information term)
#   beta = 0  ergodic guidance towards the planning target only (ablation)
# The unguided 'cfm' sets are identical to the full run 180647 and are reused
# from results/active_sensing_20261023_tsvec_svgd via their fingerprint, so
# only cfm_guided is computed. cfm_guided uses the cfm seeds -> paired.
# Estimate: 4 cfm_guided sets per configuration, ~1-1.3 h each (CFM planning
# dominates; the guidance adds ~5 % per the local test).

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

T0=$(date +%s)
for BETA in 1 0; do
    OUT_TAG=guidance_b${BETA}_20261024
    echo "[job] $(date) beta=${BETA} -> results/${OUT_TAG}_tsvec_svgd"
    srun --unbuffered python -u run_mission_eval.py \
        --out_tag ${OUT_TAG} \
        --refiner tsvec_svgd \
        --conditions none_known,ten_samples \
        --strategies ucb \
        --methods cfm,cfm_guided \
        --beliefs rbf,ak \
        --max_rounds 5 \
        --kappa0 1.0 --kappa1 0.05 \
        --full_execution \
        --guide_weight 0.5 --guide_beta ${BETA} \
        --workers 2 \
        --parallel_sets 4 \
        --time_budget_h 1.8 || exit 1
    srun --unbuffered python -u plot_active_sensing_eval.py --out_tag ${OUT_TAG}_tsvec_svgd
    echo "[job] $(date) beta=${BETA} done after $(( $(date +%s) - T0 )) s"
done
