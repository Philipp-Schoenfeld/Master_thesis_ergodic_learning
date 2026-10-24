cd "C:/Users/Philipp/Documents/Uni/Master_Thesis/Master_thesis_ergodic_learning/thesis_architecture/evaluation_full_matrix"
until grep -q "\[smooth_v2\] done:" results/v3_gpu.log && grep -q "\[smooth_v2\] done:" results/v3_sunpm.log && grep -q "\[smooth_v2\] done:" results/v3_sunjerk.log; do sleep 60; done
echo "[finish] $(date '+%F %T') all three runs done, merging"
python -u merge_smoothness_dbs.py --out_tag smoothness_comparison_v3_20261011 --src_tags smoothness_comparison_v3_gpu_20261011,smoothness_comparison_v3_sunpm_20261011,smoothness_comparison_v3_sunjerk_20261011 && \
python -u plot_smoothness_comparison.py --out_tag smoothness_comparison_v3_20261011
echo "[finish] $(date '+%F %T') done rc=$?"
