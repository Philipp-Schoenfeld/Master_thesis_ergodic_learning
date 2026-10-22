cd "C:/Users/Philipp/Documents/Uni/Master_Thesis/Master_thesis_ergodic_learning/thesis_architecture/evaluation_full_matrix"
until grep -q "\[smooth_v2\] done:" results/v6_gpu.log && grep -q "\[smooth_v2\] done:" results/v6_sunpm.log && grep -q "\[smooth_v2\] done:" results/v6_sunjerk.log; do sleep 60; done
echo "[finish] $(date '+%F %T') all three runs done, merging"
python -u merge_smoothness_dbs.py --out_tag smoothness_comparison_v6_20261012 --src_tags smoothness_comparison_v6_gpu_20261012,smoothness_comparison_v6_sunpm_20261012,smoothness_comparison_v6_sunjerk_20261012 && \
python -u plot_smoothness_comparison.py --out_tag smoothness_comparison_v6_20261012 && \
python -u compute_convergence.py --out_tag smoothness_comparison_v6_20261012 && \
python -u plot_smoothness_overview.py --out_tag smoothness_comparison_v6_20261012 && \
python -u qualitative_smoothness_fig.py smoothness_comparison_v6_20261012
echo "[finish] $(date '+%F %T') done rc=$?"
