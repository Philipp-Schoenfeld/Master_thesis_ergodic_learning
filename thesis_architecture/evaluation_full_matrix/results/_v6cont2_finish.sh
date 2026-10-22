cd "C:/Users/Philipp/Documents/Uni/Master_Thesis/Master_thesis_ergodic_learning/thesis_architecture/evaluation_full_matrix"
until grep -q "\[continue\] done:" results/v6cont2_gpu.log && grep -q "\[continue\] done:" results/v6cont2_sunpm.log && grep -q "\[continue\] done:" results/v6cont2_sunjerk.log; do sleep 60; done
echo "[finish] $(date '+%F %T') all three continuations done, merging"
rm -rf results/smoothness_comparison_v6cont2_20261012
python -u merge_smoothness_dbs.py --out_tag smoothness_comparison_v6cont2_20261012 \
  --src_tags smoothness_comparison_v6cont2_gpu_20261012,smoothness_comparison_v6cont2_sunpm_20261012,smoothness_comparison_v6cont2_sunjerk_20261012 \
  --src_variants "cfm_bspline_svgd,linear_bspline_svgd,linear_waypoint_svgd,linear_waypoint_svgd_smooth,cfm_adam_bspline,linear_adam_bspline,linear_adam_waypoint,linear_adam_waypoint_smooth|linear_sun_pointmass|linear_sun_jerk" && \
python -u plot_smoothness_comparison.py --out_tag smoothness_comparison_v6cont2_20261012 && \
python -u compute_convergence.py --out_tag smoothness_comparison_v6cont2_20261012 && \
python -u plot_smoothness_overview.py --out_tag smoothness_comparison_v6cont2_20261012 && \
python -u qualitative_smoothness_fig.py smoothness_comparison_v6cont2_20261012 && \
python -u backfill_results_to_db.py --out_tag smoothness_comparison_v6cont2_20261012
echo "[finish] $(date '+%F %T') done rc=$?"
