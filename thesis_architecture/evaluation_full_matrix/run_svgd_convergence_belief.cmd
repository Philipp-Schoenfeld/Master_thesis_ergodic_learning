@echo off
rem [ARCHIVED, 3 strategies lse/ucb/eid, stopped after 375 runs - resume only if wanted]
rem SVGD convergence, belief target: the solver only sees the (ignorant) target density of the
rem knowledge state (never the ground truth); E is evaluated against the TRUE density.
rem 12 shapes x 3 knowledge states x 3 strategies x (30 CFM + 30 random walk + 30 linear) = 9720 runs,
rem 1000 SVGD iterations each, every intermediate state stored. Estimated ~5.5 h on 10 workers.
rem Self-test first; resumable: running this file again continues with the runs not stored yet.
rem Results: results\svgd_convergence_belief_all3_partial_20261008\svgd_convergence.db  (~2.1 GB)
rem Log:     results\svgd_convergence_belief_all3_partial_20261008.log
rem Plots afterwards: see plot commands at the bottom of this file.
cd /d "%~dp0"
set TAG=svgd_convergence_belief_all3_partial_20261008
set LOG=results\%TAG%.log
if not exist results mkdir results
echo [chain] %DATE% %TIME% starting self-test >> %LOG%
python -u test_svgd_convergence.py >> %LOG% 2>&1
if errorlevel 1 (
    echo [chain] %DATE% %TIME% self-test FAILED - full run not started >> %LOG%
    exit /b 1
)
echo [chain] %DATE% %TIME% self-test passed - starting full run >> %LOG%
python -u run_svgd_convergence.py --svgd_target belief --n_iters 1000 --conditions ground_truth,half_known,ten_samples --workers 10 --out_tag %TAG% >> %LOG% 2>&1
echo [chain] %DATE% %TIME% full run exited with code %errorlevel% >> %LOG%
exit /b %errorlevel%

rem ---- plots (run after the run has finished; also work on a partial DB) ----
rem python plot_svgd_convergence.py --out_tag svgd_convergence_belief_20261008
rem python plot_svgd_convergence.py --out_tag svgd_convergence_belief_20261008 --metric J --no_per_shape
rem python plot_svgd_extra_metrics.py --out_tag svgd_convergence_belief_20261008
