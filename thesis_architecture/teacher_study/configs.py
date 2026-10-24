r"""
configs.py -- the configuration grids of the teacher study
===========================================================
Named grids; `python run_teacher_study.py --grid <name>`.

Weight scales (see `tsvec_svgd._diff_matrix`): every smoothness residual is h^2 * x^(order) with
h = 1/(T-1), so order 2 keeps the project's w = 15. Jerk residuals are ~omega times larger than
acceleration residuals and velocity residuals ~omega times smaller (omega ~ 2 pi * #loops), so the
sweeps use order-specific ranges; the pilot exists to find the useful range, not to be final.

  smoke   tiny, seconds -- exercises every solver / force / stage once
  pilot   3 shapes, ~48 configs -- find the weight ranges, prune the grid
  full    the pruned grid for all 25 holdout shapes, incl. Philipp's GUI-Adam reference (`gui_*`)
"""
from pipeline import Cfg

ACC_W = (5.0, 15.0, 50.0, 150.0)
JERK_W = (0.03, 0.1, 0.3)
VEL_W = (1000.0, 3000.0)
#: iteration budgets from the shape-A calibration (2026-10-24): LM converges after ~100-300 its,
#: TSVEC needs > 600 (1500: still improving), Adam > 2500 (5000: still improving slowly)
N_TSVEC, N_ADAM, N_LM, N_SINKHORN, N_STEIN = 2000, 5000, 300, 600, 1000


def _wp(solver, tag, order, ws, n_iters, **kw):
    return [Cfg(name=f'{tag}_o{order}_w{w:g}', solver=solver, nxi=None, w_smooth=w, smooth_order=order,
                n_iters=n_iters, **kw) for w in ws]


def pilot():
    cfgs = []
    # --- TSVEC-SVGD (paper-faithful), waypoints. w=0 needs a strong LM damping (trust region).
    cfgs.append(Cfg('tsvec_o2_w0', 'tsvec', None, 0.0, 2, damping=10.0, n_iters=N_TSVEC))
    cfgs += _wp('tsvec', 'tsvec', 2, ACC_W, N_TSVEC)
    cfgs += _wp('tsvec', 'tsvec', 3, JERK_W, N_TSVEC)
    cfgs += _wp('tsvec', 'tsvec', 1, VEL_W + (10000.0,), N_TSVEC)
    # --- Adam-SVGD (Liu & Wang / Lee et al.), waypoints
    cfgs += _wp('adam', 'adam', 2, ACC_W, N_ADAM)
    cfgs += _wp('adam', 'adam', 3, JERK_W, N_ADAM)
    cfgs += _wp('adam', 'adam', 1, (3000.0,), N_ADAM)
    # --- deterministic Levenberg-Marquardt reference (not SVGD, no swarm), waypoints
    cfgs += _wp('lm', 'lm', 2, ACC_W + (500.0,), N_LM)
    cfgs += _wp('lm', 'lm', 3, JERK_W + (0.01,), N_LM)
    cfgs += _wp('lm', 'lm', 1, VEL_W, N_LM)
    #     more Fourier modes in the solver's energy
    cfgs.append(Cfg('lm_K16_o2_w15', 'lm', None, 15.0, 2, n_iters=N_LM, K=16))
    cfgs.append(Cfg('lm_K16_o2_w50', 'lm', None, 50.0, 2, n_iters=N_LM, K=16))
    #     smoothness continuation: strong -> weak weight
    cfgs.append(Cfg('lm_cont_o2_w15', 'lm', None, 15.0, 2, n_iters=N_LM, schedule=(150.0, 50.0, 15.0)))
    cfgs.append(Cfg('lm_cont_o2_w50', 'lm', None, 50.0, 2, n_iters=N_LM, schedule=(500.0, 150.0, 50.0)))
    cfgs.append(Cfg('lm_cont_o3_w0.1', 'lm', None, 0.1, 3, n_iters=N_LM, schedule=(1.0, 0.3, 0.1)))
    # --- optimise the control points directly (the "regular" pipeline), for comparison
    cfgs.append(Cfg('tsvec_bs25_o2_w15', 'tsvec', 25, 15.0, 2, n_iters=N_TSVEC))
    cfgs.append(Cfg('adam_bs25_o2_w15', 'adam', 25, 15.0, 2, n_iters=N_ADAM))
    cfgs.append(Cfg('lm_bs25_o2_w15', 'lm', 25, 15.0, 2, n_iters=N_LM))
    # --- Sun et al. (flow matching ergodic coverage): the solver family that generated the DB
    cfgs.append(Cfg('sun_stein_pm_w0', 'sun_stein', None, 0.0, 2, n_iters=N_STEIN))
    cfgs.append(Cfg('sun_stein_jerk_w0', 'sun_stein', None, 0.0, 2, dynamics='jerk', n_iters=N_STEIN))
    cfgs += _wp('sun_sinkhorn', 'sun_sinkhorn', 2, (0.0, 3000.0, 30000.0), N_SINKHORN)
    cfgs += _wp('sun_sinkhorn', 'sun_sinkhorn', 3, (0.3, 1.0, 3.0), N_SINKHORN)
    # --- warm start from the current teacher (does a short refinement already help?)
    cfgs.append(Cfg('tsvec_teacher_o2_w15', 'tsvec', None, 15.0, 2, init='teacher', n_iters=300))
    cfgs.append(Cfg('lm_teacher_o2_w15', 'lm', None, 15.0, 2, init='teacher', n_iters=N_LM))
    cfgs.append(Cfg('lm_teacher_o3_w0.1', 'lm', None, 0.1, 3, init='teacher', n_iters=N_LM))
    return cfgs


def smoke():
    """Every solver / force / stage once, tiny iteration counts."""
    kw = dict(projections=((25, 5), (20, 5)), polish_iters=(5,))
    return [
        Cfg('s_tsvec_o2', 'tsvec', None, 15.0, 2, n_iters=4, **kw),
        Cfg('s_tsvec_o3', 'tsvec', None, 2.0, 3, n_iters=4, **kw),
        Cfg('s_tsvec_o1', 'tsvec', None, 300.0, 1, n_iters=4, **kw),
        Cfg('s_adam_o2', 'adam', None, 15.0, 2, n_iters=6, **kw),
        Cfg('s_lm_o3', 'lm', None, 2.0, 3, n_iters=4, **kw),
        Cfg('s_tsvec_bs25', 'tsvec', 25, 15.0, 2, n_iters=4, **kw),
        Cfg('s_tsvec_teacher', 'tsvec', None, 15.0, 2, init='teacher', n_iters=3, **kw),
        Cfg('s_gui_adam', 'gui_adam', None, 15.0, 2, n_iters=4, lr=8e-3, **kw),
        Cfg('s_sun_stein', 'sun_stein', None, 200.0, 2, n_iters=3, **kw),
        Cfg('s_sun_sinkhorn', 'sun_sinkhorn', None, 300.0, 2, n_iters=3, **kw),
    ]


def full():
    """Pruned after the pilot (3 shapes, 49 configs, 2026-10-24) + Philipp's reference setup `gui_*`."""
    cfgs = []
    # --- Philipp's reference: GUI Adam-SVGD, 8 particles per trajectory, K=8, lr 8e-3, raw waypoints
    cfgs.append(Cfg('gui_lr8e-3_n1000', 'gui_adam', None, 15.0, 2, n_iters=1000, lr=8e-3))
    cfgs.append(Cfg('gui_lr8e-3_n3000', 'gui_adam', None, 15.0, 2, n_iters=3000, lr=8e-3))
    cfgs.append(Cfg('gui_lr2e-3_n3000', 'gui_adam', None, 15.0, 2, n_iters=3000, lr=2e-3))
    # --- Levenberg-Marquardt (deterministic reference)
    cfgs += _wp('lm', 'lm', 2, (5.0, 15.0, 50.0), N_LM)
    cfgs.append(Cfg('lm_K16_o2_w15', 'lm', None, 15.0, 2, n_iters=N_LM, K=16))
    cfgs.append(Cfg('lm_K16_o2_w50', 'lm', None, 50.0, 2, n_iters=N_LM, K=16))
    cfgs.append(Cfg('lm_cont_o2_w15', 'lm', None, 15.0, 2, n_iters=N_LM, schedule=(150.0, 50.0, 15.0)))
    cfgs.append(Cfg('lm_K16_cont_o2_w15', 'lm', None, 15.0, 2, n_iters=N_LM, K=16, schedule=(150.0, 50.0, 15.0)))
    cfgs.append(Cfg('lm_o3_w0.01', 'lm', None, 0.01, 3, n_iters=N_LM))
    cfgs.append(Cfg('lm_o1_w3000', 'lm', None, 3000.0, 1, n_iters=N_LM))
    # --- Adam-SVGD (joint swarm, Liu & Wang / Lee et al.)
    cfgs += _wp('adam', 'adam', 2, (5.0, 15.0), N_ADAM)
    # --- TSVEC-SVGD
    cfgs += _wp('tsvec', 'tsvec', 2, (5.0, 15.0), N_TSVEC)
    # --- Sun et al. (Sinkhorn flow)
    cfgs.append(Cfg('sun_sinkhorn_o2_w3000', 'sun_sinkhorn', None, 3000.0, 2, n_iters=N_SINKHORN))
    cfgs.append(Cfg('sun_sinkhorn_o3_w3', 'sun_sinkhorn', None, 3.0, 3, n_iters=N_SINKHORN))
    return cfgs


GRIDS = {'smoke': smoke, 'pilot': pilot, 'full': full}
PILOT_SHAPES = ('A', 'korean_5', 'organic_30')
