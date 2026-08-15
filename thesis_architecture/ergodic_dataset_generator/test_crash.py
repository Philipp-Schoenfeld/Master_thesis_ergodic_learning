from shape_library import get_shape, make_pdf_and_score
from ergodic_solver import run_ergodic_coverage
import numpy as np

name = "rand_poly_139"
print(f"Testing {name}")
shape_def = get_shape(name)
pdf, score_fn = make_pdf_and_score(shape_def)
rng = np.random.default_rng(abs(hash(name)) % (2**31))
x0 = tuple(rng.uniform(0.05, 0.25, size=2))

kwargs = dict(dt=0.05, tsteps=200, num_iters=600, step_size=0.01, h=0.01, score_scale=1.0)
try:
    traj_xy, init_traj = run_ergodic_coverage(
        score_fn, x0=x0, shape_def=shape_def, verbose=True, **kwargs
    )
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
