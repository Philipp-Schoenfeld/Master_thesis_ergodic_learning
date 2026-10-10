r"""
tsvec_svgd.py -- preconditioned SVGD on trajectory points, no dynamics model
==============================================================================
Direct reimplementation of the core algorithm of Li, Jin, Teng, Gong,
Chalvatzaki, "Stein Variational Ergodic Surface Coverage with SE(3)
Constraints" (TSVEC), arXiv:2603.09458 -- the SAME paper CLAUDE.md already
cites as this thesis's theoretical downstream-solver reference -- reduced
from their SE(3) via-point formulation to flat 2D positions (Philipp's
request, 2026-10-09: "bspline_svgd"/"waypoint_svgd", "basiere das auf dieser
svgd solver variante").

What carries over from the paper, read directly from arXiv:2603.09458v3
(the HTML/abstract rendering garbles several equations -- this module was
written against the downloaded PDF, not a lossy summary):

* Energy (their Eq. 20a/20d, `_residual` below): smoothness V_s (3-point
  acceleration penalty) + ergodic metric V_e (Fourier-coefficient matching),
  both already in sum-of-squares form V = 0.5*||r||^2 in the paper. Their
  V_a (surface-normal alignment) and V_f (SDF attachment) are SE(3)/3D-surface
  specific and have no counterpart on a flat 2D density target -- dropped.
  The paper itself has NO boundary term ("left for future work" per their own
  text); `w_boundary` below is this project's addition, needed because our
  domain is a fixed bounded square rather than a mesh surface. Weights: NOT
  the paper's 5.0/3.0 -- see "Energy scale" below.
* Kernel (their Eq. 22): per-timestep RBF kernels, fixed bandwidth l = 0.05
  (their value; they found an adaptive/median heuristic "less effective than
  expected" and removed it) -- NOT a single kernel over the flattened
  trajectory (that would be the older `svgd_batched.BatchedSvgdTorch`
  kernel). Averaged over t rather than summed -- see "Update-rule fixes".
* Update (their Eq. 3/18/19 restricted to a flat manifold -- SE(3)'s right
  perturbation P (+) tau*phi(P), parallel transport Ad, and the tangent-space
  gradient projection (Eq. 9/10/17) all reduce to the identity when there is
  no orientation component, i.e. plain P += tau*phi(P), phi the standard SVGD
  field): phi*(X_j) = (1/N) sum_i [k(X_i,X_j)*score(X_i) + grad_1 k(X_i,X_j)],
  score = -grad V = -J_r^T r.
* Preconditioner (their Eq. 21, "we adopt the block-diagonal preconditioning
  method proposed in [Detommaso et al. 2018]"): instead of applying phi*
  directly, solve
      (sum_i H_i * k(X_i,X')^2 + grad_1 k(X_i,X')^(x2)) alpha = phi*(X')
  for the update direction alpha, where H_i = J_r(X_i)^T J_r(X_i) is
  particle i's own Gauss-Newton Hessian and grad_1 k(X_i,X')^(x2) is the
  outer product of the (here: untransported) kernel gradient.
* 100 particles, N_t = 200 steps, step size tau = 0.1, 1000 iterations in
  their benchmark. This module's particle count and N_t are caller-supplied
  (`run_smoothness_comparison_v2.py` uses 30 particles -- Philipp's explicit
  choice: one JOINT, mutually-interacting SVGD swarm of 30, not 30 independent
  runs, so the "30-trajectory distribution" IS the particle swarm); step size
  and iteration count are also caller-supplied (2000 iterations requested).

Energy scale and update-rule fixes (2026-10-10)
-----------------------------------------------
The first version (paper weights 3.0/5.0, summed kernel, absolute damping
1e-3, no 1/N on the system) did not optimise coverage: on the round-2 run
(job 164986) the linear-init variants stayed at ergodic error ~20-26 after
2000 iterations, and the trajectories visibly were not ergodic (Philipp).
Traced on shape A (30 linear chords, raw waypoints):

  * Energy scale -- the dominant cause. p ~ exp(-V), and with w_e = 3.0 our
    Fourier ergodic energy is V ~ 0.1, so the target density is nearly flat
    and the kernel repulsion outweighs the score: SVGD diffuses the particles
    instead of concentrating them (ergodic error 23 -> 35 with only the
    update-rule fixes below). The paper's absolute weights belong to its own
    point-cloud/LBO energy and do not transfer. Using the project's own 2D
    TSVEC weights (`tsvec_2d.py`: ergodic 600, smoothness 15, boundary 30,
    i.e. the paper's ergodic weight x200) gives 23 -> 0.24 (raw, w_s=15) /
    0.44 (B-spline, w_s=15) within ~100 iterations, vs. ~3.6 for the sun
    solver after 2000 and ~3.1 for CFM alone.
  * Missing 1/N on Eq. 21's system. Detommaso et al.'s SVN, which the paper
    says it adopts, averages the preconditioner over N; phi* already has the
    1/N, so without it every step was N x too small.
  * Summed kernel. Eq. 21 weights Hessians by k^2 but the score by k, so with
    k_jj = T the step shrinks by another 1/T. Averaging over t keeps k in
    [0, 1] (`_kernel_and_grad`).
  * Absolute damping. For raw waypoints H = J^T J has rank <= #modes + active
    smoothness/boundary residuals < 2T; in the rest of the space only the
    damping regularised the system, so the repulsion term took huge,
    jagged steps there (path length 1 -> 40 with no coverage gain). Damping
    is now relative to the system's mean diagonal (`LM_DAMPING`).

With these, the smoothness "freeze" that the first version hit at w_s = 5.0
does not occur at the project's w_s = 15 either: it was a symptom of the
energy-scale imbalance, not of the smoothness term itself.

Deliberately NOT shared code with `sun_refine.py`: that module simulates a
dynamics model (PointMassLQR / JerkPenalizedLQR via a PID-tracked control
sequence) and applies the Stein gradient through an LQR solve; this module
has no dynamics at all -- particles ARE the trajectory points (or B-spline
control points of them), moved directly by the preconditioned SVGD update.

Both parameterizations share one residual/kernel/preconditioner
implementation; only how particle parameters P map to dense points X differs
(`render`): X = B @ P for B-spline control points, X = P for raw waypoints.

Jacobian: ANALYTIC, not autodiff
---------------------------------
An earlier version of this module computed J_r = d(residual)/d(P) via
`torch.func.vmap(jacrev(...))`. Correct (checked against finite differences),
but catastrophically slow: ~8.7 s/iteration at the real scale (N=30, T=128)
on CPU, i.e. ~4.9 h for one (shape, variant) cell at 2000 iterations -- not
tractable for 25 shapes. Every residual term here is in fact either LINEAR in
X (smoothness) or has a closed-form elementwise derivative (ergodic: a cosine
basis, derivative is the sine basis; boundary: a clamp, derivative is an
indicator) and X itself is linear in P (X = B@P or X = P). So the full
Jacobian is assembled analytically: d(r)/d(X) in closed form per term, then
chain-ruled through the constant d(X)/d(P) (`BK` below, `kron(B, I_2)` or the
identity) with a single batched matmul -- no autodiff graph at all. Checked
against the same finite-difference test the autodiff version used
(`test_tsvec_svgd.py::test_residual_jacobian`), bit-for-bit equivalent result,
and ~2 orders of magnitude faster (see that file's timing test).
"""
import os
import sys

import numpy as np
import torch

_arch = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _arch not in sys.path:
    sys.path.insert(0, _arch)

from ergodic_energy_torch import W_BOUNDARY, W_ERGODIC, W_SMOOTH  # noqa: E402

#: fixed kernel bandwidth, the paper's own value (not swept/adaptive, see
#: module docstring -- "except for the torus (0.1)", not applicable here).
KERNEL_LENGTH = 0.05
#: the paper's own weights for V_s / V_e (their Sec. V-A), kept for
#: reference only -- NOT the defaults. Their absolute scale is calibrated to
#: their point-cloud/LBO energy; on our Fourier ergodic metric they leave
#: V ~ 0.1, so p ~ exp(-V) is nearly flat and the kernel repulsion swamps the
#: score (see "Energy scale" in the module docstring). The defaults are the
#: project's own 2D TSVEC weights instead (`SE3_SVGD/tsvec_2d.py` ==
#: `ergodic_energy_torch.W_*`: ergodic 600, smoothness 15, boundary 30).
W_SMOOTH_PAPER = 5.0
W_ERGODIC_PAPER = 3.0
BOUNDARY_MARGIN = 0.03
#: Levenberg-Marquardt damping, RELATIVE to the mean diagonal of each
#: particle's (N-averaged) Eq. 21 system. Not in the paper. Needed because
#: H = J^T J is rank-deficient for raw waypoints (rank <= #Fourier modes +
#: #active smoothness/boundary residuals < 2T), and an absolute damping
#: constant lets the kernel-repulsion term blow up in that null space.
LM_DAMPING = 0.1
#: step size on the preconditioned direction -- the paper's own TSVEC value;
#: with the N-averaged system alpha is a proper Newton step, so this is a
#: 10%-damped Newton step.
TAU = 0.1
#: particle-parameter clip after each update, same convention as
#: `svgd_batched.py`'s CP_CLIP.
PARAM_CLIP = (0.02, 0.98)


def make_k_grid_torch(K, device='cpu', dtype=torch.float32):
    from ergodic_energy_torch import make_k_grid
    k_idx, Lambda = make_k_grid(K)
    return (torch.as_tensor(k_idx, dtype=dtype, device=device),
           torch.as_tensor(Lambda, dtype=dtype, device=device))


def _accel_matrix(T, device='cpu', dtype=torch.float32):
    """(T-2, T): accel = A @ X, the 3-point finite-difference stencil
    [1, -2, 1] along the time axis."""
    A = torch.zeros(T - 2, T, device=device, dtype=dtype)
    idx = torch.arange(T - 2)
    A[idx, idx] = 1.0
    A[idx, idx + 1] = -2.0
    A[idx, idx + 2] = 1.0
    return A


def _residual(X, k_idx, Lambda, phi_k, w_smooth, w_ergodic, w_boundary,
             margin=BOUNDARY_MARGIN, A=None):
    """Single particle, dense points: X: (T,2) -> r: (R,). Forward-only
    (used by tests/energy reporting); `TsvecSvgd.step` computes (r, J)
    together via the analytic path, not by calling this + autodiff."""
    from ergodic_energy_torch import coeffs_from_points
    T = X.shape[0]
    if A is None:
        A = _accel_matrix(T, device=X.device, dtype=X.dtype)
    accel = A @ X
    r_s = (w_smooth ** 0.5) * accel.reshape(-1)
    c = coeffs_from_points(X.unsqueeze(0), k_idx).squeeze(0)
    r_e = torch.sqrt(w_ergodic * Lambda) * (c - phi_k)
    lo = torch.clamp(margin - X, min=0.0)
    hi = torch.clamp(X - (1.0 - margin), min=0.0)
    r_b = (w_boundary ** 0.5) * torch.cat([lo.reshape(-1), hi.reshape(-1)])
    return torch.cat([r_s, r_e, r_b])


def _kernel_and_grad(Xs):
    """Xs: (N, T, 2) -> (k (N, N), dk/dX_i (N_i, N_j, T, 2)).

    Eq. 22's per-timestep RBF, AVERAGED over t instead of summed, so k is in
    [0, 1] for any trajectory length. Eq. 21 weights the Hessians by k^2 but
    the score by k, so with the paper's sum a self-similarity k_jj = T scales
    the step by 1/T (1/128 here) on top of everything else."""
    T = Xs.shape[1]
    diff = Xs.unsqueeze(1) - Xs.unsqueeze(0)                  # (N, N, T, 2)
    w = torch.exp(-(diff ** 2).sum(-1) / KERNEL_LENGTH)       # (N, N, T)
    k = w.mean(-1)
    dk = (-2.0 / (KERNEL_LENGTH * T)) * w.unsqueeze(-1) * diff
    return k, dk


class TsvecSvgd:
    """Preconditioned SVGD on trajectory points (B-spline control points or
    raw waypoints), see module docstring. One instance per (nxi, T,
    B-spline-or-not) configuration; `run` drives `n_iters` steps on a batch
    of `N` mutually-interacting particles (one JOINT swarm)."""

    def __init__(self, B, K=10, device='cpu', dtype=torch.float32,
                w_smooth=W_SMOOTH, w_ergodic=W_ERGODIC,
                w_boundary=W_BOUNDARY, damping=LM_DAMPING, margin=BOUNDARY_MARGIN):
        """`B`: (T, nxi) numpy/torch render matrix for B-spline control
        points, or None for raw waypoints (particles ARE the T dense points).
        `w_smooth=0.0` reproduces the "no smoothness force" variant."""
        self.device = torch.device(device)
        self.dtype = dtype
        self.B = None if B is None else (
            B.to(dtype=dtype, device=self.device) if torch.is_tensor(B)
            else torch.as_tensor(np.asarray(B), dtype=dtype, device=self.device))
        self.T = self.B.shape[0] if self.B is not None else None
        self.nxi = self.B.shape[1] if self.B is not None else None
        self.k_idx, self.Lambda = make_k_grid_torch(K, device=self.device, dtype=dtype)
        self.w_smooth = w_smooth
        self.w_ergodic = w_ergodic
        self.w_boundary = w_boundary
        self.damping = damping
        self.margin = margin

    def render(self, P):
        """(N, nxi_or_T, 2) -> (N, T, 2) dense points."""
        return torch.einsum('ti,nid->ntd', self.B, P) if self.B is not None else P

    def _dX_dP(self, T, D_in):
        """(T*2, D_in*2): d(X.reshape(-1))/d(P.reshape(-1)) -- constant,
        `kron(B, I_2)` (or the identity for raw waypoints). Point-major
        flattening (x0,y0,x1,y1,...) on both sides, so interleaving a 2x2
        identity block per (t,p) pair via `torch.kron` is exactly right."""
        if self.B is None:
            return torch.eye(T * 2, device=self.device, dtype=self.dtype)
        return torch.kron(self.B, torch.eye(2, device=self.device, dtype=self.dtype))

    def _residual_and_dX(self, X, phi_k):
        """X: (N, T, 2) -> (r (N,R), dr_dX (N,R,T*2)) in closed form.
        R = (T-2)*2 [smoothness] + S [ergodic] + 4*T [boundary lo+hi]."""
        N, T, _ = X.shape
        A = _accel_matrix(T, device=X.device, dtype=X.dtype)          # (T-2, T)

        # -- smoothness: r_s = sqrt(w_s) * (A @ X), LINEAR in X --
        accel = torch.einsum('ts,nsd->ntd', A, X)                     # (N, T-2, 2)
        r_s = (self.w_smooth ** 0.5) * accel.reshape(N, -1)
        dr_s_dX = (self.w_smooth ** 0.5) * torch.kron(
            A, torch.eye(2, device=X.device, dtype=X.dtype))          # ((T-2)*2, T*2)
        dr_s_dX = dr_s_dX.unsqueeze(0).expand(N, -1, -1)

        # -- ergodic: c_s = mean_t cos(pi k_s,0 x_t) * cos(pi k_s,1 y_t) -- a
        # SEPARABLE PRODUCT of per-axis cosines (`ergodic_energy_torch.
        # fourier_basis`: cos(args).prod(dim=-1)), NOT cos(pi k_s . x_t) --
        # closed-form product-rule derivative accordingly.
        k_idx, Lambda = self.k_idx, self.Lambda
        argx = np.pi * torch.einsum('s,nt->nst', k_idx[:, 0], X[..., 0])  # (N, S, T)
        argy = np.pi * torch.einsum('s,nt->nst', k_idx[:, 1], X[..., 1])
        cx, sx = torch.cos(argx), torch.sin(argx)
        cy, sy = torch.cos(argy), torch.sin(argy)
        c = (cx * cy).mean(dim=-1)                                      # (N, S)
        w_e_sqrt = torch.sqrt(self.w_ergodic * Lambda)                  # (S,)
        r_e = w_e_sqrt.unsqueeze(0) * (c - phi_k.unsqueeze(0))          # (N, S)
        # d(c_s)/d(x_t) = -(pi/T) k_s,0 sin(arg_x) cos(arg_y)
        # d(c_s)/d(y_t) = -(pi/T) k_s,1 cos(arg_x) sin(arg_y)
        dc_dx = (-np.pi / T) * k_idx[:, 0].view(1, -1, 1) * sx * cy      # (N, S, T)
        dc_dy = (-np.pi / T) * k_idx[:, 1].view(1, -1, 1) * cx * sy
        dc_dX = torch.stack([dc_dx, dc_dy], dim=-1)                      # (N, S, T, 2)
        dr_e_dX = w_e_sqrt.view(1, -1, 1, 1) * dc_dX
        dr_e_dX = dr_e_dX.reshape(N, k_idx.shape[0], T * 2)             # (N, S, T*2)

        # -- boundary: piecewise-linear clamp, diagonal Jacobian per term --
        lo_active = (self.margin - X) > 0                               # (N,T,2)
        hi_active = (X - (1.0 - self.margin)) > 0
        r_lo = (self.w_boundary ** 0.5) * torch.clamp(self.margin - X, min=0.0).reshape(N, -1)
        r_hi = (self.w_boundary ** 0.5) * torch.clamp(X - (1.0 - self.margin), min=0.0).reshape(N, -1)
        d_lo = (-(self.w_boundary ** 0.5)) * lo_active.to(X.dtype).reshape(N, -1)   # (N, T*2)
        d_hi = (self.w_boundary ** 0.5) * hi_active.to(X.dtype).reshape(N, -1)
        dr_lo_dX = torch.diag_embed(d_lo)                                # (N, T*2, T*2)
        dr_hi_dX = torch.diag_embed(d_hi)

        r = torch.cat([r_s, r_e, r_lo, r_hi], dim=1)
        dr_dX = torch.cat([dr_s_dX, dr_e_dX, dr_lo_dX, dr_hi_dX], dim=1)
        return r, dr_dX

    def step(self, P, phi_k, tau):
        """One preconditioned SVGD update. P: (N, nxi_or_T, 2) -> P' (same
        shape). Analytic Jacobian throughout -- see module docstring."""
        N = P.shape[0]
        D_in = P.shape[1]
        D = D_in * 2
        X = self.render(P)                                      # (N, T, 2)
        T = X.shape[1]

        r, dr_dX = self._residual_and_dX(X, phi_k)               # (N,R), (N,R,T*2)
        BK = self._dX_dP(T, D_in)                                 # (T*2, D)
        J = torch.matmul(dr_dX, BK.unsqueeze(0))                  # (N, R, D)
        H = torch.bmm(J.transpose(1, 2), J)                       # (N, D, D) Gauss-Newton
        score = -torch.bmm(J.transpose(1, 2), r.unsqueeze(-1)).squeeze(-1)  # (N, D)

        k, dk_dXi = _kernel_and_grad(X)                            # (N,N), (N_i,N_j,T,2)
        g = torch.matmul(dk_dXi.reshape(N, N, T * 2), BK.unsqueeze(0))   # (N_i, N_j, D)

        phi_star = (torch.einsum('ij,id->jd', k, score) + g.sum(dim=0)) / N   # (N, D)

        # Eq. 21 with Detommaso et al.'s 1/N (the paper's cited method; its
        # Eq. 21 omits it, which divides every step by N relative to phi*)
        term1 = torch.einsum('ij,iab->jab', k ** 2, H)             # (N, D, D)
        term2 = torch.bmm(g.permute(1, 2, 0), g.permute(1, 0, 2))  # (N_j, D, D)
        lhs = (term1 + term2) / N
        diag_mean = lhs.diagonal(dim1=1, dim2=2).mean(-1).view(N, 1, 1)
        eye = torch.eye(D, device=self.device, dtype=self.dtype)
        lhs = lhs + (self.damping * diag_mean + 1e-8) * eye

        alpha = torch.linalg.solve(lhs, phi_star.unsqueeze(-1)).squeeze(-1)  # (N, D)
        P_new = (P.reshape(N, D) + tau * alpha).clamp(*PARAM_CLIP).reshape(P.shape)
        return P_new, X

    def _to_tensor(self, a):
        """numpy array OR torch tensor (any device) -> tensor on self.device.
        `np.asarray` chokes on a CUDA tensor (callers may legitimately pass
        one, e.g. `target_coeffs_from_grid` run on `device='cuda'` already
        returns one), so torch tensors are moved directly instead."""
        if torch.is_tensor(a):
            return a.to(dtype=self.dtype, device=self.device)
        return torch.as_tensor(np.asarray(a), dtype=self.dtype, device=self.device)

    def run(self, init_P, phi_k, n_iters, tau=TAU, record=True):
        """init_P: (N, nxi_or_T, 2) numpy/torch. -> dict(final=P (N,.,2),
        log=(n_iters+1, N, nxi_or_T, 2) or None -- entry 0 = init, entry i =
        particle state AFTER iteration i, same convention as `sun_refine.py`."""
        P = self._to_tensor(init_P)
        phi_k = self._to_tensor(phi_k)
        log = None
        if record:
            log = torch.empty((n_iters + 1,) + P.shape, dtype=self.dtype, device=self.device)
            log[0] = P
        for it in range(n_iters):
            P, _X = self.step(P, phi_k, tau)
            if record:
                log[it + 1] = P
        return dict(final=P, log=log)

    def _residual_fn(self, phi_k):
        """Single-particle forward-only residual (no Jacobian), for energy
        reporting/tests -- `step`/`run` do not use this."""
        k_idx, Lambda = self.k_idx, self.Lambda
        w_s, w_e, w_b, margin = self.w_smooth, self.w_ergodic, self.w_boundary, self.margin
        B = self.B

        def f(P_flat):
            P = P_flat.reshape(-1, 2)
            X = (B @ P) if B is not None else P
            return _residual(X, k_idx, Lambda, phi_k, w_s, w_e, w_b, margin)
        return f
