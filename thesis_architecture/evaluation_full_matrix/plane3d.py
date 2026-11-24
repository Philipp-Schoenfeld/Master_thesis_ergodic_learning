r"""
plane3d.py
==========
World, ergodic metric and SVGD solvers of the 3D exploration comparison
(`run_exploration_comparison_3d.py`).

The 2D heldout densities are placed on a planar patch inside the unit cube, and everything that
plans, measures or scores works on the 3D coordinates (x, y, z) of that patch: the networks are the
3D-trained SE(3) planners, the ergodic metric is the Fourier metric of the unit CUBE (K^3 modes),
the TSVEC-SVGD moves 3D control points and keeps them on the plane with the SDF attachment term
V_f of Li et al. (arXiv:2603.09458) -- nothing is computed in 2D and lifted afterwards. The plane is
only used as a chart for drawing, so the figures look like the 2D ones.

Plane patch
    Same placement as the plane surfaces of the 3D dataset (`3D_ergodic_learning/surfaces.py`:
    unit square fitted into [MARGIN, 1 - MARGIN]^3, MARGIN = 0.14, centre (0.5, 0.5, 0.5)), i.e.
    the geometry the 3D networks were trained on. Orientations (`PLANE_NORMALS`):
      flat       normal (0, 0, 1), the patch z = 0.5          (`ebene_flach`)
      tilted     normal (0, sin 50deg, cos 50deg)             (`ebene_gekippt`)
      diagonal   normal (1, 1, 1) / sqrt 3                     (`ebene_diagonal`)
    Chart: world(u, v) = o + s (u e1 + v e2), (u, v) in [0, 1]^2, s = 1 - 2 MARGIN = 0.72,
    (e1, e2, n) = the orthonormal frame of `surfaces._frame(n)` (flat: e1 = x, e2 = y).

Units
    "plane units" = the 2D units of the 2D benchmark (the patch is the unit square there). Every
    length-type setting and metric of the 3D benchmark (sensor radius, GP lengthscale, path
    budget, coverage radii, path length, smoothness, SVGD kernel bandwidth and boundary margin)
    is given in plane units and converted with `s`, so on the flat plane the geometry is exactly
    the 2D one scaled by s. Ergodic errors are NOT comparable to the 2D numbers (cube basis with
    K^3 modes and the 3D weights Lambda_k = (1 + |k|^2)^-2).
"""
import math

import numpy as np
import torch

MARGIN = 0.14                       # surfaces.MARGIN of the 3D dataset
PLANE_SCALE = 1.0 - 2.0 * MARGIN    # side length of the patch in world units
_t50 = math.radians(50.0)
PLANE_NORMALS = {'flat': (0.0, 0.0, 1.0),
                 'tilted': (0.0, math.sin(_t50), math.cos(_t50)),
                 'diagonal': (1.0 / math.sqrt(3.0),) * 3}

# energy weights of the 2D TSVEC (`ergodic_energy_torch.W_*`, `tsvec_2d.py`), kept for the 3D port
W_ERGODIC = 600.0
W_SMOOTH = 15.0
W_BOUNDARY = 30.0
#: weight of the plane attachment residual sqrt(w) * SDF / s (Li et al.'s V_f for a plane)
W_PLANE = 1000.0
K_DEFAULT = 10
BOUNDARY_MARGIN = 0.03              # plane units, as the 2D TSVEC
PARAM_CLIP = (0.02, 0.98)           # control points, in plane coordinates (u, v)
KERNEL_LENGTH = 0.05                # TSVEC kernel bandwidth, squared plane units
LM_DAMPING = 0.1
TAU = 0.1
ADAM_LR, ADAM_BETA = 2e-3, 100.0    # run_smoothness_comparison_v2.ADAM_*
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8


def _frame(w):
    """Orthonormal (e1, e2, w) to a direction -- `3D_ergodic_learning/surfaces._frame`."""
    w = np.asarray(w, dtype=np.float64)
    w = w / max(np.linalg.norm(w), 1e-12)
    up = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(up, w)
    e1 /= max(np.linalg.norm(e1), 1e-12)
    e2 = np.cross(w, e1)
    return e1, e2, w


class Plane:
    """The planar patch and its (u, v) chart. All methods take numpy arrays or torch tensors."""

    def __init__(self, key='flat', scale=PLANE_SCALE):
        if key not in PLANE_NORMALS:
            raise KeyError(f'unknown plane {key!r}; known: {sorted(PLANE_NORMALS)}')
        self.key = key
        self.e1, self.e2, self.n = _frame(PLANE_NORMALS[key])
        self.s = float(scale)
        self.c = np.full(3, 0.5)
        self.o = self.c - 0.5 * self.s * (self.e1 + self.e2)        # world point of (u, v) = (0, 0)
        self.E = np.stack([self.e1, self.e2])                        # (2, 3)

    def _t(self, a, like):
        return torch.as_tensor(a, dtype=like.dtype, device=like.device)

    def to_world(self, uv):
        if torch.is_tensor(uv):
            return self._t(self.o, uv) + self.s * uv @ self._t(self.E, uv)
        return self.o + self.s * np.asarray(uv) @ self.E

    def to_uv(self, xyz):
        if torch.is_tensor(xyz):
            return (xyz - self._t(self.o, xyz)) @ self._t(self.E, xyz).T / self.s
        return (np.asarray(xyz) - self.o) @ self.E.T / self.s

    def sdf(self, xyz):
        """Signed distance to the plane, world units."""
        if torch.is_tensor(xyz):
            return (xyz - self._t(self.c, xyz)) @ self._t(self.n, xyz)
        return (np.asarray(xyz) - self.c) @ self.n

    def snap(self, xyz, clamp=True):
        """Nearest point on the patch (the residual of the attachment term, see `plane_dev`)."""
        uv = self.to_uv(xyz)
        if clamp:
            uv = uv.clamp(0.0, 1.0) if torch.is_tensor(uv) else np.clip(uv, 0.0, 1.0)
        return self.to_world(uv)

    def grid_uv(self, res, device='cpu', dtype=torch.float32):
        """(res*res, 2) cell centres in the chart, row-major [y, x] like the 2D grids."""
        ys, xs = torch.meshgrid(torch.linspace(0, 1, res, device=device, dtype=dtype),
                                torch.linspace(0, 1, res, device=device, dtype=dtype), indexing='ij')
        return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)

    def grid_world(self, res, device='cpu', dtype=torch.float32):
        return self.to_world(self.grid_uv(res, device, dtype))

    def corners(self):
        return self.to_world(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=np.float64))

    def describe(self):
        f = lambda v: '(' + ', '.join(f'{x:+.3f}' for x in v) + ')'        # noqa: E731
        return dict(plane=self.key, normal=f(self.n), e1=f(self.e1), e2=f(self.e2),
                    origin=f(self.o), scale=self.s)


# ===========================================================================
# Ergodic metric on the unit cube
# ===========================================================================

def make_k_grid3(K=K_DEFAULT):
    """(K^3, 3) index grid (k1 outermost) and Lambda_k = (1 + |k|^2)^(-(d+1)/2), d = 3."""
    k = np.array([[a, b, c] for a in range(K) for b in range(K) for c in range(K)], dtype=np.float64)
    return k, (1.0 + (k ** 2).sum(1)) ** (-2.0)


class Ergodic3D:
    """Fourier ergodic metric on [0, 1]^3, the 3D counterpart of `ExploreExploitErgodic`
    (weight w * 0.5 * Lambda_k, median split of |k| into explore / exploit bands)."""

    def __init__(self, K=K_DEFAULT, device='cpu', w=W_ERGODIC):
        self.K, self.device, self.w = K, torch.device(device), float(w)
        k, lam = make_k_grid3(K)
        self.k_idx = torch.as_tensor(k, dtype=torch.float32, device=self.device)
        self.Lambda = torch.as_tensor(lam, dtype=torch.float32, device=self.device)
        norm = np.sqrt((k ** 2).sum(1))
        self.low = torch.as_tensor(norm <= np.median(norm), device=self.device)
        self._k1 = torch.arange(K, dtype=torch.float32, device=self.device)

    def _tables(self, pts):
        a = math.pi * pts.unsqueeze(-1) * self._k1.to(pts.dtype)       # (..., T, 3, K)
        return torch.cos(a)

    def coeffs(self, pts, weights=None):
        """(..., T, 3) -> (..., K^3): time average (or `weights`-weighted average) of the basis.
        Separable: c[a, b, c] = mean_t cos(pi a x_t) cos(pi b y_t) cos(pi c z_t)."""
        pts = pts.to(self.device, torch.float32)
        C = self._tables(pts)
        if weights is None:
            out = torch.einsum('...ta,...tb,...tc->...abc', C[..., 0, :], C[..., 1, :], C[..., 2, :])
            out = out / pts.shape[-2]
        else:
            w = weights.to(self.device, torch.float32)
            w = w / w.sum(-1, keepdim=True).clamp(min=1e-12)
            out = torch.einsum('...t,...ta,...tb,...tc->...abc', w, C[..., 0, :], C[..., 1, :],
                               C[..., 2, :])
        return out.reshape(*out.shape[:-3], -1)

    def target_coeffs(self, density, plane):
        """Target coefficients of a density grid (R, R) on the patch (indexed [y, x]): the
        density-weighted average of the basis over the 3D grid points of the patch."""
        R = density.shape[-1]
        pts = plane.grid_world(R, device=self.device)
        return self.coeffs(pts, density.reshape(-1).clamp(min=0.0))

    def errors(self, c, phi_k):
        """(..., S), (S,) -> weighted squared errors (..., S)."""
        return self.w * 0.5 * self.Lambda * (c - phi_k) ** 2

    def energy(self, curves, phi_k):
        """(C, T, 3) -> (C,)"""
        return self.errors(self.coeffs(curves), phi_k).sum(-1)

    def score(self, curve, phi_k):
        e = self.errors(self.coeffs(curve), phi_k)
        lo, hi = float(e[self.low].sum()), float(e[~self.low].sum())
        return dict(E_ergodic_total=lo + hi, E_ergodic_explore=lo, E_ergodic_exploit=hi)


# ===========================================================================
# TSVEC-SVGD in 3D (port of `exploration/common/tsvec_svgd.TsvecSvgd`)
# ===========================================================================

def _accel_matrix(T, device, dtype):
    A = torch.zeros(T - 2, T, device=device, dtype=dtype)
    i = torch.arange(T - 2)
    A[i, i], A[i, i + 1], A[i, i + 2] = 1.0, -2.0, 1.0
    return A


class TsvecSvgd3D:
    """Preconditioned SVGD (Li et al., Eq. 21 with Detommaso et al.'s 1/N, per-timestep RBF kernel
    averaged over t, relative Levenberg-Marquardt damping) on 3D B-spline control points, one
    JOINT swarm per `run` call. Identical to the 2D `TsvecSvgd` except:

      * ergodic residual on the K^3 cube basis (separable product of three cosines, analytic
        product-rule Jacobian),
      * plane attachment r_p = sqrt(w_plane) * SDF(X_t) / s -- Li et al.'s V_f for a plane,
        which the 2D port had to drop,
      * boundary residual on the patch edges, in plane coordinates (u, v) instead of the unit
        square, and the control-point clip on (u, v) (the normal component is left to r_p),
      * smoothness, boundary and kernel distances in plane units (world / s), so on the flat
        plane these terms are the 2D ones.

    The whole Jacobian is assembled directly in control-point space (J = dr/dX @ dX/dP with
    X = B P), as in the 2D module, without autodiff."""

    def __init__(self, B, plane, K=K_DEFAULT, device='cpu', dtype=torch.float32, w_smooth=W_SMOOTH,
                 w_ergodic=W_ERGODIC, w_boundary=W_BOUNDARY, w_plane=W_PLANE, beta=1.0,
                 damping=LM_DAMPING, margin=BOUNDARY_MARGIN):
        self.device, self.dtype = torch.device(device), dtype
        self.B = torch.as_tensor(np.asarray(B), dtype=dtype, device=self.device)
        self.T, self.nxi = self.B.shape
        self.plane = plane
        self.s = plane.s
        self.erg = Ergodic3D(K, device=self.device)
        self.k_idx, self.Lambda = self.erg.k_idx.to(dtype), self.erg.Lambda.to(dtype)
        self.beta = float(beta)
        self.w_smooth, self.w_ergodic = w_smooth * beta, w_ergodic * beta
        self.w_boundary, self.w_plane = w_boundary * beta, w_plane * beta
        self.damping, self.margin = damping, margin
        t = lambda a: torch.as_tensor(a, dtype=dtype, device=self.device)   # noqa: E731
        self.o, self.c, self.n, self.E = t(plane.o), t(plane.c), t(plane.n), t(plane.E)
        A = _accel_matrix(self.T, self.device, dtype)
        I3 = torch.eye(3, device=self.device, dtype=dtype)
        self.J_s = (self.w_smooth ** 0.5 / self.s) * torch.kron(A @ self.B, I3)   # ((T-2)*3, nxi*3)
        self.A = A

    def render(self, P):
        return torch.einsum('ti,nid->ntd', self.B, P)

    def _pointwise_J(self, G):
        """G (N, T, k, 3) = dr/dX_t of k residuals per point -> (N, T*k, nxi*3)."""
        N, T, k, _ = G.shape
        return torch.einsum('ntkd,ti->ntkid', G, self.B).reshape(N, T * k, self.nxi * 3)

    def residual_and_J(self, P, phi_k):
        """P (N, nxi, 3) -> r (N, R), J = dr/dP (N, R, nxi*3)."""
        N = P.shape[0]
        X = self.render(P)
        T = X.shape[1]
        # smoothness (linear)
        r_s = (self.w_smooth ** 0.5 / self.s) * torch.einsum('ts,nsd->ntd', self.A, X).reshape(N, -1)
        J_s = self.J_s.unsqueeze(0).expand(N, -1, -1)
        # ergodic: c_s = mean_t prod_d cos(pi k_sd X_td)
        arg = math.pi * X.unsqueeze(1) * self.k_idx.view(1, -1, 1, 3)         # (N, S, T, 3)
        cs, sn = torch.cos(arg), torch.sin(arg)
        c = cs.prod(-1).mean(-1)                                               # (N, S)
        w_e = torch.sqrt(self.w_ergodic * self.Lambda)
        r_e = w_e * (c - phi_k)
        others = torch.stack([cs[..., 1] * cs[..., 2], cs[..., 0] * cs[..., 2],
                              cs[..., 0] * cs[..., 1]], dim=-1)                # (N, S, T, 3)
        dc = (-math.pi / T) * self.k_idx.view(1, -1, 1, 3) * sn * others
        J_e = w_e.view(1, -1, 1) * torch.einsum('nstd,ti->nsid', dc, self.B).reshape(N, -1, self.nxi * 3)
        # patch boundary in plane coordinates
        uv = (X - self.o) @ self.E.T / self.s                                  # (N, T, 2)
        sb = self.w_boundary ** 0.5
        lo, hi = torch.clamp(self.margin - uv, min=0.0), torch.clamp(uv - (1.0 - self.margin), min=0.0)
        r_b = sb * torch.cat([lo, hi], dim=-1)                                 # (N, T, 4)
        g_lo = -(sb / self.s) * (lo > 0).to(X.dtype).unsqueeze(-1) * self.E    # (N, T, 2, 3)
        g_hi = (sb / self.s) * (hi > 0).to(X.dtype).unsqueeze(-1) * self.E
        J_b = self._pointwise_J(torch.cat([g_lo, g_hi], dim=2))
        # plane attachment (V_f)
        sp = self.w_plane ** 0.5 / self.s
        r_p = sp * ((X - self.c) @ self.n)                                     # (N, T)
        J_p = self._pointwise_J((sp * self.n).view(1, 1, 1, 3).expand(N, T, 1, 3))
        r = torch.cat([r_s, r_e, r_b.reshape(N, -1), r_p], dim=1)
        J = torch.cat([J_s, J_e, J_b, J_p], dim=1)
        return r, J, X

    def clip(self, P):
        """Clamp the in-plane coordinates of the control points to PARAM_CLIP."""
        uv = (P - self.o) @ self.E.T / self.s
        h = (P - self.c) @ self.n
        uv = uv.clamp(*PARAM_CLIP)
        return self.o + self.s * uv @ self.E + h.unsqueeze(-1) * self.n

    def _kernel(self, X):
        """per-timestep RBF averaged over t, on plane-unit coordinates; dk/dX in world units."""
        Xs = X / self.s
        T = X.shape[1]
        diff = Xs.unsqueeze(1) - Xs.unsqueeze(0)                               # (N, N, T, 3)
        w = torch.exp(-(diff ** 2).sum(-1) / KERNEL_LENGTH)
        k = w.mean(-1)
        dk = (-2.0 / (KERNEL_LENGTH * T * self.s)) * w.unsqueeze(-1) * diff
        return k, dk

    def step(self, P, phi_k, tau=TAU, start=None):
        N = P.shape[0]
        D = self.nxi * 3
        r, J, X = self.residual_and_J(P, phi_k)
        H = torch.bmm(J.transpose(1, 2), J)
        score = -torch.bmm(J.transpose(1, 2), r.unsqueeze(-1)).squeeze(-1)
        k, dk = self._kernel(X)
        g = torch.einsum('ijtd,tp->ijpd', dk, self.B).reshape(N, N, D)
        phi_star = (torch.einsum('ij,id->jd', k, score) + g.sum(0)) / N
        lhs = (torch.einsum('ij,iab->jab', k ** 2, H) + torch.bmm(g.permute(1, 2, 0), g.permute(1, 0, 2))) / N
        diag = lhs.diagonal(dim1=1, dim2=2).mean(-1).view(N, 1, 1)
        lhs = lhs + (self.damping * diag + 1e-8) * torch.eye(D, device=self.device, dtype=self.dtype)
        alpha = torch.linalg.solve(lhs, phi_star.unsqueeze(-1)).squeeze(-1)
        P_new = self.clip((P.reshape(N, D) + tau * alpha).reshape(P.shape))
        if start is not None:
            P_new[:, 0] = start
        return P_new

    def run(self, P0, phi_k, n_iters, record=False, start=None):
        P = torch.as_tensor(np.asarray(P0) if not torch.is_tensor(P0) else P0,
                            dtype=self.dtype).to(self.device).clone()
        phi_k = torch.as_tensor(phi_k, dtype=self.dtype).to(self.device)
        if start is not None:
            start = torch.as_tensor(start, dtype=self.dtype).to(self.device).reshape(3)
            P[:, 0] = start
        log = None
        if record:
            log = torch.empty((n_iters + 1,) + tuple(P.shape), dtype=self.dtype, device=self.device)
            log[0] = P
        for it in range(n_iters):
            P = self.step(P, phi_k, start=start)
            if record:
                log[it + 1] = P
        return dict(final=P, log=log)


class AdamSvgd3D:
    """Reference SVGD with Adam (Liu & Wang's phi* with the median-heuristic RBF kernel on the
    whole parameter vector, Adam with bias correction), energy = `TsvecSvgd3D`'s V = 0.5 ||r||^2
    with all weights x beta -- the 3D port of `exploration/common/adam_svgd.AdamSvgd`. The Adam
    step is applied in plane units (x s in world units), so on the flat plane it equals 2D."""

    def __init__(self, B, plane, K=K_DEFAULT, device='cpu', beta=ADAM_BETA, lr=ADAM_LR):
        self.energy = TsvecSvgd3D(B, plane, K=K, device=device, beta=beta)
        self.device, self.lr, self.s = self.energy.device, lr, plane.s

    @staticmethod
    def svgd_direction(Pf, score):
        N = Pf.shape[0]
        if N == 1:
            return score
        sq = torch.cdist(Pf, Pf) ** 2
        iu = torch.triu_indices(N, N, 1, device=Pf.device)
        h = torch.clamp(torch.median(sq[iu[0], iu[1]]) / np.log(N + 1.0), min=1e-8)
        Kx = torch.exp(-sq / h)
        rep = (2.0 / h) * (Kx.sum(1, keepdim=True) * Pf - Kx @ Pf)
        return (Kx @ score + rep) / N

    def run(self, P0, phi_k, n_iters, record=False, start=None):
        en = self.energy
        P = torch.as_tensor(np.asarray(P0) if not torch.is_tensor(P0) else P0,
                            dtype=en.dtype).to(self.device).clone()
        phi_k = torch.as_tensor(phi_k, dtype=en.dtype).to(self.device)
        if start is not None:
            start = torch.as_tensor(start, dtype=en.dtype).to(self.device).reshape(3)
            P[:, 0] = start
        shape = P.shape
        m = torch.zeros(shape[0], shape[1] * 3, device=self.device, dtype=en.dtype)
        v = torch.zeros_like(m)
        log = None
        if record:
            log = torch.empty((n_iters + 1,) + tuple(shape), dtype=en.dtype, device=self.device)
            log[0] = P
        for it in range(n_iters):
            r, J, _ = en.residual_and_J(P, phi_k)
            g = torch.bmm(J.transpose(1, 2), r.unsqueeze(-1)).squeeze(-1)
            d = self.svgd_direction(P.reshape(shape[0], -1), -g)
            t = it + 1
            m = ADAM_B1 * m + (1 - ADAM_B1) * d
            v = ADAM_B2 * v + (1 - ADAM_B2) * d * d
            step = self.lr * self.s * (m / (1 - ADAM_B1 ** t)) / (torch.sqrt(v / (1 - ADAM_B2 ** t)) + ADAM_EPS)
            P = en.clip((P.reshape(shape[0], -1) + step).reshape(shape))
            if start is not None:
                P[:, 0] = start
            if record:
                log[t] = P
        return dict(final=P, log=log)


class SolverBatch3D:
    """`bs.run(init_curves, phi_k, start, seeds, n_iters, record)` contract of the 2D mission loop on
    top of `TsvecSvgd3D` / `AdamSvgd3D`: dense candidate curves (C, T, 3) are fitted to control
    points by least squares, refined as ONE joint swarm against the shared target, start pinned.
    `seeds` is unused (both solvers are deterministic given the initialisation)."""

    def __init__(self, solver):
        self.solver = solver
        self.B = solver.energy.B if isinstance(solver, AdamSvgd3D) else solver.B

    def run(self, init_curves, phi_k, start, seeds, n_iters, record=False):
        curves = torch.as_tensor(np.asarray(init_curves), dtype=torch.float64)
        C, T, _ = curves.shape
        B64 = self.B.detach().cpu().double()
        rhs = curves.permute(1, 0, 2).reshape(T, C * 3)
        P0 = torch.linalg.lstsq(B64, rhs).solution.reshape(-1, C, 3).permute(1, 0, 2).float()
        st = None if start is None else np.asarray(start, dtype=np.float64).reshape(-1, 3)[0]
        out = self.solver.run(P0, phi_k, n_iters, record=record, start=st)
        log = out['log'].permute(1, 0, 2, 3) if record and out['log'] is not None else None
        return dict(cps=log, final_cps=out['final'].detach().cpu().numpy().astype(np.float32))
