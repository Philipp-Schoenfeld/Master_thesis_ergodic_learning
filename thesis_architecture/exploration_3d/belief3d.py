"""
belief3d.py
===========
Gaussian-process belief over the target density on a 3D surface, plus the
measurement model of the robot.

The belief lives on a fixed set of M points sampled on the figure's surface.
The kernel acts on the *position and the surface normal*:
``k = exp(-|p-p'|^2 / 2 l^2) * exp(-|n-n'|^2 / 2 l_n^2)``.  Without the normal
term the front and the back of a thin shape (a word, a digit: a few millimetres
apart) would share information although only the front carries density.

Kernels (``make_belief``): ``rbf`` as above; ``ak`` replaces the position part by the
Attentive Kernel (Chen, Khardon, Liu, RSS 2022) as in ``common/ak_belief.py`` of the 2D
side, i.e. 10 RBF primitives (0.01-0.5, on positions scaled to [-1, 1]) mixed with
location-dependent weights from a small network that is refitted on the marginal
likelihood after every ``observe``; the normal term stays.  ``np`` (pretrained attentive
neural process) only exists for 2D inputs, there is no 3D checkpoint.
"""

import copy
import threading

import numpy as np
import torch
from scipy.spatial import cKDTree


def rbf_kernel(a, b, variance=1.0):
    """Unit-lengthscale RBF on pre-scaled features (N, d), (M, d) -> (N, M)."""
    return variance * torch.exp(-0.5 * torch.cdist(a, b).pow(2))


class AttentiveKernel3D(torch.nn.Module):
    """Attentive Kernel on positions in [0, 1]^3: k(x, x') = (z . z') sum_p z_p z'_p rbf_p(x, x')
    with attention weights z(x) = normalise(softmax(MLP(2x - 1)))."""

    def __init__(self, n_primitives=10, hidden=10, ls_min=0.01, ls_max=0.5, in_dim=3, seed=0):
        super().__init__()
        self.register_buffer('lengthscales', torch.linspace(ls_min, ls_max, n_primitives))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden), torch.nn.Tanh(),
                torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                torch.nn.Linear(hidden, n_primitives))

    def weights(self, x):
        z = torch.softmax(self.net(2.0 * x - 1.0), dim=-1)
        return torch.nn.functional.normalize(z, dim=-1)

    def forward(self, a, b):
        za, zb = self.weights(a), self.weights(b)
        d2 = torch.cdist(2.0 * a, 2.0 * b).pow(2)
        mix = 0.0
        for p in range(len(self.lengthscales)):          # one (N, M) matrix per primitive
            mix = mix + za[:, p:p + 1] * zb[:, p].unsqueeze(0) * torch.exp(
                -0.5 * d2 / self.lengthscales[p] ** 2)
        return (za @ zb.transpose(0, 1)) * mix


class SurfaceBelief:
    def __init__(self, pts, nrm, lengthscale=0.08, normal_scale=0.5, variance=1.0,
                 noise=0.05, device='cpu', dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.var, self.noise = variance, noise
        self.ls, self.nl = lengthscale, normal_scale
        pts = np.asarray(pts, dtype=np.float32)
        nrm = np.asarray(nrm, dtype=np.float32)
        feat = np.concatenate([pts / lengthscale, nrm / normal_scale], axis=1)
        self.grid = torch.from_numpy(feat).to(self.device, dtype)       # (M, 6) belief points
        self.M = self.grid.shape[0]
        self.X = torch.zeros((0, 6), device=self.device, dtype=dtype)
        self.y = torch.zeros((0,), device=self.device, dtype=dtype)
        self._cache = None
        self._lock = threading.RLock()      # the planning thread and the GUI both read/write
        self.chunk = 4096                   # belief points per posterior block

    # -- state ---------------------------------------------------------------
    def clone(self):
        b = SurfaceBelief.__new__(SurfaceBelief)
        b.__dict__.update(self.__dict__)
        b.X, b.y, b._cache = self.X.clone(), self.y.clone(), None
        b._lock = threading.RLock()
        return b

    @property
    def n_obs(self):
        return self.X.shape[0]

    def observe_indices(self, idx, vals):
        """Measurements at belief points ``idx`` (K,) with values ``vals`` (K,)."""
        idx = torch.as_tensor(np.asarray(idx), device=self.device, dtype=torch.long)
        vals = torch.as_tensor(np.asarray(vals), device=self.device, dtype=self.dtype)
        with self._lock:
            self.X = torch.cat([self.X, self.grid[idx]], dim=0)
            self.y = torch.cat([self.y, vals], dim=0)
            self._cache = None

    # -- posterior -----------------------------------------------------------
    def _k(self, a, b):
        """Covariance of two feature blocks (N, 6), (M, 6); subclasses swap only this."""
        return rbf_kernel(a, b, self.var)

    def _factor(self):
        if self._cache is None:
            n = self.X.shape[0]
            K = self._k(self.X, self.X)
            K = K + (self.noise ** 2) * torch.eye(n, device=self.device, dtype=self.dtype)
            L = torch.linalg.cholesky(K)
            alpha = torch.cholesky_solve(self.y.unsqueeze(-1), L)
            self._cache = (L, alpha)
        return self._cache

    @torch.no_grad()
    def posterior(self):
        """(mu, sigma), each (M,)."""
        with self._lock:
            return self._posterior()

    def _posterior(self):
        if self.X.shape[0] == 0:
            z = torch.zeros(self.M, device=self.device, dtype=self.dtype)
            return z, torch.full_like(z, self.var ** 0.5)
        L, alpha = self._factor()
        mu = torch.empty(self.M, device=self.device, dtype=self.dtype)
        var = torch.empty_like(mu)
        c = self.chunk
        for s in range(0, self.M, c):                       # chunked: (M x n) can be large
            Ks = self._k(self.grid[s:s + c], self.X)
            mu[s:s + c] = (Ks @ alpha).squeeze(-1)
            v = torch.cholesky_solve(Ks.transpose(0, 1), L)
            var[s:s + c] = self.var - (Ks * v.transpose(0, 1)).sum(dim=-1)
        return mu, var.clamp(min=1e-12).sqrt()

    def total_uncertainty(self):
        return float(self.posterior()[1].sum())


class AttentiveSurfaceBelief(SurfaceBelief):
    """``SurfaceBelief`` with the Attentive Kernel on the positions (see module docstring)."""

    def __init__(self, *args, fit_iters=200, lr=1e-3, seed=0, **kw):
        super().__init__(*args, **kw)
        self.ak = AttentiveKernel3D(seed=seed).to(self.device, self.dtype)
        self.ak.requires_grad_(False)
        self.fit_iters, self.lr = fit_iters, lr
        self._opt = None
        self.chunk = 512                    # the kernel builds one (N, M) block per primitive

    def _k(self, a, b):
        pos = self.ak(a[:, :3] * self.ls, b[:, :3] * self.ls)
        return self.var * pos * torch.exp(-0.5 * torch.cdist(a[:, 3:], b[:, 3:]).pow(2))

    def clone(self):
        b = super().clone()
        b.ak, b._opt = copy.deepcopy(self.ak), None
        return b

    def observe_indices(self, idx, vals):
        super().observe_indices(idx, vals)
        with self._lock:
            self._fit()

    def _fit(self):
        params = list(self.ak.parameters())
        if self._opt is None:
            self._opt = torch.optim.Adam(params, lr=self.lr)
        X, y = self.X.detach(), self.y.detach().unsqueeze(-1)
        eye = (self.noise ** 2 + 1e-5) * torch.eye(X.shape[0], device=self.device, dtype=self.dtype)
        for p in params:
            p.requires_grad_(True)
        with torch.enable_grad():
            for _ in range(self.fit_iters):
                L = torch.linalg.cholesky(self._k(X, X) + eye)
                alpha = torch.cholesky_solve(y, L)
                nll = 0.5 * (y * alpha).sum() + torch.log(torch.diagonal(L)).sum()
                self._opt.zero_grad(set_to_none=True)
                nll.backward()
                self._opt.step()
        for p in params:
            p.requires_grad_(False)
        self._cache = None


KERNELS = ('rbf', 'ak', 'np')


def make_belief(kernel, pts, nrm, seed=0, **kw):
    """Empty belief of the chosen kernel (``KERNELS``)."""
    if kernel == 'rbf':
        return SurfaceBelief(pts, nrm, **kw)
    if kernel == 'ak':
        return AttentiveSurfaceBelief(pts, nrm, seed=seed, **kw)
    if kernel == 'np':
        raise NotImplementedError('the pretrained neural process is 2D only; there is no 3D checkpoint')
    raise KeyError(kernel)


class Sensor:
    """Measures the ground-truth density on the surface under the robot.

    The footprint of a sensor at position ``p`` is every belief point within
    ``radius`` (Euclidean) of ``p``.  Values are the projected ground-truth
    density plus Gaussian noise; at most ``max_points`` measurements per call are
    kept, chosen at random (the GP costs O(n^3)).
    """

    def __init__(self, figure, noise=0.02, seed=0):
        self.pts = figure.pts
        self.w = figure.w
        self.lit_idx = np.flatnonzero(figure.lit)               # only the lit side is measurable
        self.tree = cKDTree(self.pts[self.lit_idx])
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def footprint(self, curve, radius):
        """Indices of all surface points within ``radius`` of any curve point."""
        hits = self.tree.query_ball_point(np.asarray(curve, dtype=np.float64), r=radius)
        idx = np.unique(np.concatenate([np.asarray(h, dtype=np.int64) for h in hits])) \
            if len(hits) else np.zeros(0, dtype=np.int64)
        return self.lit_idx[idx]

    def measure(self, curve, radius, max_points=96):
        idx = self.footprint(curve, radius)
        if len(idx) > max_points:
            idx = self.rng.choice(idx, size=max_points, replace=False)
        vals = self.w[idx] + self.noise * self.rng.normal(size=len(idx)).astype(np.float32)
        return idx, vals

    def prior(self, n=12):
        """``n`` random surface points as prior knowledge (single measurements)."""
        idx = self.rng.choice(self.lit_idx, size=n, replace=False)
        return idx, self.w[idx] + 0.02 * self.rng.normal(size=n).astype(np.float32)
