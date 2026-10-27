r"""
belief_guidance.py
==================
Inference-time guidance of the CFM planner with a GP belief: the expected
ergodic error AFTER measuring along a candidate trajectory.

For a GP belief over the density field f, the ergodic target coefficients
phi_k = <F_k, f> / Z are linear in f (fixed normaliser Z), hence jointly
Gaussian. Their posterior covariance after additionally measuring at the
points P(c) of a candidate curve c depends only on WHERE is measured, not on
the values -- so it is available in closed form, differentiable in c, before
anything is driven. The guidance objective per candidate is

    G(c) = sum_k 0.5 Lambda_k [ (c_k - phi_k^target)^2
                                - beta * (Sigma_kk^now - Sigma_kk^after(c)) ]

first term: ergodic error against the round's planning target (the same
criterion the mission selects candidates by); second term: how much of the
uncertainty about the target coefficients the measurements along c remove,
weighted by the same Lambda_k as the metric (low frequencies first). beta = 0
leaves pure ergodic guidance. The reduction is a Schur complement:

    C = A K(G, P) - A K(G, X) (K_XX + s^2 I)^-1 K(X, P)          (M x m)
    S = K(P, P) - K(P, X) (K_XX + s^2 I)^-1 K(X, P) + s^2 I       (m x m)
    Sigma^now - Sigma^after = diag(C S^-1 C^T) / Z^2

with X the measurements so far, G a guidance grid and A_kg = F_k(g). Works for
every belief with a kernel (`GPBelief._k`: RBF, Attentive Kernel); the neural
process has no closed-form covariance.
"""

import torch

from ergodic_energy_torch import coeffs_from_points, fourier_basis


class SpectralInfoGuide:
    """Gradient of G with respect to the control points, for the candidates of
    ONE shape (one belief, one planning target).

    belief:  GPBelief / AKBelief before this round's measurements
    phi:     (R, R) planning target density of the round
    ee:      ExploreExploitErgodic (k_idx, Lambda of the project metric)
    B:       (T, nxi) render matrix control points -> curve points
    beta:    weight of the information term (0 = ergodic guidance only)
    n_meas:  points of the curve treated as future measurement locations
    grid:    resolution of the integration grid for A K(G, .)
    """

    def __init__(self, belief, phi, ee, B, beta=1.0, n_meas=32, grid=32, jitter=1e-6):
        self.beta, self.jitter = float(beta), jitter
        dev = belief.device
        self.dev = dev
        self.k_idx = ee.k_idx.to(dev)
        self.Lambda = ee.Lambda.to(device=dev, dtype=torch.float64)
        self.B = B.to(device=dev, dtype=torch.float32)
        self.phi_t = ee.target_coeffs(phi).to(device=dev, dtype=torch.float32)
        self.meas_idx = torch.linspace(0, B.shape[0] - 1, n_meas, device=dev).round().long()
        self.belief = belief
        self.noise2 = float(belief.noise) ** 2
        if self.beta <= 0.0:
            return
        lin = torch.linspace(0.0, 1.0, grid, device=dev, dtype=torch.float64)
        gy, gx = torch.meshgrid(lin, lin, indexing='ij')
        self.G = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)          # (g, 2)
        self.A = fourier_basis(self.G, self.k_idx.double()).T.contiguous()       # (M, g)
        mu, _ = belief.posterior(self.G)
        # Fixed normaliser of phi = <F, f> / Z: the believed mass on the grid,
        # floored so an empty prior belief (mean 0) does not blow the term up.
        self.Z = float(mu.clamp(min=0.0).sum().clamp(min=0.1 * self.G.shape[0]))
        X = belief.X.to(torch.float64)
        self.X = X
        if X.shape[0]:
            K_XX = belief._k(X, X).double() + self.noise2 * torch.eye(X.shape[0], device=dev,
                                                                     dtype=torch.float64)
            self.L = torch.linalg.cholesky(K_XX)
            AK_GX = self.A @ belief._k(self.G, X).double()                       # (M, n)
            self.W = torch.linalg.solve_triangular(self.L, AK_GX.T, upper=False).T  # (M, n)
        else:
            self.L = self.W = None

    def objective(self, cps):
        """G per candidate. cps: (N, nxi, 2) -> (N,)"""
        curves = torch.einsum('pi,nid->npd', self.B, cps.float())
        c = coeffs_from_points(curves, self.k_idx)
        G = 0.5 * (self.Lambda.float() * (c - self.phi_t) ** 2).sum(-1)
        if self.beta > 0.0:
            G = G - self.beta * self.info_reduction(curves.clamp(0.0, 1.0)).float()
        return G

    def info_reduction(self, curves):
        """sum_k 0.5 Lambda_k (Sigma^now - Sigma^after)_kk / Z^2 per candidate."""
        P = curves[:, self.meas_idx].double()                                    # (N, m, 2)
        N, m, _ = P.shape
        flat = P.reshape(-1, 2)
        k = self.belief._k
        C = (self.A @ k(self.G, flat).double()).reshape(-1, N, m).permute(1, 0, 2)  # (N, M, m)
        eye = torch.eye(m, device=self.dev, dtype=torch.float64)
        S = torch.stack([k(P[i], P[i]).double() for i in range(N)])              # (N, m, m)
        if self.L is not None:
            V = torch.linalg.solve_triangular(self.L, k(self.X, flat).double(), upper=False)
            V = V.reshape(-1, N, m).permute(1, 0, 2)                              # (N, n, m)
            C = C - self.W.unsqueeze(0) @ V
            S = S - V.transpose(1, 2) @ V
        S = S + (self.noise2 + self.jitter) * eye
        Ls = torch.linalg.cholesky(S)
        T = torch.linalg.solve_triangular(Ls, C.transpose(1, 2), upper=False)    # (N, m, M)
        return 0.5 * (self.Lambda * T.pow(2).sum(1)).sum(-1) / self.Z ** 2

    def grad(self, cps):
        """dG/dcps, (N, nxi, 2)."""
        with torch.enable_grad():
            x = cps.detach().float().requires_grad_(True)
            self.objective(x).sum().backward()
        return x.grad.detach()


class ShapeGuides:
    """Dispatch a batch of candidates that spans several shapes: candidate i
    belongs to `guides[i // n_init]`, offset by `first` (the batch offset)."""

    def __init__(self, guides, n_init, first=0):
        self.guides, self.n_init, self.first = guides, n_init, first

    def __call__(self, cps):
        out = torch.empty_like(cps, dtype=torch.float32)
        idx = torch.arange(self.first, self.first + cps.shape[0]) // self.n_init
        for j in idx.unique().tolist():
            sel = (idx == j).nonzero().squeeze(1).to(cps.device)
            out[sel] = self.guides[j].grad(cps[sel]).to(out.device)
        return out
