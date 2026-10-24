"""
attentive_kernel_belief.py
==========================
Density belief from a few point measurements: a Gaussian process with the
Attentive Kernel (Chen, Khardon, Liu, "AK: Attentive Kernel for Information
Gathering", RSS 2022).

Many independent GPs (one per task) are fitted in parallel. Every task owns its
own kernel network, amplitude and noise; the summed loss keeps the gradients of
different tasks disjoint, so one Adam optimiser fits all of them at once.

Settings follow the authors' reference demo: 10 fixed RBF primitives with
lengthscales in [0.01, 0.5] on inputs scaled to [-1, 1], a 2-10-10-P tanh
network with softmax and L2 normalisation, standardised targets, and Adam on
the negative log marginal likelihood (lr 1e-2 for amplitude/noise, 1e-3 for
the network).
"""

import math

import torch
import torch.nn.functional as F


def _inv_softplus(y):
    return math.log(math.expm1(y))


class BatchedAttentiveKernelGP(torch.nn.Module):
    """T independent GPs with Attentive Kernels on 2D inputs in [0, 1]^2."""

    def __init__(self, n_tasks, n_primitives=10, hidden=10, ls_min=0.01, ls_max=0.5,
                 jitter=1e-6, dtype=torch.float64, device='cpu', generator=None):
        super().__init__()
        self.jitter = jitter
        kw = dict(dtype=dtype, device=device)
        self.register_buffer('lengthscales', torch.linspace(ls_min, ls_max, n_primitives, **kw))

        def linear(fan_in, fan_out):
            # Same init as torch.nn.Linear, drawn independently per task.
            bound = 1.0 / math.sqrt(fan_in)
            w = (torch.rand(n_tasks, fan_in, fan_out, generator=generator, **kw) * 2 - 1) * bound
            b = (torch.rand(n_tasks, 1, fan_out, generator=generator, **kw) * 2 - 1) * bound
            return torch.nn.Parameter(w), torch.nn.Parameter(b)

        self.nn_w1, self.nn_b1 = linear(2, hidden)
        self.nn_w2, self.nn_b2 = linear(hidden, hidden)
        self.nn_w3, self.nn_b3 = linear(hidden, n_primitives)
        self.raw_amplitude = torch.nn.Parameter(torch.full((n_tasks,), _inv_softplus(1.0), **kw))
        self.raw_noise = torch.nn.Parameter(torch.full((n_tasks,), _inv_softplus(1.0), **kw))

    @staticmethod
    def _scale(x):
        return 2.0 * x - 1.0

    def _representation(self, x):
        h = torch.tanh(torch.baddbmm(self.nn_b1, x, self.nn_w1))
        h = torch.tanh(torch.baddbmm(self.nn_b2, h, self.nn_w2))
        z = torch.softmax(torch.baddbmm(self.nn_b3, h, self.nn_w3), dim=-1)
        return F.normalize(z, dim=-1)

    def kernel(self, x1, x2):
        """x1: (T, N, 2), x2: (T, M, 2) in [0, 1]. Returns (T, N, M)."""
        x1, x2 = self._scale(x1), self._scale(x2)
        z1, z2 = self._representation(x1), self._representation(x2)
        d2 = torch.cdist(x1, x2).pow(2).unsqueeze(-1)
        rbf = torch.exp(-0.5 * d2 / self.lengthscales.pow(2))
        mix = torch.einsum('tnp,tmp,tnmp->tnm', z1, z2, rbf)
        attention = z1 @ z2.transpose(1, 2)
        return F.softplus(self.raw_amplitude)[:, None, None] * attention * mix

    def _cholesky(self, x):
        K = x.shape[1]
        eye = torch.eye(K, dtype=x.dtype, device=x.device)
        noise = F.softplus(self.raw_noise)[:, None, None]
        return torch.linalg.cholesky(self.kernel(x, x) + (noise + self.jitter) * eye)

    def negative_log_marginal_likelihood(self, x, y):
        """x: (T, K, 2), y: (T, K) standardised. Returns (T,)."""
        L = self._cholesky(x)
        alpha = torch.cholesky_solve(y.unsqueeze(-1), L)
        quad = 0.5 * (y.unsqueeze(-1) * alpha).sum(dim=(1, 2))
        logdet = torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(dim=1)
        return quad + logdet + 0.5 * x.shape[1] * math.log(2 * math.pi)

    @torch.no_grad()
    def posterior_mean(self, x, y, xq, q_chunk=1024):
        """Standardised posterior mean at xq: (T, Q, 2) -> (T, Q)."""
        L = self._cholesky(x)
        alpha = torch.cholesky_solve(y.unsqueeze(-1), L)
        out = [self.kernel(xq[:, i:i + q_chunk], x) @ alpha
               for i in range(0, xq.shape[1], q_chunk)]
        return torch.cat(out, dim=1).squeeze(-1)


def measure_density(true_grids, k, noise_std, generator=None):
    """Simulates K point measurements per grid at uniform random locations.

    true_grids: (T, H, W), row index = y, column index = x (as in sample_particles).
    Returns (T, K, 3) with columns (x, y, noisy density value).
    """
    T, H, W = true_grids.shape
    dev = true_grids.device
    xy = torch.rand(T, k, 2, generator=generator, device=dev, dtype=true_grids.dtype)
    col = torch.round(xy[..., 0] * (W - 1)).long()
    row = torch.round(xy[..., 1] * (H - 1)).long()
    flat = true_grids.reshape(T, -1)
    val = torch.gather(flat, 1, row * W + col)
    val = val + noise_std * torch.randn(val.shape, generator=generator, device=dev, dtype=val.dtype)
    return torch.cat([xy, val.unsqueeze(-1)], dim=-1)


def fit_belief_grids(measurements, grid_res=64, n_iters=300, lr_hyper=1e-2, lr_nn=1e-3,
                     generator=None, dtype=torch.float64):
    """Fits one AK-GP per task and evaluates its posterior mean on a grid.

    measurements: (T, K, 3) with (x, y, value).
    Returns (T, grid_res, grid_res) belief grids, clamped at 0 and max-normalised
    like the true density grids in the particle runner.
    """
    T = measurements.shape[0]
    dev = measurements.device
    m = measurements.to(dtype)
    x, y_raw = m[..., :2], m[..., 2]
    y_mean = y_raw.mean(dim=1, keepdim=True)
    y_std = y_raw.std(dim=1, keepdim=True).clamp(min=1e-3)
    y = (y_raw - y_mean) / y_std

    gp = BatchedAttentiveKernelGP(T, dtype=dtype, device=dev, generator=generator)
    nn_params = [p for n, p in gp.named_parameters() if n.startswith('nn_')]
    hyper_params = [gp.raw_amplitude, gp.raw_noise]
    opt = torch.optim.Adam([{'params': hyper_params, 'lr': lr_hyper},
                            {'params': nn_params, 'lr': lr_nn}])
    for _ in range(n_iters):
        opt.zero_grad(set_to_none=True)
        gp.negative_log_marginal_likelihood(x, y).sum().backward()
        opt.step()

    lin = torch.linspace(0.0, 1.0, grid_res, dtype=dtype, device=dev)
    gy, gx = torch.meshgrid(lin, lin, indexing='ij')
    xq = torch.stack([gx, gy], dim=-1).reshape(1, -1, 2).expand(T, -1, -1)
    mean = gp.posterior_mean(x, y, xq) * y_std + y_mean
    return normalize_belief(mean.reshape(T, grid_res, grid_res))


def normalize_belief(mean_grid):
    """(T, G, G) predicted density -> clamped at 0, max-normalised, float32."""
    grid = mean_grid.clamp(min=0.0)
    peak = grid.amax(dim=(1, 2), keepdim=True)
    # A belief that is zero everywhere carries no information: fall back to uniform.
    grid = torch.where(peak > 0, grid / peak.clamp(min=1e-12), torch.ones_like(grid))
    return grid.float()
