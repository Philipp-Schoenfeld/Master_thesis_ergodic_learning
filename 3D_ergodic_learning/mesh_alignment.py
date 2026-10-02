r"""
mesh_alignment.py
==================
SE(3)-Tangentenausrichtung ("Constraint 5") fuer Bahnen des echten 3D-Flow-
Netzes auf den `surfaces.py`-Dreiecksnetzen.

Uebernimmt Formel und Ablauf aus
`thesis_architecture/constraints/05_se3_alignment/alignment.py`
(`SurfaceTangentAlignment`) unveraendert:

    E = w_surface * 0.5 * mean( SDF(C)^2 )          (auf der Flaeche bleiben)
      + w_align   * 0.5 * mean( (t_hat . n_hat)^2 )     (Tangente in der Ebene)

mit `t_hat` der Einheitstangente (zentrale Differenzen) und `n_hat` der
Einheitsnormalen. Portiert wird nur die SDF-Quelle: `alignment.py` benutzt die
analytische SDF von drei Primitivkoerpern (`thesis_architecture/shapes_3d.py`);
`run_surface_eval.py` bewertet aber auf den zehn (bzw. deutlich mehr)
beliebigen Dreiecksnetzen aus `surfaces.py` (Kugel, Wuerfel, Bunny, Torus, ...),
fuer die es keine geschlossene SDF-Formel gibt. `TrimeshSDF` unten ersetzt sie
durch den (unsigned) Abstand zum naechsten Netzpunkt, per
`trimesh.proximity`. Das Vorzeichen wird nicht gebraucht: die Energie ist
quadratisch, und der Abstands-Gradient zeigt auf beiden Seiten der Flaeche
korrekt von ihr weg -- derselbe "eingefrorener Bezugspunkt"-Trick, den
`alignment.py`s eigene `.normals()` schon fuer das Normalenfeld verwendet
(Gradient des Abstands zu einem als konstant behandelten naechsten Punkt =
Einheitsvektor weg von diesem Punkt = die Flaechennormale).

Absichtlich nicht direkt aus `thesis_architecture/constraints/05_se3_alignment`
importiert: dessen `alignment.py` zieht ueber `constraints/common.py` Module
gleichen Namens (`obstacles`, `model_zoo`, `ergodic_energy_torch`), die in
diesem Ordner (`3D_ergodic_learning/`) mit anderem Inhalt existieren -- ein
sys.path-Import wuerde je nach Einfuegereihenfolge das falsche Modul ziehen.
Die paar generischen, abhaengigkeitsfreien Helfer (`tangents`,
`curve_energy_grad`, `polish`) sind deshalb 1:1 aus `constraints/common.py`
uebernommen statt neu erfunden.
"""

import numpy as np
import torch


# ── aus thesis_architecture/constraints/common.py uebernommen (reine Torch-Helfer) ──
def tangents(curve, h=None):
    """Zentrale-Differenzen-Tangenten einer dichten Kurve, (B, pts, nd)."""
    pts = curve.shape[1]
    h = h if h is not None else 1.0 / (pts - 1)
    d = torch.zeros_like(curve)
    d[:, 1:-1] = (curve[:, 2:] - curve[:, :-2]) / (2 * h)
    d[:, 0] = (curve[:, 1] - curve[:, 0]) / h
    d[:, -1] = (curve[:, -1] - curve[:, -2]) / h
    return d


def curve_energy_grad(cps, energy_fn, B, normalize=True):
    """dE/d(Kontrollpunkte) fuer eine beliebige skalare Energie der dichten
    Kurve, per Autograd durch die (lineare) Basis-Abbildung curve = B @ cps."""
    with torch.enable_grad():
        c = cps.detach().requires_grad_(True)
        curve = torch.einsum('pi,bid->bpd', B, c)
        e = energy_fn(curve)
        (g,) = torch.autograd.grad(e, c)
    if normalize:
        g = g / B.sum(0).clamp(min=1e-8)[None, :, None]
    return g


def _clip(g, max_force):
    if max_force is None:
        return g
    n = g.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return g * (max_force / n).clamp(max=1.0)


def polish(cps, force, iters=250, lr=1.0, tol=1e-7, max_force=None):
    """Reiner Abstieg auf der Constraint-Energie."""
    cps = cps.clone()
    for _ in range(iters):
        g = _clip(force(cps), max_force)
        if g.abs().max() < tol:
            break
        cps = cps - lr * g
    return cps


# ── neu: Punkt-zu-Netz-Abstand als SDF-Ersatz fuer surfaces.py-Meshes ──
class TrimeshSDF:
    """(unsigned) Abstand zum naechsten Punkt auf einem `trimesh.Trimesh`.

    Bietet dieselbe `.sdf(P) -> (...,)`-Schnittstelle wie
    `thesis_architecture/shapes_3d.py`s `SDFShape`-Unterklassen, damit
    `SurfaceTangentAlignment` unten unveraendert bleiben kann.
    """

    def __init__(self, mesh):
        self.mesh = mesh

    def sdf(self, P):
        shape = P.shape
        P_np = P.detach().cpu().numpy().reshape(-1, 3).astype(np.float64)
        closest, _, _ = self.mesh.nearest.on_surface(P_np)
        closest_t = torch.from_numpy(closest).to(dtype=P.dtype, device=P.device)
        closest_t = closest_t.reshape(shape)
        return (P - closest_t).norm(dim=-1)


class SurfaceTangentAlignment:
    """SE(3)-Tangentenausrichtung -- Logik 1:1 aus
    `thesis_architecture/constraints/05_se3_alignment/alignment.py`, nur mit
    `TrimeshSDF` statt einer analytischen Primitiv-SDF."""

    def __init__(self, shape, w_surface=1.0, w_align=1.0):
        self.shape = shape
        self.w_surface = float(w_surface)
        self.w_align = float(w_align)

    def normals(self, curve):
        """Einheits-Flaechennormalen an den Kurvenpunkten, als konstantes Feld."""
        with torch.enable_grad():
            P = curve.detach().requires_grad_(True)
            (n,) = torch.autograd.grad(self.shape.sdf(P).sum(), P)
        return n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    def cos_tn(self, curve):
        """|cos| zwischen Einheitstangente und -normale, (B, pts). 0 = tangential."""
        t = tangents(curve)
        t = t / t.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return (t * self.normals(curve)).sum(dim=-1).abs()

    def energy(self, curve):
        e = self.w_surface * 0.5 * (self.shape.sdf(curve) ** 2).mean()
        if self.w_align > 0:
            t = tangents(curve)
            t = t / t.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            cos = (t * self.normals(curve)).sum(dim=-1)
            e = e + self.w_align * 0.5 * (cos ** 2).mean()
        return e

    def report(self, curve):
        """(max |SDF|, mean |cos(t, n)|) -- Oberflaechenadhaerenz und Tangentialitaet."""
        return (self.shape.sdf(curve).abs().max().item(),
                self.cos_tn(curve).mean().item())


def refine_with_alignment(cps_base, con, B, iters=300, lr=0.2, max_force=0.5):
    """Politur der vom 3D-Flow-Netz erzeugten Kontrollpunkte auf der
    kombinierten Oberflaechen- + Ausrichtungs-Energie. Startet direkt von den
    Netz-Kontrollpunkten -- keine z-Initialisierung noetig, die Bahn liegt
    bereits in 3D (anders als beim 2D-Modell in `run_alignment.py`)."""
    force = lambda c: curve_energy_grad(c, con.energy, B)      # noqa: E731
    return polish(cps_base, force, iters=iters, lr=lr, max_force=max_force)
