r"""
belief.py
=========
Gauss-Prozess-Glaube ueber ein unbekanntes Dichtefeld.

Das ist der Baustein, der die ganze Fragestellung erst moeglich macht: er liefert
zu jedem Punkt einen Schaetzwert `mu` und eine Unsicherheit `sigma`. Die
Zielverteilung wird daraus abgeleitet, statt vorgegeben zu sein — genau die
Verschiebung, um die es bei "erkunden, bevor man abdecken kann" geht.

Zwei Eigenschaften sind nicht verhandelbar und bestimmen die Implementierung:

1. **Differenzierbar in den Messorten.** Variante B propagiert durch die
   Glaubensaktualisierung zurueck, um die Trajektorie zu lernen. Deshalb ist der
   Posterior als `torch.linalg.solve` geschrieben und nicht ueber eine
   Gitterheuristik.
2. **Guenstig genug fuer eine Schleife.** Variante C plant im Receding Horizon
   dutzendfach pro Mission neu, jedes Mal mit aktualisiertem Glauben. Die
   Cholesky-Zerlegung wird deshalb einmal pro Aktualisierung gebildet und fuer
   alle Gitterabfragen wiederverwendet.

Konvention: das Feld lebt auf [0,1]^2, Werte sind nichtnegative Dichten. Der
Prior-Mittelwert ist 0, was bedeutet "hier ist vermutlich nichts" — die
konservative Annahme, die einen Roboter zum Nachsehen zwingt, statt ihn
unbesuchte Gebiete optimistisch fuer voll zu halten.
"""

import torch


def rbf_kernel(a, b, lengthscale, variance=1.0):
    """(N,2), (M,2) -> (N,M). Differenzierbar in beiden Argumenten."""
    d2 = torch.cdist(a, b).pow(2)
    return variance * torch.exp(-0.5 * d2 / (lengthscale ** 2))


class GPBelief:
    """Posterior ueber ein Dichtefeld, aufgebaut aus Punktmessungen.

    Args:
        grid_res:    Aufloesung des Auswertungsgitters
        lengthscale: Korrelationslaenge in Domaenenbreiten. 0.08 heisst, dass
                     eine Messung rund 8 % der Kante weit Information traegt —
                     grob die Strichbreite der Buchstabenformen im Datensatz.
        variance:    Prior-Varianz, also die Unsicherheit an einem voellig
                     unbesuchten Ort.
        noise:       Messrauschen. Verhindert zugleich, dass die Gram-Matrix
                     bei dicht beieinanderliegenden Messpunkten singulaer wird,
                     was auf einer abgefahrenen Trajektorie der Normalfall ist.

                     **Der Default 1e-2 ist zu klein.** Beim Auswerten in
                     `apply_cfm_belief.py` gemessen: Messpunkte auf einer Bahn
                     sind keine unabhaengigen Stichproben, sondern liegen dicht
                     beieinander. Der GP interpoliert dann das Rauschen, und
                     der Posterior-Mittelwert schiesst auf einer kurzen Bahn
                     nach [-2.5, 3.9] ueber, obwohl die Wahrheit in [0,1] liegt
                     (|alpha| ~ 780). Ein Sweep ueber kurze und lange Bahnen:

                         noise   RMSE kurz   RMSE lang
                         0.010       0.611       0.229
                         0.020       0.349       0.163
                         0.050       0.227       0.173
                         0.100       0.187       0.146

                     0.05 ist auf beiden besser als 0.01. Der Default bleibt
                     hier unveraendert, damit die bereits gefahrenen Vergleiche
                     der Varianten A-E ihre Zahlen behalten — neue Auswertungen
                     sollten `noise` aber explizit hoeher setzen.
    """

    def __init__(self, grid_res=48, lengthscale=0.08, variance=1.0,
                 noise=1e-2, device='cpu', dtype=torch.float32):
        self.res = grid_res
        self.ls = lengthscale
        self.var = variance
        self.noise = noise
        self.device = torch.device(device)
        self.dtype = dtype

        ys, xs = torch.meshgrid(
            torch.linspace(0, 1, grid_res, device=device, dtype=dtype),
            torch.linspace(0, 1, grid_res, device=device, dtype=dtype),
            indexing='ij')
        # (R*R, 2) in (x, y), passend zur Konvention der Partikelwolken
        self.grid = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)

        self.X = torch.zeros((0, 2), device=device, dtype=dtype)
        self.y = torch.zeros((0,), device=device, dtype=dtype)
        self._cache = None

    # ── Zustand ────────────────────────────────────────────────────────────
    def clone(self):
        """Flache Kopie mit eigenen Messdaten — fuer Rollouts, die den
        Ausgangsglauben nicht veraendern duerfen."""
        b = GPBelief(self.res, self.ls, self.var, self.noise,
                     self.device, self.dtype)
        b.X, b.y = self.X.clone(), self.y.clone()
        return b

    def observe(self, pts, vals):
        """Messungen hinzufuegen. pts: (N,2), vals: (N,)

        Geraet *und* dtype werden angeglichen. Nur den dtype zu casten reicht
        nicht: `observation.measure` gibt die Punkte auf dem Geraet der
        uebergebenen Bahn zurueck, und die liegt bei einer Auswertung gegen das
        Wahrheitsgitter auf der CPU, waehrend der Glaube bei einem Netz-Planer
        auf der GPU steht. Ohne die Angleichung bricht `torch.cat` ab — und
        zwar erst in der Anytime-Auswertung, also nachdem alle Missionen schon
        gerechnet wurden.
        """
        self.X = torch.cat(
            [self.X, pts.to(device=self.device, dtype=self.dtype)], dim=0)
        self.y = torch.cat(
            [self.y, vals.to(device=self.device, dtype=self.dtype)], dim=0)
        self._cache = None
        return self

    @property
    def n_obs(self):
        return self.X.shape[0]

    # ── Posterior ──────────────────────────────────────────────────────────
    def _factor(self):
        """Cholesky der Gram-Matrix plus geloester Beobachtungsvektor."""
        if self._cache is not None:
            return self._cache
        n = self.X.shape[0]
        K = rbf_kernel(self.X, self.X, self.ls, self.var)
        K = K + (self.noise ** 2) * torch.eye(n, device=self.device,
                                              dtype=self.dtype)
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(self.y.unsqueeze(-1), L)
        self._cache = (L, alpha)
        return self._cache

    def posterior(self, pts):
        """mu, sigma an beliebigen Punkten. pts: (N,2) -> (N,), (N,)

        Ohne Messungen ist mu = 0 und sigma = sqrt(variance): maximale
        Unsicherheit ueberall, was den Explorationsterm anfangs das ganze Gebiet
        abdecken laesst.
        """
        pts = pts.to(device=self.device, dtype=self.dtype)
        if self.X.shape[0] == 0:
            z = torch.zeros(pts.shape[0], device=self.device, dtype=self.dtype)
            return z, torch.full_like(z, self.var ** 0.5)

        L, alpha = self._factor()
        Ks = rbf_kernel(pts, self.X, self.ls, self.var)          # (N, n)
        mu = (Ks @ alpha).squeeze(-1)
        v = torch.cholesky_solve(Ks.transpose(0, 1), L)          # (n, N)
        var = self.var - (Ks * v.transpose(0, 1)).sum(dim=-1)
        return mu, var.clamp(min=1e-12).sqrt()

    def posterior_grid(self):
        """mu, sigma auf dem Auswertungsgitter, je (R, R)."""
        mu, sd = self.posterior(self.grid)
        return mu.view(self.res, self.res), sd.view(self.res, self.res)

    def total_uncertainty(self):
        """Summierte Standardabweichung ueber das Gitter.

        Das Mass, gegen das der Informationsgewinn einer Mission gerechnet wird.
        Bewusst die Summe der Standardabweichungen und nicht die der Varianzen:
        letztere wird von wenigen sehr unsicheren Zellen dominiert und verdeckt,
        ob breit oder nur punktuell erkundet wurde.
        """
        return self.posterior_grid()[1].sum()

    # ── Vorausschau ────────────────────────────────────────────────────────
    def uncertainty_after(self, pts):
        """Gesamtunsicherheit, *wenn* an `pts` gemessen wuerde. Differenzierbar.

        Nutzt eine Eigenschaft des Gauss-Prozesses, die diese ganze Variante erst
        moeglich macht: **die Posterior-Varianz haengt nur von den Messorten ab,
        nicht von den gemessenen Werten.** Man kann also exakt ausrechnen, wie
        viel Unsicherheit eine geplante Bahn aufloesen wird, bevor man sie
        abfaehrt — ohne die Messung zu simulieren und ohne eine Annahme ueber
        ihren Ausgang zu treffen.

        Der Gradient dieser Groesse nach `pts` ist damit ein exaktes Signal
        dafuer, wohin die Bahn laufen sollte, um zu lernen. Variante B
        propagiert genau hier zurueck.
        """
        pts = pts.to(device=self.device, dtype=self.dtype)
        X = torch.cat([self.X, pts], dim=0) if self.X.shape[0] else pts
        n = X.shape[0]
        K = rbf_kernel(X, X, self.ls, self.var)
        K = K + (self.noise ** 2) * torch.eye(n, device=X.device, dtype=X.dtype)
        L = torch.linalg.cholesky(K)
        Ks = rbf_kernel(self.grid, X, self.ls, self.var)
        v = torch.cholesky_solve(Ks.transpose(0, 1), L)
        var = self.var - (Ks * v.transpose(0, 1)).sum(dim=-1)
        return var.clamp(min=1e-12).sqrt().sum()


class MaskiertesWissen(GPBelief):
    r"""Grundwahrheit in einem Gebiet, gar kein Wissen ausserhalb.

    Der Unterschied zu `prior_points`: dort werden im bekannten Gebiet *n
    Punktmessungen* gezogen und der GP interpoliert dazwischen — mit
    erheblicher Restunsicherheit und einem Mittelwert, der die Wahrheit nur
    annaehert. Hier ist das Feld im bekannten Gebiet **exakt** bekannt und
    ausserhalb **gar nicht**.

    Warum kein dicht konditionierter GP: ein Gitter aus Wahrheitspunkten im
    Abstand 0,04 ist bei Korrelationslaenge 0,08 zu 97 % korreliert, und die
    Cholesky-Zerlegung der Gram-Matrix bricht in float32 zusammen (gemessen:
    ab 205 Punkten nicht mehr positiv definit). Ausserdem wuerde ein GP
    Information *ueber die Grenze hinweg* tragen — genau das, was hier nicht
    gewollt ist: ausserhalb soll kein Wissen vorliegen, auch kein indirektes.

    Missionsmessungen entlang der abgefahrenen Bahn wirken unveraendert ueber
    `observe` und schliessen die Luecke nach und nach. Im bekannten Gebiet
    aendern sie nichts — dort steht die Wahrheit ohnehin schon.
    """

    def __init__(self, maske, wahrheit, sigma_bekannt=0.0, **kw):
        super().__init__(**kw)
        self.maske = maske.to(device=self.device)
        self.wahrheit = wahrheit.to(device=self.device, dtype=self.dtype)
        self.sigma_bekannt = float(sigma_bekannt)

    def clone(self):
        b = MaskiertesWissen(self.maske, self.wahrheit, self.sigma_bekannt,
                             grid_res=self.res, lengthscale=self.ls,
                             variance=self.var, noise=self.noise,
                             device=self.device, dtype=self.dtype)
        b.X, b.y = self.X.clone(), self.y.clone()
        return b

    def _nachschlagen(self, pts, feld):
        R = feld.shape[-1]
        ix = (pts[:, 0] * (R - 1)).round().long().clamp(0, R - 1)
        iy = (pts[:, 1] * (R - 1)).round().long().clamp(0, R - 1)
        return feld[iy, ix]

    def posterior(self, pts):
        mu, sd = super().posterior(pts)
        pts = pts.to(device=self.device, dtype=self.dtype)
        drin = self._nachschlagen(pts, self.maske.to(self.dtype)) > 0.5
        mu = torch.where(drin, self._nachschlagen(pts, self.wahrheit), mu)
        sd = torch.where(drin, torch.full_like(sd, self.sigma_bekannt), sd)
        return mu, sd


def muster_maske(pattern, res, device='cpu'):
    """Bool-Gitter (res, res): True, wo das Feld als bekannt gilt."""
    a = torch.linspace(0, 1, res, device=device)
    Y, X = torch.meshgrid(a, a, indexing='ij')
    if pattern in ('zufall', 'alles', None):
        return torch.ones_like(X, dtype=torch.bool)
    if pattern == 'haelfte':
        return X < 0.5
    if pattern == 'quadranten':
        return ((X < 0.5) & (Y < 0.5)) | ((X >= 0.5) & (Y >= 0.5))
    if pattern == 'loch':
        return ((X - 0.5) ** 2 + (Y - 0.5) ** 2) > 0.28 ** 2
    raise KeyError(pattern)
