r"""
ppo.py
======
Option B: ein RL-Agent, der die Einstellung je Runde im Missionsloop waehlt.

Der Unterschied zu Option A
---------------------------
Option A lernt, das *kurzsichtige* Orakel nachzuahmen: welche Einstellung
verbessert die Abdeckung **in dieser Runde** am meisten. Damit erbt sie dessen
Grenze — eine Einstellung, die jetzt wenig bringt, aber die naechsten drei
Runden vorbereitet (etwa erst breit erkunden und danach ausbeuten), kann sie
nicht finden, weil sie in den Trainingsdaten als schlecht markiert ist.

PPO optimiert dagegen die **Rueckkehr der ganzen Episode**. Die Belohnung in
`rl_env.py` summiert sich genau zu `-J`, dem Ziel der Studie. Das ist der
eigentliche Grund, warum sich Option B trotz des viel hoeheren Aufwands lohnen
kann — und zugleich die Hypothese, die der Vergleich mit A pruefen soll.

Der gemischte Aktionsraum
-------------------------
Eine Aktion ist `(Phi-Modell, freier Parameter, SVGD-Budget)`: zwei diskrete
Groessen und eine stetige. Standardbibliotheken (Stable-Baselines3) bilden so
etwas nicht ohne eigenes Politiknetz ab, deshalb hier ein schlankes eigenes
PPO — bei diesem Aktionsraum sind ihre Vorteile (fertige Vektorisierung,
Logging) ohnehin gering, ihr Nachteil (wenig Kontrolle ueber die Koepfe)
bleibt.

Aufbau des Netzes:

    Rumpf         17 Zustandsmerkmale -> 128 -> 128
    Kopf diskret  -> |Modelle| x |SVGD-Budgets| Kategorien (Voreinstellung 12)
    Kopf stetig   -> je Kategorie ein Mittelwert fuer den normierten Parameter
    Kopf Wert     -> V(s)

Der stetige Kopf haengt **an der diskreten Wahl**: das gute tau eines
Level-Set-Modells hat mit dem guten kappa eines UCB-Modells nichts zu tun,
auch nicht in normierter Skala. Ein gemeinsamer Mittelwert muesste beide
Optima gleichzeitig treffen.

Gezogen wird aus einer Normalverteilung und danach auf [0,1] beschnitten; die
Log-Wahrscheinlichkeit ist die der unbeschnittenen Groesse (das uebliche
"clipped Gaussian"-Vorgehen). Eine `tanh`-Quetschung waere die Alternative,
braucht aber den Jacobi-Term und macht die Werte an den Raendern zaeh — und
gerade die Raender (tau gross, SVGD null) sind hier interessante Aktionen.

Warum erst Verhaltensklonen, dann PPO
-------------------------------------
Ein PPO-Lauf, der bei zufaelliger Initialisierung beginnt, verbringt die
ersten Hunderte Episoden damit, den Aktionsraum ueberhaupt kennenzulernen —
bei rund 2 s Rechenzeit je Runde ist das die teuerste Art, Zeit zu verlieren.
Die Orakeldaten aus `oracle.py` liegen aber ohnehin vor und enthalten je
Entscheidung die beste Aktion. Das Vortraining (`--bc_epochen`) klont dieses
Verhalten und laesst PPO dort anfangen, wo Option A aufhoert. Damit ist der
Vergleich am Ende auch sauber lesbar: was PPO **zusaetzlich** findet, ist der
Beitrag der Vorausschau, nicht der Vorsprung eines besseren Startpunkts.

    python -m exploration_optimierung.policy.ppo --iterationen 40
    python -m exploration_optimierung.policy.ppo --nur_bc      # nur Klonen
"""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from .. import DEFAULT_CKPT, RESULTS_DIR
from .. import mission as M
from . import DATASET_CSV, FESTE_POLICY, POLICY_DIR
from .budget import Zeitbudget
from .features import (MODELL_ORDNUNG, ZUSTANDS_MERKMALE, param_norm,
                       param_von_norm, zustands_merkmale)
from .rl_env import MissionUmgebung

MODELL_PT = os.path.join(POLICY_DIR, 'policy_b.pt')
BERICHT_JSON = os.path.join(RESULTS_DIR, 'policy_b_training.json')

SVGD_BUCKETS = (0, 25, 100)


class HybridPolitik(nn.Module):
    """Politik und Wertfunktion mit gemischtem Aktionsraum."""

    def __init__(self, n_zustand=len(ZUSTANDS_MERKMALE),
                 modelle=MODELL_ORDNUNG, svgd_buckets=SVGD_BUCKETS,
                 breite=128, log_std_init=-1.0):
        super().__init__()
        self.modelle = list(modelle)
        self.svgd_buckets = list(svgd_buckets)
        self.n_disk = len(self.modelle) * len(self.svgd_buckets)

        self.rumpf = nn.Sequential(
            nn.Linear(n_zustand, breite), nn.Tanh(),
            nn.Linear(breite, breite), nn.Tanh())
        self.kopf_disk = nn.Linear(breite, self.n_disk)
        self.kopf_mu = nn.Linear(breite, self.n_disk)
        self.kopf_wert = nn.Linear(breite, 1)
        # Eine eigene Streuung je Kategorie, nicht zustandsabhaengig: bei den
        # wenigen tausend Uebergaengen dieses Aufbaus ist eine gelernte
        # zustandsabhaengige Streuung fast nur Rauschen.
        self.log_std = nn.Parameter(torch.full((self.n_disk,), log_std_init))

        # Kleine Anfangsgewichte an den Koepfen: zu Beginn eine fast
        # gleichverteilte Politik statt einer zufaellig sehr sicheren.
        for kopf, faktor in ((self.kopf_disk, 0.01), (self.kopf_mu, 0.01),
                             (self.kopf_wert, 1.0)):
            nn.init.orthogonal_(kopf.weight, faktor)
            nn.init.zeros_(kopf.bias)

    # -- Zerlegung der diskreten Kategorie ----------------------------------
    def zerlege(self, d):
        """Kategorie -> (Modellname, SVGD-Iterationen)."""
        n_s = len(self.svgd_buckets)
        return self.modelle[int(d) // n_s], self.svgd_buckets[int(d) % n_s]

    def kategorie(self, modell, svgd):
        n_s = len(self.svgd_buckets)
        return self.modelle.index(modell) * n_s + self.svgd_buckets.index(svgd)

    # -- Vorwaerts ----------------------------------------------------------
    def forward(self, x):
        h = self.rumpf(x)
        return (self.kopf_disk(h), torch.sigmoid(self.kopf_mu(h)),
                self.kopf_wert(h).squeeze(-1))

    def verteilungen(self, x):
        logits, mu, wert = self(x)
        return Categorical(logits=logits), mu, wert

    def handeln(self, x, gierig=False):
        """-> (aktionen, log_p, wert, d, z) fuer eine Beobachtung (B, F)."""
        kat, mu, wert = self.verteilungen(x)
        if gierig:
            d = kat.probs.argmax(dim=-1)
            z = mu.gather(-1, d.unsqueeze(-1)).squeeze(-1)
            log_p = torch.zeros_like(z)
        else:
            d = kat.sample()
            mu_d = mu.gather(-1, d.unsqueeze(-1)).squeeze(-1)
            std_d = self.log_std[d].exp()
            z = Normal(mu_d, std_d).sample()
            log_p = (kat.log_prob(d)
                     + Normal(mu_d, std_d).log_prob(z))
        aktionen = self.zu_aktionen(d, z)
        return aktionen, log_p, wert, d, z

    def zu_aktionen(self, d, z):
        """Rohgroessen -> `[(modell, param, svgd), ...]` fuer die Mission."""
        out = []
        for di, zi in zip(d.tolist(), z.tolist()):
            modell, svgd = self.zerlege(di)
            out.append((modell, param_von_norm(modell, zi), int(svgd)))
        return out

    def bewerte(self, x, d, z):
        """Log-Wahrscheinlichkeit, Entropie und Wert fuer gegebene Aktionen."""
        kat, mu, wert = self.verteilungen(x)
        mu_d = mu.gather(-1, d.unsqueeze(-1)).squeeze(-1)
        std_d = self.log_std[d].exp()
        norm = Normal(mu_d, std_d)
        log_p = kat.log_prob(d) + norm.log_prob(z)
        entropie = kat.entropy() + norm.entropy()
        return log_p, entropie, wert


class Normierer:
    """Laufende Mittelwert-/Streuungsschaetzung der Beobachtungen.

    Die Merkmale haben sehr verschiedene Groessenordnungen (`mu_max` ~ 1,
    `n_obs` ~ 0,05, `glaube_cov` ~ 0,3). Ohne Normierung dominiert die
    Gewichtsinitialisierung, welches Merkmal ueberhaupt ankommt.
    """

    def __init__(self, n):
        self.n = 0
        self.mittel = np.zeros(n, dtype=np.float64)
        self.m2 = np.ones(n, dtype=np.float64)
        self.fest = False

    def aktualisiere(self, x):
        if self.fest:
            return
        for zeile in np.atleast_2d(x):
            self.n += 1
            delta = zeile - self.mittel
            self.mittel += delta / self.n
            self.m2 += delta * (zeile - self.mittel)

    def __call__(self, x):
        std = np.sqrt(self.m2 / max(self.n - 1, 1)) if self.n > 1 else np.ones_like(self.mittel)
        std[std < 1e-6] = 1.0
        return ((np.atleast_2d(x) - self.mittel) / std).astype(np.float32)

    def zustand(self):
        return dict(n=self.n, mittel=self.mittel, m2=self.m2)

    def laden(self, d):
        self.n, self.mittel, self.m2 = int(d['n']), np.asarray(d['mittel']), np.asarray(d['m2'])
        return self


class RLRichtlinie:
    """Die trainierte Politik als Regler fuer `mission.LaengenMission`.

    Ausgewertet wird **gierig** (haeufigste Kategorie, Mittelwert des stetigen
    Kopfes): eine Auswertung soll die gelernte Regel zeigen, nicht die
    Streuung der Exploration.
    """

    def __init__(self, politik, normierer, device='cpu'):
        self.politik = politik.to(device).eval()
        self.normierer = normierer
        self.device = torch.device(device)
        self.letzte_wahl = []

    def __call__(self, r, zustaende):
        x = np.stack([zustands_merkmale(z) for z in zustaende])
        t = torch.as_tensor(self.normierer(x), device=self.device)
        with torch.no_grad():
            aktionen, _lp, _v, _d, _z = self.politik.handeln(t, gierig=True)
        self.letzte_wahl = aktionen
        return aktionen

    def speichern(self, pfad, meta=None):
        os.makedirs(os.path.dirname(pfad), exist_ok=True)
        torch.save(dict(state_dict=self.politik.state_dict(),
                        modelle=self.politik.modelle,
                        svgd_buckets=self.politik.svgd_buckets,
                        zustands_merkmale=ZUSTANDS_MERKMALE,
                        normierer=self.normierer.zustand(),
                        meta=meta or {}), pfad)
        return pfad

    @classmethod
    def laden(cls, pfad, device='cpu'):
        ck = torch.load(pfad, map_location=device, weights_only=False)
        if list(ck.get('zustands_merkmale', ZUSTANDS_MERKMALE)) != list(ZUSTANDS_MERKMALE):
            raise ValueError(f"{os.path.basename(pfad)} wurde mit anderen "
                             "Zustandsmerkmalen trainiert — neu trainieren.")
        pol = HybridPolitik(modelle=ck['modelle'],
                            svgd_buckets=ck['svgd_buckets'])
        pol.load_state_dict(ck['state_dict'])
        norm = Normierer(len(ZUSTANDS_MERKMALE)).laden(ck['normierer'])
        norm.fest = True
        return cls(pol, norm, device=device)


# ---------------------------------------------------------------------------
# Verhaltensklonen aus den Orakeldaten
# ---------------------------------------------------------------------------

def lade_bc_daten(pfad=DATASET_CSV, svgd_buckets=SVGD_BUCKETS,
                  modelle=MODELL_ORDNUNG):
    """Beste Aktion je Entscheidung -> (X, d, u).

    Nur die Zeilen mit `ist_bester = 1`; die uebrigen Kandidatenzeilen sind
    fuer Option A da (dort ist gerade die ganze Bewertungskurve das Signal),
    fuer das Klonen zaehlt allein, was das Orakel gefahren haette.
    """
    X, d, u = [], [], []
    with open(pfad, 'r', newline='', encoding='utf-8') as f:
        for z in csv.DictReader(f):
            if int(z['ist_bester']) != 1:
                continue
            modell, svgd = z['modell'], int(z['svgd'])
            if modell not in modelle or svgd not in svgd_buckets:
                continue
            X.append([float(z[m]) for m in ZUSTANDS_MERKMALE])
            d.append(modelle.index(modell) * len(svgd_buckets)
                     + list(svgd_buckets).index(svgd))
            u.append(param_norm(modell, float(z['param'])))
    if not X:
        raise SystemExit(
            f"{pfad} enthaelt keine Zeilen mit den Buckets {svgd_buckets} — "
            "`policy.oracle` mit denselben --svgd_buckets laufen lassen.")
    return (np.asarray(X, dtype=np.float32), np.asarray(d, dtype=np.int64),
            np.asarray(u, dtype=np.float32))


def verhaltensklonen(politik, normierer, X, d, u, device, epochen=200,
                     lr=1e-3, batch=256, still=False):
    """Kreuzentropie auf der Kategorie, MSE auf dem stetigen Parameter."""
    normierer.aktualisiere(X)
    Xt = torch.as_tensor(normierer(X), device=device)
    dt = torch.as_tensor(d, device=device)
    ut = torch.as_tensor(u, device=device)
    opt = torch.optim.AdamW(politik.parameters(), lr=lr, weight_decay=1e-4)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()

    verlauf = []
    for ep in range(epochen):
        politik.train()
        perm = torch.randperm(Xt.shape[0], device=device)
        s_ce = s_mse = 0.0
        for a in range(0, Xt.shape[0], batch):
            b = perm[a:a + batch]
            logits, mu, _wert = politik(Xt[b])
            l_ce = ce(logits, dt[b])
            l_mse = mse(mu.gather(-1, dt[b].unsqueeze(-1)).squeeze(-1), ut[b])
            opt.zero_grad()
            (l_ce + l_mse).backward()
            opt.step()
            s_ce += float(l_ce) * len(b)
            s_mse += float(l_mse) * len(b)
        verlauf.append(dict(epoche=ep, ce=s_ce / Xt.shape[0],
                            mse=s_mse / Xt.shape[0]))
        if not still and (ep % 25 == 0 or ep == epochen - 1):
            with torch.no_grad():
                logits, _mu, _w = politik(Xt)
                treffer = float((logits.argmax(-1) == dt).float().mean())
            print(f"    BC Epoche {ep:3d}  CE {verlauf[-1]['ce']:.4f}  "
                  f"MSE {verlauf[-1]['mse']:.4f}  Trefferquote {treffer:.1%}")
    return verlauf


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

def sammle(env, politik, normierer, device, episoden=1):
    """Rollouts -> Uebergangspuffer (flach ueber Episode x Form)."""
    puffer = dict(x=[], d=[], z=[], logp=[], wert=[], rew=[], fertig=[])
    ergebnisse = []
    for _ in range(episoden):
        beob = env.reset()
        normierer.aktualisiere(beob)
        for t in range(env.n_max):
            xt = torch.as_tensor(normierer(beob), device=device)
            with torch.no_grad():
                aktionen, logp, wert, d, z = politik.handeln(xt)
            beob2, rew, fertig, info = env.step(aktionen)
            puffer['x'].append(xt.cpu().numpy())
            puffer['d'].append(d.cpu().numpy())
            puffer['z'].append(z.cpu().numpy())
            puffer['logp'].append(logp.cpu().numpy())
            puffer['wert'].append(wert.cpu().numpy())
            puffer['rew'].append(rew)
            puffer['fertig'].append(np.full(len(rew), float(fertig)))
            beob = beob2
            if not fertig:
                normierer.aktualisiere(beob)
        ergebnisse.append(dict(q_ende=float(np.mean(info['q'])),
                               zeilen=info['zeilen'],
                               aktionen=info['aktionen']))
    return {k: np.stack(v) for k, v in puffer.items()}, ergebnisse


def gae(rew, wert, fertig, gamma=1.0, lam=0.95):
    """Verallgemeinerter Vorteilsschaetzer ueber (T, S)-Puffer.

    `gamma = 1.0` ist hier kein Versehen: der Horizont ist endlich (`n_max`
    Runden, danach ist die Episode wirklich zu Ende, nicht abgeschnitten), und
    die Summe der Belohnungen ist bis auf eine Konstante genau `-J`. Jede
    Abzinsung wuerde spaete Runden geringer gewichten, als die Zielfunktion
    der Studie es tut.
    """
    T, S = rew.shape
    vorteil = np.zeros_like(rew)
    letzter = np.zeros(S, dtype=np.float32)
    for t in reversed(range(T)):
        nicht_ende = 1.0 - fertig[t]
        naechster_wert = wert[t + 1] if t + 1 < T else np.zeros(S, dtype=np.float32)
        delta = rew[t] + gamma * naechster_wert * nicht_ende - wert[t]
        letzter = delta + gamma * lam * nicht_ende * letzter
        vorteil[t] = letzter
    return vorteil, vorteil + wert


def ppo_schritt(politik, opt, puffer, vorteil, ziel, device, epochen=4,
                clip=0.2, c_wert=0.5, c_entropie=0.01, batch=256):
    x = torch.as_tensor(puffer['x'].reshape(-1, puffer['x'].shape[-1]), device=device)
    d = torch.as_tensor(puffer['d'].reshape(-1), device=device)
    z = torch.as_tensor(puffer['z'].reshape(-1), device=device)
    logp_alt = torch.as_tensor(puffer['logp'].reshape(-1), device=device)
    vt = torch.as_tensor(vorteil.reshape(-1), device=device)
    zt = torch.as_tensor(ziel.reshape(-1), device=device)
    vt = (vt - vt.mean()) / (vt.std() + 1e-8)

    stat = {}
    for _ in range(epochen):
        perm = torch.randperm(x.shape[0], device=device)
        for a in range(0, x.shape[0], batch):
            b = perm[a:a + batch]
            logp, entropie, wert = politik.bewerte(x[b], d[b], z[b])
            verhaeltnis = (logp - logp_alt[b]).exp()
            l1 = verhaeltnis * vt[b]
            l2 = torch.clamp(verhaeltnis, 1 - clip, 1 + clip) * vt[b]
            l_pol = -torch.min(l1, l2).mean()
            l_wert = ((wert - zt[b]) ** 2).mean()
            l_ent = entropie.mean()
            verlust = l_pol + c_wert * l_wert - c_entropie * l_ent
            opt.zero_grad()
            verlust.backward()
            nn.utils.clip_grad_norm_(politik.parameters(), 0.5)
            opt.step()
            stat = dict(politik=float(l_pol), wert=float(l_wert),
                        entropie=float(l_ent))
    return stat


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--iterationen', type=int, default=40)
    p.add_argument('--episoden_pro_iter', type=int, default=2)
    p.add_argument('--n_max', type=int, default=8)
    p.add_argument('--n_envs', type=int, default=8,
                   help="Formen je Episode; weniger = schneller, verrauschter")
    p.add_argument('--n_shapes', type=int, default=25)
    p.add_argument('--split', default='val', choices=['val', 'train'],
                   help="'train': Richtlinie auf den Trainingsformen des "
                        "Planernetzes lernen und die 25 Validierungsformen "
                        "ausschliesslich zum Testen behalten")
    p.add_argument('--bc_epochen', type=int, default=200)
    p.add_argument('--nur_bc', action='store_true')
    p.add_argument('--datensatz', default=DATASET_CSV)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--flow_steps', type=int, default=100)
    p.add_argument('--ckpt', default=DEFAULT_CKPT)
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=MODELL_PT)
    p.add_argument('--bericht', default=BERICHT_JSON)
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--max_minuten', type=float, default=None,
                   help="Zeitbudget. Laeuft es ab (oder kommt SIGTERM), endet "
                        "PPO nach der laufenden Iteration und speichert die "
                        "bis dahin beste Politik.")
    a = p.parse_args(argv)

    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    politik = HybridPolitik().to(device)
    normierer = Normierer(len(ZUSTANDS_MERKMALE))

    bc_verlauf = []
    if a.bc_epochen > 0 and os.path.exists(a.datensatz):
        X, d, u = lade_bc_daten(a.datensatz, politik.svgd_buckets,
                                politik.modelle)
        print(f"Verhaltensklonen auf {len(X)} Orakel-Entscheidungen ...")
        bc_verlauf = verhaltensklonen(politik, normierer, X, d, u, device,
                                      epochen=a.bc_epochen)
    elif a.bc_epochen > 0:
        print(f"Kein Datensatz unter {a.datensatz} — PPO startet ohne "
              "Vortraining (deutlich mehr Episoden noetig).")

    if a.nur_bc:
        RLRichtlinie(politik, normierer, device=device).speichern(
            a.out, meta=dict(bc_only=True, bc_epochen=a.bc_epochen))
        print(f"Nur Verhaltensklonen — gespeichert unter {a.out}")
        return

    n_workers = a.workers
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 2)
    pool = None
    if n_workers > 1 and any(s > 0 for s in politik.svgd_buckets):
        pool = ProcessPoolExecutor(max_workers=n_workers,
                                   initializer=M._worker_init, initargs=(0,))

    planner = M.build_planner(ckpt=a.ckpt, device=device,
                              flow_steps=a.flow_steps)
    names, truths = M.load_holdout(resolution=96, device=device,
                                   limit=a.n_shapes, split=a.split)
    args = M.build_mission_args(device, phi_model=FESTE_POLICY['phi_model'],
                                param=FESTE_POLICY['param'])
    env = MissionUmgebung(planner, truths, names, args, n_max=a.n_max,
                          n_envs=a.n_envs, pool=pool, seed=a.seed)
    print(f"PPO: {a.iterationen} Iterationen x {a.episoden_pro_iter} Episoden "
          f"x {a.n_max} Runden x {a.n_envs} Formen "
          f"= {a.iterationen * a.episoden_pro_iter * a.n_max * a.n_envs} "
          f"Uebergaenge  [{device}, Split '{a.split}', {len(names)} Formen]")

    opt = torch.optim.AdamW(politik.parameters(), lr=a.lr, weight_decay=0.0)
    verlauf, bestes, bester_ertrag = [], None, -float('inf')
    budget = Zeitbudget(a.max_minuten, name='PPO')
    t0 = time.perf_counter()
    iter_sek = 0.0
    try:
        for it in range(a.iterationen):
            if budget.abgelaufen(iter_sek):
                print(f"  Abbruch nach {it} von {a.iterationen} Iterationen "
                      f"({budget.grund()}) — die beste Politik wird "
                      "gespeichert.", flush=True)
                break
            t_iter = time.perf_counter()
            puffer, ergebnisse = sammle(env, politik, normierer, device,
                                        episoden=a.episoden_pro_iter)
            vorteil, ziel = gae(puffer['rew'], puffer['wert'], puffer['fertig'])
            stat = ppo_schritt(politik, opt, puffer, vorteil, ziel, device)
            ertrag = float(puffer['rew'].sum(axis=0).mean())
            q_ende = float(np.mean([e['q_ende'] for e in ergebnisse]))
            # J der Episode aus der Rueckkehr: J = q_0 - Rueckkehr mit q_0 = 1.
            j = 1.0 - ertrag
            verlauf.append(dict(iteration=it, ertrag=ertrag, J=j,
                                q_ende=q_ende, **stat,
                                sekunden=time.perf_counter() - t0))
            print(f"  Iter {it:3d}  Rueckkehr {ertrag:+.4f}  J~{j:.4f}  "
                  f"q(n) {q_ende:.4f}  Entropie {stat['entropie']:.3f}  "
                  f"[{(time.perf_counter() - t0) / 60:.1f} min]", flush=True)
            if ertrag > bester_ertrag:
                bester_ertrag = ertrag
                bestes = {k: v.detach().clone()
                          for k, v in politik.state_dict().items()}
            iter_sek = time.perf_counter() - t_iter
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    if bestes is not None:
        politik.load_state_dict(bestes)
    normierer.fest = True
    RLRichtlinie(politik, normierer, device=device).speichern(
        a.out, meta=dict(iterationen=a.iterationen, n_max=a.n_max,
                         n_envs=a.n_envs, split=a.split, seed=a.seed,
                         bc_epochen=a.bc_epochen,
                         bester_ertrag=bester_ertrag))
    print(f"Politik gespeichert -> {a.out}")

    os.makedirs(os.path.dirname(a.bericht), exist_ok=True)
    with open(a.bericht, 'w', encoding='utf-8') as f:
        json.dump(dict(konfiguration=vars(a), bc_verlauf=bc_verlauf[-5:],
                       ppo_verlauf=verlauf, bester_ertrag=bester_ertrag,
                       formen=names), f, indent=2, ensure_ascii=False,
                  default=str)
    print(f"Bericht gespeichert -> {a.bericht}")


if __name__ == '__main__':
    main()
