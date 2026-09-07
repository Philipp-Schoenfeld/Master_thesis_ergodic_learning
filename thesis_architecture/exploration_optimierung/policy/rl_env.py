r"""
rl_env.py
=========
Der Missionsloop als Umgebung mit `reset`/`step` — die Grundlage von Option B.

Warum ueberhaupt ein Umbau
--------------------------
`LaengenMission.run` faehrt eine ganze Mission am Stueck; ein RL-Agent braucht
die Schleife andersherum: Zustand heraus, Aktion herein, einen Schritt weiter.
Der Umbau ist trotzdem klein, weil der Simulator seit dem Policy-Hook selbst
schon je Runde nach der Aktion fragt: die Umgebung setzt die Aktion des
Agenten in eben diesen Hook und ruft `mission.round(r)` auf. Der Agent faehrt
damit **denselben** Simulator wie die feste Einstellung, das Orakel und die
ueberwachte Richtlinie.

Alle Formen zugleich = vektorisierte Umgebung
---------------------------------------------
Die Mission rechnet alle Formen im Gleichschritt (ein Flow-ODE-Aufruf statt S
einzelner, siehe `mission.py`). Genau das ist die Struktur, die PPO ohnehin
will: S parallele Episoden je Gradientenschritt. Eine Umgebung ist hier also
nicht *eine* Mission, sondern ein Buendel von S Missionen — `step` nimmt S
Aktionen und gibt S Belohnungen.

Die Belohnung
-------------
    r_t = -(q_t - q_{t-1}) - lambda_len - lambda_time * dt(svgd_iters)

`q` ist der bezogene Abdeckungsfehler `cov_norm` **gegen die Wahrheit**.
Summiert ueber eine Mission ergibt das

    sum_t r_t = (q_0 - q_n) - lambda_len * n - lambda_time * t(n)
              = const - J(Einstellung, n),

die Rueckkehr ist also bis auf eine Konstante das negative J der Studie. Der
Agent optimiert damit genau die Zielfunktion aus `objective.py` und nicht
etwas, das ihr nur aehnlich sieht.

Dass die Belohnung die Wahrheit benutzt, die Beobachtung aber nicht, ist
Absicht und der uebliche asymmetrische Aufbau: der Simulator darf im Training
mehr wissen als der Agent, solange der Agent zur Laufzeit ohne dieses Wissen
auskommt. Was er sieht, steht vollstaendig in `features.ZUSTANDS_MERKMALE` —
kein Eintrag darin beruehrt `truths`.

Die Zeitstrafe ist **modelliert**, nicht gemessen
-------------------------------------------------
`dt = T_PLAN + T_ITER * svgd_iters` statt der echten Uhr. Zwei Gruende: die
echte Wanduhr haengt an der Auslastung des Rechners und wuerde dem Agenten ein
zufaelliges Belohnungsrauschen geben, das nichts mit seiner Entscheidung zu
tun hat; und sie ist auf CPU und GPU verschieden, was die Trainingslaeufe
untereinander unvergleichbar machte. Die beiden Konstanten sind an die
gemessene SVGD-Kurve der Studie angepasst (1,67 s bei 0 Iterationen, 4,65 s
bei 400). In der **Auswertung** (`evaluate.py`) wird dagegen die echte Zeit
gemessen — dort geht es um die Zahl, nicht um das Lernsignal.
"""

import numpy as np
import torch

from .. import mission as M
from ..objective import DEFAULT_LAMBDA_LEN, DEFAULT_LAMBDA_TIME
from .features import zustands_merkmale

#: Rechenzeit je Runde, angepasst an `SVGD_CURVE` der vollen Studie
#: (0 Iter. -> 1,67 s; 400 Iter. -> 4,65 s; dazwischen praktisch linear).
T_PLAN = 1.67
T_ITER = 0.00745


def zeit_kosten(svgd_iters):
    return T_PLAN + T_ITER * float(svgd_iters)


class MissionUmgebung:
    """S parallele Laengeneinheit-Missionen mit Schritt-Schnittstelle.

    Args:
        planner: gemeinsamer `CfmPlanner` (wird nicht kopiert — das Netz ist
                 der teure Teil und bleibt zwischen Episoden geladen).
        truths, names: der Formvorrat, aus dem eine Episode zieht.
        args:    Missionsparameter (`mission.build_mission_args`); die
                 Zieldichte-Felder darin werden vom Agenten ueberschrieben.
        n_max:   Runden je Episode.
        n_envs:  wie viele Formen eine Episode gleichzeitig faehrt. None =
                 alle. Weniger Formen heisst schnellere, aber verrauschtere
                 Gradientenschritte.
    """

    def __init__(self, planner, truths, names, args, n_max=8, n_envs=None,
                 pool=None, lambda_len=DEFAULT_LAMBDA_LEN,
                 lambda_time=DEFAULT_LAMBDA_TIME, seed=0):
        self.planner = planner
        self.truths_pool = truths
        self.names_pool = list(names)
        self.args = args
        self.n_max = int(n_max)
        self.n_envs = int(n_envs or len(names))
        self.pool = pool
        self.lambda_len = float(lambda_len)
        self.lambda_time = float(lambda_time)
        self.rng = np.random.default_rng(seed)
        self.mission = None
        self.r = 0

    # -- Ablauf -------------------------------------------------------------
    def reset(self, indizes=None, seed=None):
        """Neue Episode. -> Beobachtung (S, F)."""
        if indizes is None:
            k = min(self.n_envs, len(self.names_pool))
            indizes = self.rng.choice(len(self.names_pool), size=k,
                                      replace=False)
        self.indizes = np.asarray(indizes)
        truths = self.truths_pool[self.indizes]
        names = [self.names_pool[i] for i in self.indizes]
        seed = int(self.rng.integers(1 << 30)) if seed is None else int(seed)

        self._aktionen = None
        self.mission = M.LaengenMission(
            self.planner, truths, names, self.args, svgd_iters=0, seed=seed,
            pool=self.pool, policy=lambda r, z: self._aktionen)
        self.r = 0
        # q_0: der Zustand ohne jede Bahn. `cov_norm` ist genau darauf bezogen
        # (`mission.blind_coverage`), also ist der Startwert 1,0 je Form.
        self.q_prev = np.ones(len(names), dtype=np.float64)
        return self.beobachtung()

    def beobachtung(self):
        z = self.mission.zustaende(self.r, self.n_max)
        return np.stack([zustands_merkmale(x) for x in z]).astype(np.float32)

    def step(self, aktionen):
        """Eine Runde mit je einer Aktion pro Form.

        -> (beobachtung, belohnung (S,), fertig, info)
        """
        if self.mission is None:
            raise RuntimeError("erst `reset()` aufrufen")
        self._aktionen = [tuple(a) for a in aktionen]
        zeilen = self.mission.round(self.r, n_max=self.n_max)
        self.r += 1

        q = np.asarray([z['cov_norm'] for z in zeilen], dtype=np.float64)
        dt = np.asarray([zeit_kosten(a[2]) for a in self._aktionen])
        belohnung = -(q - self.q_prev) - self.lambda_len - self.lambda_time * dt
        self.q_prev = q

        fertig = self.r >= self.n_max
        info = dict(zeilen=zeilen, q=q, aktionen=list(self._aktionen),
                    formen=[z['shape'] for z in zeilen])
        beob = (self.beobachtung() if not fertig
                else np.zeros_like(self.beobachtung()))
        return beob, belohnung.astype(np.float32), fertig, info

    # -- Bequemlichkeit -----------------------------------------------------
    def episode(self, waehle, sammle_zeilen=True):
        """Eine ganze Episode mit einem Regler `waehle(beob, zustaende)`.

        Fuer Auswertung und Verhaltensklonen; PPO benutzt `step` direkt, weil
        es die Zwischengroessen (Wert, Log-Wahrscheinlichkeit) braucht.
        """
        beob = self.reset()
        zeilen, gesamt = [], np.zeros(len(self.indizes))
        for _ in range(self.n_max):
            aktionen = waehle(beob, self.mission.zustaende(self.r, self.n_max))
            beob, r, fertig, info = self.step(aktionen)
            gesamt += r
            if sammle_zeilen:
                zeilen += info['zeilen']
            if fertig:
                break
        return zeilen, gesamt


def baue_umgebung(ckpt, device, n_shapes=25, n_max=8, n_envs=None,
                  flow_steps=100, split='val', pool=None, seed=0,
                  phi_model='niveau', param=0.6067, **kw):
    """Planer, Formen und Umgebung in einem Aufruf — fuer die Runner."""
    planner = M.build_planner(ckpt=ckpt, device=device, flow_steps=flow_steps)
    names, truths = M.load_holdout(resolution=96, device=device,
                                   limit=n_shapes, split=split)
    args = M.build_mission_args(device, phi_model=phi_model, param=param, **kw)
    env = MissionUmgebung(planner, truths, names, args, n_max=n_max,
                          n_envs=n_envs, pool=pool, seed=seed)
    return env, planner, names, truths, args
