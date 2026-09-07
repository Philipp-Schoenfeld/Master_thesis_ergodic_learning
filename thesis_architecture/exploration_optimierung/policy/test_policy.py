#!/usr/bin/env python3
r"""
test_policy.py
==============
Prueft die Stellen, an denen die gelernten Regler still falsch rechnen koennten.

Wie `exploration_optimierung/test_smoke.py` geht es nicht um Abdeckung, sondern
um die Annahmen, die einem Ergebnis hinterher niemand mehr ansieht:

* Sieht der Regler wirklich **nur** Groessen ohne die Wahrheit? Ein Merkmal,
  das die Zieldichte durchreicht, wuerde in jeder Auswertung glaenzen und in
  der Anwendung fehlen.
* Faehrt die Mission wirklich das, was der Regler waehlt — auch wenn er je
  Form und Runde etwas anderes waehlt (Zieldichte-Modell *und* SVGD-Budget)?
* Ist die Umkehrung der Parameter-Normierung exakt? Sonst faehrt der Agent
  einen anderen Wert, als er gelernt hat.
* Summiert sich die Belohnung der RL-Umgebung wirklich zu `-J`? Genau davon
  haengt ab, ob Option B die Zielfunktion der Studie optimiert oder etwas
  anderes, das ihr aehnlich sieht.
* Bleibt die Mission ohne Regler bitgleich zur bisherigen Studie?

Laeuft ohne Netz und ohne GPU: geplant wird mit einem Ersatzplaner, der
zufaellige Kontrollpunkte liefert. Geprueft wird die Mechanik, nicht die Guete.

    python -m exploration_optimierung.policy.test_policy
"""

import os
import sys

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = 'exploration_optimierung.policy'

from .. import mission as M                                  # noqa: E402
from .. import objective as O                                # noqa: E402
from ..objective import DEFAULT_LAMBDA_LEN, DEFAULT_LAMBDA_TIME  # noqa: E402
from . import features as F                                  # noqa: E402
from .features import Zustand, aktionsraster                 # noqa: E402
from .model import WertNetz, WertRichtlinie, normierung      # noqa: E402
from .ppo import HybridPolitik, Normierer, RLRichtlinie, gae  # noqa: E402
from .rl_env import MissionUmgebung, zeit_kosten             # noqa: E402

_FEHLER = []


def check(name, cond, info=''):
    print(f"  [{'ok' if cond else 'FEHLER'}] {name}" + (f"   {info}" if info else ''))
    if not cond:
        _FEHLER.append(name)
    return cond


class ErsatzPlaner:
    """Planer ohne Netz: glatte Zufallsbahnen, aber mit korrekter Form.

    Genug, um die Mechanik zu pruefen (Zeilenzuordnung, Startpunkt, Aktionen),
    und schnell genug fuer einen Selbsttest ohne GPU.
    """

    def __init__(self, pts=64, nxi=8, device='cpu'):
        self.pts, self.nxi = pts, nxi
        self.device = torch.device(device)
        self.start_cond = True
        self.aufrufe = 0
        self.letzte_batchgroesse = None

    def plan(self, particles, n_candidates=1, start=None, **kw):
        self.aufrufe += 1
        self.letzte_batchgroesse = int(particles.shape[0])
        g = torch.Generator(device='cpu').manual_seed(1234 + self.aufrufe)
        cps = torch.rand((n_candidates, self.nxi, 2), generator=g).to(self.device)
        if start is not None:
            cps[:, 0] = start.reshape(n_candidates, 2).to(self.device)
        return cps

    def render(self, cps):
        t = torch.linspace(0, 1, self.pts, device=self.device)
        i = (t * (cps.shape[1] - 1)).clamp(0, cps.shape[1] - 1.001)
        lo = i.floor().long()
        frac = (i - lo.float()).unsqueeze(-1)
        return cps[:, lo] * (1 - frac) + cps[:, (lo + 1).clamp(max=cps.shape[1] - 1)] * frac


def _welt(S=3, res=32, device='cpu'):
    """Kuenstliche Wahrheiten: je Form ein Gauss-Fleck an anderer Stelle."""
    ys, xs = torch.meshgrid(torch.linspace(0, 1, res),
                            torch.linspace(0, 1, res), indexing='ij')
    truths, names = [], []
    for i in range(S):
        cx, cy = 0.25 + 0.5 * (i % 2), 0.3 + 0.4 * ((i // 2) % 2)
        t = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 0.12 ** 2))
        truths.append((t / t.max()).to(device))
        names.append(f'form{i}')
    return names, torch.stack(truths)


# ---------------------------------------------------------------------------

def test_merkmale_ohne_wahrheit():
    """Der Merkmalsvektor darf sich nicht aendern, wenn nur die Wahrheit
    wechselt — sonst sickert sie in die Beobachtung."""
    print("\nMerkmale")
    res = 32
    mu = torch.rand(res, res) * 0.5
    sd = torch.rand(res, res) * 0.2 + 0.1
    bahn = torch.rand(40, 2)
    z = Zustand(mu=mu, sd=sd, visit=None, driven=bahn, runde=2, n_max=8,
                n_obs=30)
    v1 = z.merkmale()
    v2 = Zustand(mu=mu.clone(), sd=sd.clone(), visit=None, driven=bahn.clone(),
                 runde=2, n_max=8, n_obs=30).merkmale()
    check("gleicher Zustand -> gleicher Vektor", np.allclose(v1, v2))
    check("Laenge stimmt", len(v1) == len(F.ZUSTANDS_MERKMALE),
          f"{len(v1)} von {len(F.ZUSTANDS_MERKMALE)}")
    check("alle Werte endlich", bool(np.all(np.isfinite(v1))))
    # Der leere Anfangszustand darf nicht abstuerzen und muss definiert sein.
    z0 = Zustand(mu=torch.zeros(res, res), sd=torch.ones(res, res), visit=None,
                 driven=None, runde=0, n_max=8, n_obs=0)
    v0 = z0.merkmale()
    check("Anfangszustand (kein Wissen, keine Bahn) ist endlich",
          bool(np.all(np.isfinite(v0))))
    quelle = open(F.__file__, encoding='utf-8').read()
    check("features.py fasst `truth` nirgends an",
          'truth' not in quelle.replace('coverage_vs_truth', ''))


def test_parameter_umkehr():
    print("\nParameter-Normierung")
    ok = True
    for modell in F.MODELL_ORDNUNG:
        for u in (0.0, 0.137, 0.5, 0.913, 1.0):
            p = F.param_von_norm(modell, u)
            ok &= abs(F.param_norm(modell, p) - u) < 1e-6
    check("param_norm(param_von_norm(u)) == u fuer alle Modelle", ok)
    check("kappa ist logarithmisch",
          abs(F.param_von_norm('ucb', 0.5)
              - (0.15 * 10.0) ** 0.5) < 1e-6)
    check("SVGD-Normierung waechst monoton",
          F.svgd_norm(0) < F.svgd_norm(25) < F.svgd_norm(100) < F.svgd_norm(400))
    raster = aktionsraster(['niveau', 'ucb'], 3, (0, 25))
    check("Aktionsraster ist das Kreuzprodukt", len(raster) == 2 * 3 * 2,
          f"{len(raster)} Kandidaten")


def test_hook_faehrt_die_aktion():
    """Die Mission muss das fahren, was der Regler waehlt — je Form anders."""
    print("\nPolicy-Hook der Mission")
    names, truths = _welt(3)
    planer = ErsatzPlaner()
    args = M.build_mission_args('cpu', phi_model='niveau', param=0.5)

    gewaehlt = [('niveau', 0.3, 0), ('ucb', 2.0, 0), ('mass', 0.7, 0)]
    gesehen = []

    def regler(r, zustaende):
        gesehen.append([type(z).__name__ for z in zustaende])
        return list(gewaehlt)

    m = M.LaengenMission(planer, truths, names, args, svgd_iters=0, seed=0,
                         policy=regler)
    zeilen = m.run(2)
    check("Regler wurde je Runde einmal gefragt", len(gesehen) == 2)
    check("Regler sah je Form einen Zustand",
          all(len(g) == 3 for g in gesehen))
    modelle = {z['shape']: z['gewaehlt_modell'] for z in zeilen}
    check("jede Form fuhr ihr eigenes Modell",
          [modelle[n] for n in names] == [a[0] for a in gewaehlt],
          str([modelle[n] for n in names]))
    check("Parameter steht in der Zeile",
          all(z['gewaehlt_param'] > 0 for z in zeilen))
    check("eine Planung je Runde (alle Formen gebatcht)", planer.aufrufe == 2,
          f"{planer.aufrufe} Aufrufe")

    # Ohne Regler muss alles bleiben, wie es war.
    planer2 = ErsatzPlaner()
    m2 = M.LaengenMission(planer2, truths, names, args, svgd_iters=0, seed=0)
    z2 = m2.run(2)
    check("ohne Regler keine Aktionsspalten",
          all('gewaehlt_modell' not in z for z in z2))
    check("ohne Regler dieselbe Spaltenmenge wie bisher",
          set(z2[0]) == {'shape', 'n_exec', 'cov', 'cov_norm', 'erg_truth',
                         'belief_rmse', 'info_gain', 'path_len', 'n_obs',
                         'plan_s', 'svgd_s'})


def test_svgd_gruppierung():
    """Verschiedene SVGD-Budgets je Form duerfen sich nicht vermischen."""
    print("\nSVGD-Gruppierung")
    names, truths = _welt(4)
    planer = ErsatzPlaner()
    args = M.build_mission_args('cpu', phi_model='niveau', param=0.5)
    gerufen = []

    echt = M.refine_batch

    def spion(curves, phis, n_iters, **kw):
        gerufen.append((int(n_iters), int(curves.shape[0])))
        return echt(curves, phis, 0, **kw)      # ohne echte Iterationen

    M.refine_batch = spion
    try:
        m = M.LaengenMission(planer, truths, names, args, seed=0,
                             policy=lambda r, z: [('niveau', 0.5, 0),
                                                  ('niveau', 0.5, 25),
                                                  ('niveau', 0.5, 25),
                                                  ('niveau', 0.5, 100)])
        m.run(1)
    finally:
        M.refine_batch = echt
    check("ein Aufruf je vorkommendem Budget", len(gerufen) == 3, str(gerufen))
    check("Gruppengroessen stimmen",
          sorted(gerufen) == [(0, 1), (25, 2), (100, 1)], str(sorted(gerufen)))


def test_belohnung_ist_minus_J():
    """Die Summe der Belohnungen muss `1 - J` sein — sonst optimiert Option B
    etwas anderes als die Studie."""
    print("\nBelohnung der RL-Umgebung")
    names, truths = _welt(3)
    planer = ErsatzPlaner()
    args = M.build_mission_args('cpu', phi_model='niveau', param=0.5)
    n_max = 3
    env = MissionUmgebung(planer, truths, names, args, n_max=n_max, seed=0)
    env.reset(indizes=[0, 1, 2])

    aktion = ('niveau', 0.5, 25)
    ertrag = np.zeros(3)
    zeilen = []
    for _ in range(n_max):
        _b, r, fertig, info = env.step([aktion] * 3)
        ertrag += r
        zeilen += info['zeilen']

    q_ende = np.mean([z['cov_norm'] for z in zeilen if z['n_exec'] == n_max])
    j_erwartet = (q_ende + DEFAULT_LAMBDA_LEN * n_max
                  + DEFAULT_LAMBDA_TIME * zeit_kosten(25) * n_max)
    check("Summe der Belohnungen = 1 - J",
          abs((1.0 - ertrag.mean()) - j_erwartet) < 1e-6,
          f"1-R = {1.0 - ertrag.mean():.6f}, J = {j_erwartet:.6f}")
    check("Episode endet nach n_max Runden", fertig)


def test_gae():
    print("\nVorteilsschaetzer")
    rew = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    wert = np.zeros((3, 1), dtype=np.float32)
    fertig = np.array([[0.0], [0.0], [1.0]], dtype=np.float32)
    vorteil, ziel = gae(rew, wert, fertig, gamma=1.0, lam=1.0)
    # Ohne Wertfunktion und ohne Abzinsung ist der Vorteil die Restsumme.
    check("Vorteil = Summe der restlichen Belohnungen",
          np.allclose(vorteil.ravel(), [3.0, 2.0, 1.0]), str(vorteil.ravel()))
    check("Ziel = Vorteil + Wert", np.allclose(ziel, vorteil + wert))


def test_richtlinien_ablage(tmp='.'):
    print("\nAblage der Regler")
    raster = aktionsraster(['niveau', 'ucb'], 3, (0, 25))
    netz = WertNetz()
    X = np.random.RandomState(0).rand(50, len(F.MERKMALE)).astype(np.float32)
    mittel, streuung = normierung(X)
    r1 = WertRichtlinie(netz, mittel, streuung, raster)
    pfad = os.path.join(tmp, '_test_policy_a.pt')
    r1.speichern(pfad)
    r2 = WertRichtlinie.laden(pfad)
    names, truths = _welt(2)
    z = [Zustand(mu=truths[i] * 0.5, sd=torch.ones(32, 32) * 0.3, visit=None,
                 driven=None, runde=0, n_max=6) for i in range(2)]
    a1, a2 = r1(0, z), r2(0, z)
    check("A: Wahl ueberlebt Speichern und Laden", a1 == a2, str(a1))
    check("A: eine Aktion je Form", len(a1) == 2)
    check("A: Aktion stammt aus dem Raster", all(a in raster for a in a1))
    os.remove(pfad)

    pol = HybridPolitik()
    norm = Normierer(len(F.ZUSTANDS_MERKMALE))
    norm.aktualisiere(np.random.RandomState(1).rand(
        20, len(F.ZUSTANDS_MERKMALE)))
    b1 = RLRichtlinie(pol, norm)
    pfad_b = os.path.join(tmp, '_test_policy_b.pt')
    b1.speichern(pfad_b)
    b2 = RLRichtlinie.laden(pfad_b)
    check("B: Wahl ueberlebt Speichern und Laden", b1(0, z) == b2(0, z),
          str(b1(0, z)))
    check("B: Kategorie <-> (Modell, SVGD) ist umkehrbar",
          all(pol.kategorie(*pol.zerlege(d)) == d for d in range(pol.n_disk)))
    os.remove(pfad_b)


def test_hybrid_politik():
    print("\nPolitiknetz (Option B)")
    pol = HybridPolitik()
    x = torch.randn(7, len(F.ZUSTANDS_MERKMALE))
    aktionen, logp, wert, d, z = pol.handeln(x)
    check("eine Aktion je Zeile", len(aktionen) == 7)
    check("Aktionen sind gueltig",
          all(a[0] in F.MODELL_ORDNUNG and a[2] in pol.svgd_buckets
              for a in aktionen), str(aktionen[0]))
    check("Parameter liegt im erlaubten Bereich",
          all(F.PARAM_RANGE[F.PHI_MODELS[a[0]][0]][0] - 1e-9 <= a[1]
              <= F.PARAM_RANGE[F.PHI_MODELS[a[0]][0]][1] + 1e-9
              for a in aktionen))
    check("Log-Wahrscheinlichkeit und Wert haben die richtige Form",
          logp.shape == (7,) and wert.shape == (7,))
    lp2, ent, w2 = pol.bewerte(x, d, z)
    check("bewerte() reproduziert die Log-Wahrscheinlichkeit",
          torch.allclose(logp, lp2, atol=1e-5))
    check("Entropie ist positiv", bool((ent > 0).all()))
    gierig1 = pol.handeln(x, gierig=True)[0]
    gierig2 = pol.handeln(x, gierig=True)[0]
    check("gierige Auswertung ist deterministisch", gierig1 == gierig2)


def test_zeitbudget():
    """Ein langer Lauf muss geordnet aufhoeren, nicht abgeschnitten werden."""
    print("\nZeitbudget")
    from .budget import Zeitbudget
    b = Zeitbudget(max_minuten=10.0, reserve_min=0.0, name='Test')
    check("frisches Budget ist nicht abgelaufen", not b.abgelaufen())
    check("ein Schritt, der nicht mehr hineinpasst, beendet vorher",
          b.abgelaufen(naechster_schritt_sek=10 * 60))
    b.signal_empfangen = True
    check("SIGTERM beendet unabhaengig von der Uhr", b.abgelaufen())
    check("Grund wird benannt", 'SIGTERM' in b.grund(), b.grund())
    b.loesen()
    ohne = Zeitbudget(max_minuten=None, name='ohne')
    check("ohne Limit laeuft es weiter", not ohne.abgelaufen(1e9))
    check("Restzeit ohne Limit ist None", ohne.rest() is None)
    ohne.loesen()

    # Und im echten Rollout: ein knappes Budget kuerzt die Mission, statt sie
    # mittendrin abzuschneiden. Die erste Runde laeuft immer — vor ihr ist
    # ihre Dauer nicht bekannt, und eine Schaetzung waere geraten.
    from .oracle import orakel_rollout
    names, truths = _welt(2)
    args = M.build_mission_args('cpu', phi_model='niveau', param=0.5)
    kand = aktionsraster(['niveau'], 2, (0,))
    knapp = Zeitbudget(max_minuten=0.001, reserve_min=0.0, name='knapp')
    zeilen, gewaehlt, _m = orakel_rollout(ErsatzPlaner(), truths, names, args,
                                          kand, 4, budget=knapp)
    knapp.loesen()
    runden = {z['n_exec'] for z in zeilen}
    check("knappes Budget kuerzt die Mission (weniger als n_max Runden)",
          0 < len(runden) < 4 and runden == set(range(1, max(runden) + 1)),
          f"gefahrene Runden: {sorted(runden)} statt 1..4")
    check("jede begonnene Runde ist vollstaendig (alle Formen)",
          len(zeilen) == 2 * len(runden), f"{len(zeilen)} Zeilen")
    check("die gekuerzte Spur bleibt auswertbar",
          O.score_trace(zeilen)[0] is not None)
    check("gewaehlte Aktionen passen zur Rundenzahl",
          all(len(v) == len(runden) for v in gewaehlt.values()))


def test_gui_hook():
    """Der Knopf in der GUI muss `args` wirklich umstellen — sonst waehlt der
    Regler etwas anderes, als die Mission dann faehrt."""
    print("\nGUI-Anbindung (interactive_sim.Mission)")
    import argparse as _ap
    import importlib.util
    pfad = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), 'exploration', 'interactive_sim.py')
    if not os.path.exists(pfad):
        check("interactive_sim.py gefunden", False, pfad)
        return
    # Nur die Klasse laden, ohne die GUI zu starten: das Modul zieht
    # matplotlib-Fenster erst in `main()` hoch.
    spec = importlib.util.spec_from_file_location('_isim_test', pfad)
    modul = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(modul)
    except Exception as exc:                                   # noqa: BLE001
        check("interactive_sim laedt", False, str(exc)[:80])
        return
    check("interactive_sim laedt", True)

    args = _ap.Namespace(kappa0=1.0, phi_tau=0.25, phi_model='ucb',
                         svgd_iters=0, noise=0.02, sensor_radius=0.06,
                         max_obs=64)

    class Glaube:
        n_obs = 12

    m = modul.Mission.__new__(modul.Mission)
    m.args = args
    m.belief = Glaube()
    m.driven = []
    m.policy = lambda r, z: [('niveau', 0.42, 25)]
    m.last_action = None
    mu = torch.rand(32, 32) * 0.4
    sd = torch.ones(32, 32) * 0.3
    aktion = m._apply_policy(0, 8, mu, sd, None)
    check("Aktion wird zurueckgegeben", aktion == ('niveau', 0.42, 25),
          str(aktion))
    check("Phi-Modell steht intern richtig ('lse' fuer niveau)",
          args.phi_model == 'lse', args.phi_model)
    check("tau kommt an", abs(args.phi_tau - 0.42) < 1e-9)
    check("SVGD-Budget kommt an", args.svgd_iters == 25)

    m.policy = lambda r, z: [('mass', 0.6, 0)]
    m._apply_policy(1, 8, mu, sd, None)
    check("mass wird als kappa = w/(1-w) uebersetzt",
          abs(args.kappa0 - 0.6 / 0.4) < 1e-6, f"kappa0={args.kappa0:.4f}")
    check("SVGD zurueck auf 0", args.svgd_iters == 0)


def test_ganze_kette(tmp):
    """Orakel -> Datensatz -> Option A -> Verhaltensklonen -> PPO -> Auswertung,
    komplett mit dem Ersatzplaner. Prueft, dass die Stufen wirklich
    zusammenpassen (Spaltennamen, Kandidatenraster, Modellablage) — der Teil,
    der beim Zusammenbau schiefgeht und den keine Einzelpruefung faengt."""
    print("\nGanze Kette (Ersatzplaner, kein Netz)")
    from . import oracle as ORC
    from . import train as TR
    from . import ppo as PP
    from . import evaluate as EV
    from .model import FesteRichtlinie

    names, truths = _welt(4)
    args = M.build_mission_args('cpu', phi_model='niveau', param=0.6)
    kandidaten = aktionsraster(['niveau', 'ucb'], 3, (0, 25))

    # 1. Orakel mit Datensatz
    sammler = []
    zeilen, gewaehlt, _m = ORC.orakel_rollout(
        ErsatzPlaner(), truths, names, args, kandidaten, 3, seed=0,
        sammler=sammler)
    check("Orakel liefert eine Zeile je Form und Runde",
          len(zeilen) == 4 * 3, f"{len(zeilen)}")
    check("Datensatz hat eine Zeile je Kandidat und Entscheidung",
          len(sammler) == 4 * 3 * len(kandidaten), f"{len(sammler)}")
    check("je Entscheidung genau ein bester Kandidat",
          sum(z['ist_bester'] for z in sammler) == 4 * 3)
    check("score_norm liegt in [0,1]",
          all(-1e-9 <= z['score_norm'] <= 1 + 1e-9 for z in sammler))
    check("Spur ist mit objective.py auswertbar",
          O.score_trace(zeilen)[0]['n_exec'] >= 1)

    pfad_csv = os.path.join(tmp, '_kette_datensatz.csv')
    ORC.schreibe_datensatz(sammler, pfad_csv)
    check("Datensatz laesst sich schreiben und wieder lesen",
          os.path.getsize(pfad_csv) > 0)

    # 2. Option A
    X, y, gruppen, entscheidung, kand = TR.lade_datensatz(pfad_csv)
    check("A: Merkmalsmatrix passt zur Merkmalsliste",
          X.shape[1] == len(F.MERKMALE), f"{X.shape}")
    falten = TR.formen_falten(gruppen, 2)
    check("A: Falten trennen ganze Formen",
          all(set(gruppen[m_].tolist()) & set(gruppen[~m_].tolist()) == set()
              for m_ in falten))
    idx_val = np.where(falten[0])[0]
    idx_tr = np.where(~falten[0])[0]
    netz, (mittel, streuung), _v = TR.trainiere(X, y, idx_tr, idx_val, 'cpu',
                                                epochen=15, still=True)
    vor = TR._vorhersage(netz, X, mittel, streuung, 'cpu')
    w_modell = TR.wahl_nach_wert(vor, entscheidung, idx_val)
    w_fest = TR.wahl_fest(kand, entscheidung, idx_val)
    check("A: je Entscheidung wird genau eine Zeile gewaehlt",
          len(w_modell) == len(set(entscheidung[idx_val].tolist())))
    check("A: die feste Einstellung ist im Raster auffindbar", bool(w_fest))
    check("A: Bedauern liegt in [0,1]",
          0 <= TR.bedauern(y, entscheidung, w_modell) <= 1)
    wicht = TR.wichtigkeiten(netz, X, y, entscheidung, idx_val, mittel,
                             streuung, 'cpu', wiederholungen=1)
    check("A: Wichtigkeit fuer jedes Merkmal", len(wicht) == len(F.MERKMALE))

    richtlinie = WertRichtlinie(netz, mittel, streuung, kandidaten)
    pfad_a = os.path.join(tmp, '_kette_a.pt')
    richtlinie.speichern(pfad_a)

    # 3. Option B: Verhaltensklonen und zwei PPO-Iterationen
    Xb, d, u = PP.lade_bc_daten(pfad_csv, (0, 25), F.MODELL_ORDNUNG)
    check("B: Klondaten sind die besten Aktionen",
          len(Xb) == 4 * 3, f"{len(Xb)}")
    politik = PP.HybridPolitik(svgd_buckets=(0, 25))
    norm = PP.Normierer(len(F.ZUSTANDS_MERKMALE))
    PP.verhaltensklonen(politik, norm, Xb, d, u, 'cpu', epochen=10, still=True)
    env = MissionUmgebung(ErsatzPlaner(), truths, names, args, n_max=2,
                          n_envs=2, seed=0)
    opt = torch.optim.AdamW(politik.parameters(), lr=1e-3)
    puffer, ergebnisse = PP.sammle(env, politik, norm, 'cpu', episoden=2)
    check("B: Puffer hat die Form (Schritte, Formen)",
          puffer['rew'].shape == (4, 2), str(puffer['rew'].shape))
    vorteil, ziel = PP.gae(puffer['rew'], puffer['wert'], puffer['fertig'])
    vorher = [p.detach().clone() for p in politik.parameters()]
    stat = PP.ppo_schritt(politik, opt, puffer, vorteil, ziel, 'cpu', epochen=1)
    veraendert = any(not torch.equal(a, b) for a, b in
                     zip(vorher, politik.parameters()))
    check("B: ein PPO-Schritt aendert die Politik", veraendert)
    check("B: Verlustwerte sind endlich",
          all(np.isfinite(v) for v in stat.values()), str(stat))
    pfad_b = os.path.join(tmp, '_kette_b.pt')
    PP.RLRichtlinie(politik, norm).speichern(pfad_b)

    # 4. Auswertung mit allen drei Reglern durch denselben Simulator
    regler = {
        'fest': FesteRichtlinie('niveau', 0.6067, 0),
        'gelernt_a': WertRichtlinie.laden(pfad_a),
        'rl_b': PP.RLRichtlinie.laden(pfad_b),
    }
    tabellen, bahnen, grenzen_je, q_form, alle = {}, {}, {}, {}, []
    for name, pol in regler.items():
        zeilen_r, mission, grenzen = EV.fahre(ErsatzPlaner(), truths, names,
                                              args, pol, 2, seed=0)
        for z in zeilen_r:
            z['regler'] = name
        alle += zeilen_r
        best, tab = O.score_trace(zeilen_r)
        tabellen[name] = tab
        bahnen[name] = list(mission.driven)
        grenzen_je[name] = grenzen
        q_form[name] = [float(np.mean([z['cov_norm'] for z in zeilen_r
                                       if z['shape'] == nm and z['n_exec'] == 2]))
                        for nm in names]
    check("Auswertung: alle drei Regler liefern Spuren",
          len(tabellen) == 3 and all(t for t in tabellen.values()))
    check("Auswertung: jede Form hat eine Bahn je Regler",
          all(len(b) == 4 for b in bahnen.values()))
    check("Auswertung: die gewaehlte Aktion steht in jeder Zeile",
          all('gewaehlt_modell' in z for z in alle))

    alt = os.getcwd()
    try:
        os.chdir(tmp)
        from .. import plots as PL
        echt_dir = PL.RESULTS_DIR
        PL.RESULTS_DIR = tmp
        try:
            p1 = EV.panel(names, truths, bahnen, grenzen_je, q_form, 2,
                          tag='_kette')
            p2 = EV.kurven(tabellen, tag='_kette')
            p3 = EV.aktionen({k: [dict(runde=z['n_exec'] - 1,
                                       modell=z['gewaehlt_modell'],
                                       param=z['gewaehlt_param'],
                                       svgd=z['gewaehlt_svgd'])
                                  for z in alle if z['regler'] == k]
                              for k in regler}, tag='_kette')
        finally:
            PL.RESULTS_DIR = echt_dir
    finally:
        os.chdir(alt)
    check("Auswertung: alle drei Abbildungen entstehen",
          all(p and os.path.exists(p) for p in (p1, p2, p3)))

    for p in (pfad_csv, pfad_a, pfad_b, p1, p2, p3):
        if p and os.path.exists(p):
            os.remove(p)


def main():
    import tempfile
    print("Selbsttest der gelernten Regler (ohne Netz, ohne GPU)")
    torch.manual_seed(0)
    np.random.seed(0)
    test_merkmale_ohne_wahrheit()
    test_parameter_umkehr()
    test_hook_faehrt_die_aktion()
    test_svgd_gruppierung()
    test_belohnung_ist_minus_J()
    test_gae()
    test_hybrid_politik()
    test_zeitbudget()
    with tempfile.TemporaryDirectory() as tmp:
        test_richtlinien_ablage(tmp)
        test_ganze_kette(tmp)
    test_gui_hook()
    print("\n" + ("Alles in Ordnung." if not _FEHLER
                  else f"{len(_FEHLER)} Fehler: {', '.join(_FEHLER)}"))
    return 1 if _FEHLER else 0


if __name__ == '__main__':
    sys.exit(main())
