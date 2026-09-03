#!/usr/bin/env python3
r"""
test_smoke.py
=============
Prueft die Teile, an denen dieser Ordner still falsch rechnen koennte.

Es geht nicht um Abdeckung um ihrer selbst willen, sondern um die Annahmen,
auf denen die ganze Studie steht und die man einem Ergebnis spaeter nicht mehr
ansieht:

* Wird wirklich **eine** Laengeneinheit gefahren, und ist sie die Diagonale?
* Setzt jede Runde exakt dort an, wo die vorige endete? (Ohne das faehrt der
  Roboter Spruenge und die gebuchte Weglaenge ist zu klein.)
* Bekommt jede Form beim gebatchten Planen ihre *eigene* Bahn? Ein
  Zeilenversatz zwischen Partikelwolke und Ergebnis waere die teuerste Art,
  sich zu irren: alles laeuft durch, alle Zahlen sind Unsinn.
* Verhaelt sich die Zieldichte so, wie die Aufgabe es verlangt — erkundet und
  leer faellt auf null, erkundet und voll faellt nur anteilig?
* Liefert eine Spur wirklich alle Rundenzahlen, und ist die Praefix-Annahme
  der Zeitrechnung richtig?

    python -m exploration_optimierung.test_smoke
    python -m exploration_optimierung.test_smoke --schnell   (ohne Netz)
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = 'exploration_optimierung'

from . import DEFAULT_CKPT                      # noqa: E402
from . import mission as M                      # noqa: E402
from . import objective as O                    # noqa: E402

import apply_cfm_belief as acb                  # noqa: E402
from common.belief import GPBelief              # noqa: E402
from common.metrics import path_length          # noqa: E402

OK, FAIL = [], []


def check(name, cond, info=''):
    (OK if cond else FAIL).append(name)
    print(f"  [{'ok ' if cond else 'FEHL'}] {name}" + (f"   {info}" if info else ''))


# ---------------------------------------------------------------------------

def test_laengeneinheit():
    print("\n-- Laengeneinheit und Neuabtastung --")
    check("Laengeneinheit ist die Diagonale von [0,1]^2",
          abs(M.LENGTH_UNIT - math.sqrt(2.0)) < 1e-12,
          f"{M.LENGTH_UNIT:.6f}")

    # Eine Bahn, die deutlich laenger ist als eine Laengeneinheit.
    t = torch.linspace(0, 1, 400)
    curve = torch.stack([0.5 + 0.45 * torch.cos(8 * t),
                         0.5 + 0.45 * torch.sin(8 * t)], dim=-1)
    from common.metrics import trim_to_length
    seg = trim_to_length(curve, M.LENGTH_UNIT)
    check("Zuschnitt trifft die Laengeneinheit auf 1e-4",
          abs(path_length(seg) - M.LENGTH_UNIT) < 1e-4,
          f"{path_length(seg):.6f}")

    rs = M.resample_arclength(seg, 72)
    check("Neuabtastung erhaelt die Weglaenge",
          abs(path_length(rs) - path_length(seg)) < 2e-3,
          f"{path_length(rs):.6f}")
    d = (rs[1:] - rs[:-1]).norm(dim=-1)
    check("Neuabtastung ist gleichmaessig (Streuung < 1 % des Mittels)",
          float(d.std() / d.mean()) < 0.01, f"cv={float(d.std()/d.mean()):.5f}")
    check("Neuabtastung haelt Anfang und Ende fest",
          float((rs[0] - seg[0]).norm()) < 1e-5 and
          float((rs[-1] - seg[-1]).norm()) < 1e-5)


def test_parameter_uebersetzung():
    print("\n-- Uebersetzung des freien Parameters --")
    for w in (0.1, 0.5, 0.85):
        a = M.build_mission_args('cpu', 'mass', w)
        back = a.kappa / (1.0 + a.kappa)     # was `zieldichte` daraus macht
        check(f"mass: w={w} kommt als w={back:.4f} an", abs(back - w) < 1e-6)
    a = M.build_mission_args('cpu', 'niveau', 0.37)
    check("niveau: tau geht als phi_tau durch", abs(a.phi_tau - 0.37) < 1e-9)
    a = M.build_mission_args('cpu', 'ucb', 2.5)
    check("ucb: kappa geht direkt durch", abs(a.kappa - 2.5) < 1e-9)
    a = M.build_mission_args('cpu', 'eid', 2.5)
    check("eid: internes Modell ist 'eid'", a.phi_model == 'eid')


def test_zieldichte_verhalten():
    print("\n-- Verhalten der Zieldichte (der Kern der Aufgabe) --")
    R = 32
    a = M.build_mission_args('cpu', 'ucb', 3.0, debt_weight=0.6)
    a_gui = M.build_mission_args('cpu', 'ucb', 3.0, debt_weight=0.6,
                                 visit_sat=0.25)

    # Drei Gebiete: (1) besucht und leer, (2) besucht und voll, (3) unbesucht.
    mu = torch.zeros(R, R)
    mu[8:14, 8:14] = 0.0          # besucht, leer
    mu[8:14, 20:26] = 1.0         # besucht, voll
    sd = torch.full((R, R), 1.0)
    sd[8:14, 8:14] = 0.02         # dort wurde gemessen
    sd[8:14, 20:26] = 0.02
    visit = torch.zeros(R, R)
    visit[8:14, 8:14] = 1.0
    visit[8:14, 20:26] = 1.0

    phi, v = acb.debt_density(mu, sd, visit, a.kappa, a)
    leer = float(phi[8:14, 8:14].mean())
    voll = float(phi[8:14, 20:26].mean())
    unbe = float(phi[24:30, 24:30].mean())

    check("erkundet + leer faellt praktisch auf null", leer < 0.02,
          f"phi={leer:.4f}")
    check("erkundet + voll faellt nur anteilig, bleibt sichtbar",
          0.05 < voll < 0.9, f"phi={voll:.4f}")
    check("erkundet + voll bleibt ueber erkundet + leer", voll > 5 * leer,
          f"{voll:.4f} vs {leer:.4f}")
    check("unbesucht zieht am staerksten", unbe > voll,
          f"phi={unbe:.4f}")

    # Proportionalitaet: zwei besuchte, gefundene Gebiete verschiedener Staerke
    mu2 = torch.zeros(R, R); sd2 = torch.full((R, R), 1.0)
    vis2 = torch.zeros(R, R)
    mu2[4:10, 4:10] = 1.0; sd2[4:10, 4:10] = 0.02; vis2[4:10, 4:10] = 1.0
    mu2[4:10, 16:22] = 0.4; sd2[4:10, 16:22] = 0.02; vis2[4:10, 16:22] = 1.0
    phi2, _ = acb.debt_density(mu2, sd2, vis2, a.kappa, a)
    stark = float(phi2[4:10, 4:10].mean())
    schwach = float(phi2[4:10, 16:22].mean())
    verh = stark / max(schwach, 1e-9)
    check("Restanziehung bleibt proportional zur Intensitaet (1.0 : 0.4)",
          2.0 < verh < 3.2, f"Verhaeltnis {verh:.3f}, erwartet ~2.5")

    # Erholung. Sie kommt *nicht* aus der Normierung auf das Besuchsmaximum in
    # `debt_density`: die verlangt, dass anderswo um ein Vielfaches laenger
    # verweilt wird, und auf einer gleichmaessig abgefahrenen Bahn passiert das
    # nie -- der Nachweis dafuer steht gleich unten ("ohne Altersgewichtung").
    # Sie kommt aus der Altersgewichtung in `visitation_recent`.
    # Eine Bahn im Massstab einer echten Mission: sechs Ausfuehrungen zu je
    # einer Laengeneinheit, maeandernd von links nach rechts. Das zuerst
    # befahrene Gebiet muss bei der voreingestellten Halbwertszeit deutlich
    # schwaecher gesperrt sein als das zuletzt befahrene.
    n_r, per = 6, 120
    stuecke = []
    for r in range(n_r):
        y = 0.15 + 0.7 * r / (n_r - 1)
        xs = torch.linspace(0.08, 0.92, per)
        if r % 2:
            xs = xs.flip(0)
        stuecke.append(torch.stack([xs, torch.full((per,), y)], dim=-1))
    bahn = torch.cat(stuecke, dim=0)
    hl = 3.0 * M.LENGTH_UNIT                      # Voreinstellung des Runners
    gleich = acb.visitation_field(bahn, R, 0.06, 'cpu')
    alt = M.visitation_recent(bahn, R, 0.06, 'cpu', half_life=hl)

    def bei(feld, x, y):
        j = int(round(x * (R - 1))); i = int(round(y * (R - 1)))
        return float(feld[i, j])

    y0, y5 = 0.15, 0.85
    check("ohne Altersgewichtung sind erste und letzte Runde gleich gesperrt",
          abs(bei(gleich, 0.5, y0) - bei(gleich, 0.5, y5))
          < 0.15 * bei(gleich, 0.5, y5),
          f"{bei(gleich,0.5,y0):.2f} vs {bei(gleich,0.5,y5):.2f}")
    verh = bei(alt, 0.5, y0) / max(bei(alt, 0.5, y5), 1e-9)
    check("mit Altersgewichtung ist die erste Runde deutlich schwaecher "
          "gesperrt als die letzte",
          verh < 0.5, f"Verhaeltnis {verh:.3f}")
    check("half_life<=0 ergibt exakt die ungewichtete Aufenthaltsdichte",
          torch.allclose(M.visitation_recent(bahn, R, 0.06, 'cpu', half_life=0),
                         gleich, atol=1e-6))

    # Und die Wirkung auf die Zieldichte: ein frueh befahrenes, dichtes Gebiet
    # muss wieder anziehender sein als ein eben befahrenes gleicher Dichte.
    mu3 = torch.zeros(R, R); sd3 = torch.full((R, R), 0.02)
    i0 = int(round(y0 * (R - 1))); i5 = int(round(y5 * (R - 1)))
    mu3[i0 - 1:i0 + 2, :] = 1.0
    mu3[i5 - 1:i5 + 2, :] = 1.0
    phi_alt, _ = acb.debt_density(mu3, sd3, alt, a.kappa, a)
    check("frueh befahrenes Dichtegebiet zieht wieder staerker als eben "
          "befahrenes",
          bei(phi_alt, 0.5, y0) > 1.3 * bei(phi_alt, 0.5, y5),
          f"{bei(phi_alt,0.5,y0):.4f} vs {bei(phi_alt,0.5,y5):.4f}")
    phi_sat, _ = acb.debt_density(mu3, sd3, alt, a_gui.kappa, a_gui)
    check("bei visit_sat=0.25 bleibt die Alterung unsichtbar (deshalb 1.0)",
          abs(bei(phi_sat, 0.5, y0) - bei(phi_sat, 0.5, y5)) < 1e-6,
          f"{bei(phi_sat,0.5,y0):.4f} == {bei(phi_sat,0.5,y5):.4f}")


def test_praefix_und_bewertung():
    print("\n-- Spur, Praefix-Eigenschaft und Zielfunktion --")
    rows = []
    for n in range(1, 6):
        for s in ('A', 'B'):
            rows.append({'shape': s, 'n_exec': n,
                         'cov': 0.3 / n, 'cov_norm': 0.9 / n,
                         'erg_truth': 0.01 / n, 'belief_rmse': 0.2 / n,
                         'info_gain': 10.0 * n, 'path_len': n * M.LENGTH_UNIT,
                         'n_obs': 64 * n, 'plan_s': 50.0, 'svgd_s': 25.0})
    tbl = O.per_n(rows)
    check("jede Rundenzahl kommt in der Spur vor",
          [t['n_exec'] for t in tbl] == [1, 2, 3, 4, 5])
    check("Zeit waechst linear mit der Rundenzahl",
          abs(tbl[0]['time_s'] - 75.0 / 5) < 1e-9 and
          abs(tbl[4]['time_s'] - 75.0) < 1e-9,
          f"{tbl[0]['time_s']:.1f} .. {tbl[4]['time_s']:.1f}")

    best, tab = O.score_trace(rows, lambda_len=0.02, lambda_time=0.0)
    check("J beruecksichtigt die Wegstrafe",
          abs(tab[2]['J'] - (0.9 / 3 + 0.06)) < 1e-9)
    check("beste Rundenzahl ist die mit dem kleinsten J",
          best['n_exec'] == min(tab, key=lambda r: r['J'])['n_exec'],
          f"n={best['n_exec']}")

    hoch, _ = O.score_trace(rows, lambda_len=0.5, lambda_time=0.0)
    check("hoehere Wegstrafe waehlt weniger Ausfuehrungen",
          hoch['n_exec'] <= best['n_exec'],
          f"n={hoch['n_exec']} statt {best['n_exec']}")
    front = O.pareto(tab)
    check("Pareto-Front ist in q streng fallend",
          all(front[i + 1]['q'] < front[i]['q'] for i in range(len(front) - 1)))


def test_mission(device, ckpt, n_shapes=3, n_max=3, svgd_iters=0):
    print(f"\n-- Vollstaendige Mission ({n_shapes} Formen, {n_max} Runden, "
          f"svgd={svgd_iters}) --")
    names, truths = M.load_holdout(resolution=96, device=device, limit=n_shapes)
    check("Holdout-Formen geladen", len(names) == n_shapes, ', '.join(names))
    planner = M.build_planner(ckpt=ckpt, device=device)
    check("Checkpoint ist startpunkt-konditioniert", planner.start_cond)

    rows, m = M.run_config(planner, truths, names, 'ucb', 3.0, n_max,
                           svgd_iters=svgd_iters, seed=0)
    check("eine Zeile je Form und Rundenzahl",
          len(rows) == n_shapes * n_max, f"{len(rows)} Zeilen")

    for i in range(n_shapes):
        per = [r for r in rows if r['shape'] == names[i]]
        laengen = [r['path_len'] for r in sorted(per, key=lambda r: r['n_exec'])]
        schritte = np.diff([0.0] + laengen)
        check(f"{names[i]}: jede Runde faehrt eine Laengeneinheit",
              all(abs(s - M.LENGTH_UNIT) < 0.05 for s in schritte),
              ' '.join(f"{s:.3f}" for s in schritte))
        check(f"{names[i]}: Abdeckungsfehler faellt",
              per[-1]['cov'] < per[0]['cov'],
              f"{per[0]['cov']:.4f} -> {per[-1]['cov']:.4f}")
        check(f"{names[i]}: bezogener Fehler liegt in (0, 1.2]",
              0.0 < per[-1]['cov_norm'] <= 1.2, f"{per[-1]['cov_norm']:.3f}")

    # Bahn ist zusammenhaengend: keine Spruenge zwischen den Runden.
    for i in range(n_shapes):
        d = (m.driven[i][1:] - m.driven[i][:-1]).norm(dim=-1)
        check(f"{names[i]}: Bahn ohne Sprung (groesster Schritt < 0.1)",
              float(d.max()) < 0.1, f"max {float(d.max()):.4f}")
        check(f"{names[i]}: Bahn bleibt im Einheitsquadrat",
              float(m.driven[i].min()) >= -1e-4 and
              float(m.driven[i].max()) <= 1 + 1e-4)
    return names, truths, planner


def test_batch_zuordnung(device, ckpt):
    r"""Bekommt Zeile i wirklich die Bahn zu Wolke i?

    Der Test stellt zwei Wolken auf, deren Traeger sich nicht ueberlappen
    (linkes und rechtes Fuenftel), und prueft, dass die zugehoerige Bahn
    tatsaechlich auf ihrer Seite liegt. Waeren die Zeilen vertauscht, liefe
    trotzdem alles durch — nur waeren saemtliche Zahlen der Studie falsch.
    """
    print("\n-- Zuordnung beim gebatchten Planen --")
    planner = M.build_planner(ckpt=ckpt, device=device)
    R = 64
    links = torch.zeros(R, R, device=device); links[:, :R // 5] = 1.0
    rechts = torch.zeros(R, R, device=device); rechts[:, -R // 5:] = 1.0
    parts = torch.stack([acb.phi_particles(links, 256, device=device),
                         acb.phi_particles(rechts, 256, device=device)])
    starts = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device=device)
    with torch.no_grad():
        curves = planner.render(planner.plan(parts, n_candidates=2, start=starts))
    mx = [float(c[:, 0].mean()) for c in curves]
    check("Bahn 0 folgt der linken Wolke, Bahn 1 der rechten",
          mx[0] < 0.5 < mx[1], f"mittleres x: {mx[0]:.3f} / {mx[1]:.3f}")
    check("beide Startpunkte sitzen exakt",
          all(float((curves[i, 0] - starts[i]).norm()) < 1e-5 for i in range(2)))


def test_svgd(device, ckpt):
    r"""Zieht die Verfeinerung die Bahn zur Zieldichte, und zwar monoton?

    Geprueft wird nicht ein einzelner Zahlenwert nach fester Iterationszahl —
    der haengt an der Schrittweite und sagt fuer sich genommen nichts —,
    sondern die Eigenschaft, auf die sich die Suche ueber `--svgd_iters`
    stuetzt: **mehr Iterationen bringen die Bahn naeher an die Zieldichte.**
    Waere das nicht so, waere die Iterationszahl kein sinnvoller Suchparameter.
    """
    print("\n-- SVGD-Verfeinerung --")
    from common.svgd_refine import SvgdRefiner
    R = 64
    phi = np.zeros((R, R))
    phi[R // 3:2 * R // 3, R // 3:2 * R // 3] = 1.0      # Block um y = 0.5
    curve = np.stack([np.linspace(0.05, 0.95, 256),
                      np.full(256, 0.05)], axis=-1)      # Linie am unteren Rand

    ys = []
    for n in (0, 60, 200, 400):
        out = SvgdRefiner(seed=0).refine(curve.copy(), phi, n, nxi=25)
        check(f"n_iters={n}: Verfeinerung liefert dieselbe Form",
              out.shape == (256, 2))
        ys.append(float(out[:, 1].mean()))
    check("n_iters=0 ist ein reiner Durchreicher", abs(ys[0] - 0.05) < 1e-9,
          f"y={ys[0]:.4f}")
    check("mehr Iterationen ziehen die Bahn weiter zur Zieldichte",
          all(ys[i + 1] > ys[i] + 0.02 for i in range(len(ys) - 1)),
          " -> ".join(f"{y:.3f}" for y in ys))
    check("bei 400 Iterationen ist die Bahn im Traeger der Zieldichte "
          "angekommen", ys[-1] > 0.33, f"y={ys[-1]:.3f} (Traeger ab 0.33)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--schnell', action='store_true',
                   help='nur die Teile ohne Netz und ohne GPU')
    p.add_argument('--ckpt', default=DEFAULT_CKPT)
    p.add_argument('--device', default=None)
    p.add_argument('--svgd', type=int, default=30,
                   help='SVGD-Iterationen in der vollstaendigen Mission')
    a = p.parse_args()

    print("=== Selbsttest exploration_optimierung ===")
    test_laengeneinheit()
    test_parameter_uebersetzung()
    test_zieldichte_verhalten()
    test_praefix_und_bewertung()

    if not a.schnell:
        device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n  Geraet: {device}, Checkpoint: {os.path.basename(a.ckpt)}")
        test_batch_zuordnung(device, a.ckpt)
        test_svgd(device, a.ckpt)
        test_mission(device, a.ckpt, n_shapes=3, n_max=3, svgd_iters=0)
        test_mission(device, a.ckpt, n_shapes=2, n_max=2, svgd_iters=a.svgd)

    print(f"\n=== {len(OK)} bestanden, {len(FAIL)} fehlgeschlagen ===")
    for f in FAIL:
        print(f"  FEHLGESCHLAGEN: {f}")
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
