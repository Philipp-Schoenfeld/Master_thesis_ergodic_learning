#!/usr/bin/env python3
r"""
run_length.py
=============
Experiment 2: **Rekonstruktion nach Laenge** — wie gut deckt eine Bahn die
Zieldichte ab, wenn ihr ein Wegbudget vorgegeben wird, und welcher
Steuermechanismus haelt dieses Budget ueberhaupt ein?

Vier Arme, ein Sampler
----------------------
    frei         keine Laengenvorgabe                       (Referenz)
    kond         gelernter FiLM-Laengenkanal des Checkpoints
    kraft        Inferenz-Kraft `TargetLength` (constraints/04_path_length)
    kond+kraft   beides zusammen

Alle vier laufen durch `constraints.common.guided_generate` mit demselben Seed,
denselben Konditionierungspartikeln und demselben `cfg_weight`. Der einzige
Unterschied ist der zugeschaltete Mechanismus.

Zwei getrennte Fragen
---------------------
1. **Laengentreue** — wird die angeforderte Laenge erreicht? Leitkennzahl ist
   `autoritaet`, die Steigung von "erreicht ueber angefordert": 1 heisst
   perfekte Kontrolle, 0 heisst keinerlei Wirkung. Ein einzelner relativer
   Fehler beantwortet das nicht (siehe `evalkit/metrics_length.py`).
2. **Rekonstruktion** — wie gut wird die Dichte abgedeckt? Aufgetragen wird
   ueber die **erreichte**, nicht die angeforderte Laenge, sonst vergleicht man
   Budgets statt Verfahren.

Warum die Leiter absolut ist
----------------------------
Die Laengeneinbettung normiert als `u = (log1p(L) - log1p(log_ref))/log_scale`.
Wo im trainierten Bereich eine Anforderung liegt, haengt also an `log_ref` des
jeweiligen Checkpoints und nicht am Vielfachen der freien Laenge. Eine relative
Leiter wuerde genau diesen Versatz verstecken. Der Kopf des Laufs druckt zu
jedem Leitersprossen den zugehoerigen u-Wert mit aus.

Ein Hinweis, der fuer die Auswertung entscheidend ist
-----------------------------------------------------
`model_zoo.load_model` gibt seit dieser Auswertung `length_freqs` an das Modell
weiter. Vorher wurde ein mit `--length_freqs linear` trainierter Checkpoint
stillschweigend mit Oktav-Frequenzen geladen — die Laengenkonditionierung
bekam dann sin/cos-Merkmale, die ihr MLP nie gesehen hatte, und wirkte
scheinbar gar nicht. Aeltere Zahlen zur "Laengenautoritaet" (siehe
`constraints/README.md`) sind damit nicht vergleichbar.

Beispiele
---------
    python run_length.py                       # alle Modelle aus transfer/
    python run_length.py --quick               # Rauchtest
    python run_length.py --ladder 4 6 8 11 14 18 --length_cfg 0.0
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

from evalkit import models as M                               # noqa: E402
from evalkit import plots as P                                # noqa: E402
from evalkit.generate import ARME, basis, erzeuge, partikel_aus_form   # noqa: E402
from evalkit.metrics_length import (autoritaet, bei_budget, flaeche_unter,   # noqa: E402
                                    laengentreue, nutzen_pro_weg, rekonstruktion)
from evalkit.progress import balken, beschreibe, schreibe      # noqa: E402

from common import (HOLDOUT_SHAPES, ErgodicMetrics, cp_to_curve_np,   # noqa: E402
                    summarise, write_metrics)

LEITER_STD = [4.0, 6.0, 8.0, 11.0, 14.0, 18.0]


def parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models', nargs='*', default=None)
    p.add_argument('--transfer', default=None)
    p.add_argument('--shapes', nargs='*', default=None)
    p.add_argument('--ladder', nargs='*', type=float, default=LEITER_STD,
                   help="angeforderte Ziellaengen (absolut)")
    p.add_argument('--arms', nargs='*', default=['kond', 'kraft', 'kond+kraft'],
                   help=f"Arme neben 'frei'; moeglich: {ARME[1:]}")
    p.add_argument('--length_cfg', type=float, default=0.0,
                   help="eigene CFG-Staerke des Laengenkanals")
    p.add_argument('--panel_lengths', nargs='*', type=float, default=None,
                   help="Ziellaengen fuer die Panel-Abbildungen (Vorgabe: kuerzeste und laengste)")
    p.add_argument('--panel_arms', nargs='*', default=None,
                   help="Arme fuer die Panel-Abbildungen (Vorgabe: alle gefahrenen)")

    p.add_argument('--force_weight', type=float, default=30.0)
    p.add_argument('--max_force', type=float, default=0.5)
    p.add_argument('--polish_steps', type=int, default=400)
    p.add_argument('--polish_lr', type=float, default=0.05)

    p.add_argument('--grid_res', type=int, default=64)
    p.add_argument('--bandbreite', type=float, default=0.06)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--pts', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None)
    p.add_argument('--out', default=None)
    p.add_argument('--tag', default='len')
    p.add_argument('--quick', action='store_true')
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.quick:
        a.shapes = a.shapes or HOLDOUT_SHAPES[:3]
        a.ladder = [4.0, 8.0, 14.0]
    shapes = a.shapes or HOLDOUT_SHAPES
    leiter = sorted(a.ladder)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(out, exist_ok=True)

    device = M.geraet(a.device)
    modelle = M.lade(a.models, device, ordner=a.transfer)
    M.manifest(modelle, out, f'{a.tag}_models.json')

    kraft_kw = dict(force_weight=a.force_weight, max_force=a.max_force,
                    polish_steps=a.polish_steps, polish_lr=a.polish_lr)

    # Arme, die dieses Modell ueberhaupt fahren kann.
    arme_je_modell = {}
    for m in modelle:
        arme = [x for x in a.arms if x in ARME]
        if not m.length_cond:
            weg = [x for x in arme if 'kond' in x]
            if weg:
                print(f"  [!] '{m.name}' hat length_cond=False — Arme {weg} "
                      f"entfallen fuer dieses Modell.")
            arme = [x for x in arme if 'kond' not in x]
        arme_je_modell[m.name] = arme
        u = lambda L, mm=m: ((math.log1p(L) - math.log1p(mm.meta['log_ref']))
                             / max(mm.meta['log_scale'], 1e-9))
        print(f"  '{m.name}': log_ref={m.meta['log_ref']:.2f} "
              f"log_scale={m.meta['log_scale']:.3f} freqs={m.meta.get('length_freqs')}")
        print("      Leiter (Ziel -> u): " +
              "  ".join(f"{L:g}->{u(L):+.2f}" for L in leiter))
    print()

    gesamt = sum(len(shapes) * (1 + len(arme_je_modell[m.name]) * len(leiter))
                 for m in modelle)
    bar = balken(gesamt, 'Path Length & Reconstruction', position=0)

    zeilen, panels = [], {}
    for m in modelle:
        B = basis(m.meta, a.pts, 5, device=device)
        erg = ErgodicMetrics(m.meta['nxi'], device)
        arme = arme_je_modell[m.name]

        for shape in shapes:
            d_map, parts = partikel_aus_form(shape, m.meta, device,
                                             grid_res=a.grid_res, seed=a.seed)
            ziel_t = torch.tensor(d_map, dtype=torch.float32, device=device)

            beschreibe(bar, modell=m.name, form=shape, arm='frei')
            cps, curve = erzeuge(m, parts, B, arm='frei', steps=a.steps,
                                 seed=a.seed, device=device)
            frei = dict(modell=m.name, shape=shape, arm='frei', ziel=float('nan'),
                        **rekonstruktion(curve, ziel_t, bandbreite=a.bandbreite,
                                         erg=erg, cps=cps),
                        rel_fehler=float('nan'), abs_fehler=float('nan'))
            zeilen.append(frei)
            bar.update(1)

            form_zeilen = [frei]
            for L in leiter:
                for arm in arme:
                    beschreibe(bar, modell=m.name, form=shape, arm=arm, L=L)
                    cps, curve = erzeuge(m, parts, B, arm=arm, ziel_laenge=L,
                                         length_cfg=a.length_cfg, steps=a.steps,
                                         seed=a.seed, device=device, kraft=kraft_kw)
                    rek = rekonstruktion(curve, ziel_t, bandbreite=a.bandbreite,
                                         erg=erg, cps=cps)
                    z = dict(modell=m.name, shape=shape, arm=arm, ziel=L, **rek,
                             **laengentreue(rek['laenge'], L))
                    zeilen.append(z)
                    form_zeilen.append(z)
                    bar.update(1)

                    _panel_sammeln(panels, a, m, arm, L, shape, d_map, cps, curve, z)

            nutzen_pro_weg(form_zeilen)
    bar.close()

    write_metrics(zeilen, out, f'{a.tag}_metrics.csv')
    agg = _zusammenfassung(zeilen, leiter, out, a.tag)
    _abbildungen(zeilen, agg, panels, leiter, out, a)
    return zeilen


def _panel_sammeln(panels, a, m, arm, L, shape, d_map, cps, curve, z):
    p_len = a.panel_lengths if a.panel_lengths is not None else [min(a.ladder), max(a.ladder)]
    p_arm = a.panel_arms if a.panel_arms is not None else a.arms
    if arm not in p_arm or not any(abs(L - x) < 1e-9 for x in p_len):
        return
    panels.setdefault((m.name, arm, L), []).append(dict(
        shape=shape, d_map=d_map, curve=cp_to_curve_np(cps[0].cpu().numpy()),
        label=f"L*={L:g}",
        kopf=f"'{shape}'  L={z['laenge']:.2f}  ({z['rel_fehler']:+.0%})"))


# ── Aggregation ──────────────────────────────────────────────────────────────
def _mittel(sub, key):
    v = np.array([z[key] for z in sub if key in z], dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float('nan')


def _zusammenfassung(zeilen, leiter, out, tag):
    """Aggregate je (Modell, Arm, Ziel) und je (Modell, Arm)."""
    agg = []
    for modell in sorted({z['modell'] for z in zeilen}):
        for arm in sorted({z['arm'] for z in zeilen}):
            for L in leiter:
                sub = [z for z in zeilen if z['modell'] == modell and z['arm'] == arm
                       and (np.isnan(z['ziel']) if arm == 'frei' else z['ziel'] == L)]
                if not sub:
                    continue
                agg.append(dict(modell=modell, arm=arm, ziel=L, n=len(sub),
                                laenge=_mittel(sub, 'laenge'),
                                abs_fehler=_mittel(sub, 'abs_fehler'),
                                rel_fehler=_mittel(sub, 'rel_fehler'),
                                rec_jsd=_mittel(sub, 'rec_jsd'),
                                rec_cov=_mittel(sub, 'rec_cov'),
                                E_erg=_mittel(sub, 'E_erg'),
                                cov_solver=_mittel(sub, 'cov_solver'),
                                nutzen_pro_weg=_mittel(sub, 'nutzen_pro_weg')))
                if arm == 'frei':
                    break
    write_metrics(agg, out, f'{tag}_summary.csv')

    # Autoritaet und Rekonstruktionsflaeche je (Modell, Arm)
    tab = []
    for modell in sorted({z['modell'] for z in zeilen}):
        for arm in sorted({z['arm'] for z in zeilen}):
            sub = [z for z in zeilen if z['modell'] == modell and z['arm'] == arm
                   and np.isfinite(z['ziel'])]
            if not sub:
                continue
            au = autoritaet([z['ziel'] for z in sub], [z['laenge'] for z in sub])
            punkte = [(x['laenge'], x['rec_jsd']) for x in
                      [d for d in agg if d['modell'] == modell and d['arm'] == arm]]
            tab.append(dict(modell=modell, arm=arm, **au,
                            abs_fehler=_mittel(sub, 'abs_fehler'),
                            rec_jsd=_mittel(sub, 'rec_jsd'),
                            auc_rec=flaeche_unter([p[0] for p in punkte],
                                                  [p[1] for p in punkte])))
    write_metrics(tab, out, f'{tag}_authority.csv')

    print("\n  Path length authority  (slope: achieved over requested)")
    kopf = (f"    {'Model':<26}{'Arm':<12}{'Slope':>10}{'R^2':>8}"
            f"{'Range':>9}{'|relE|':>9}{'rec_jsd':>10}{'AUC':>9}")
    print(kopf)
    print("    " + "-" * (len(kopf) - 4))
    for t in tab:
        print(f"    {t['modell']:<26}{t['arm']:<12}{t['autoritaet']:>10.3f}"
              f"{t['r2']:>8.3f}{t['spanne_erreicht']:>9.2f}{t['abs_fehler']:>9.3f}"
              f"{t['rec_jsd']:>10.4f}{t['auc_rec']:>9.4f}")

    print("\n  Comparison at equal *achieved* length  (rec_jsd, lower is better)")
    budgets = [b for b in (5.0, 7.0, 9.0, 12.0)]
    print(f"    {'Modell':<26}{'Arm':<12}" + "".join(f"L={b:g}".rjust(10) for b in budgets))
    for modell in sorted({d['modell'] for d in agg}):
        for arm in sorted({d['arm'] for d in agg}):
            sub = [d for d in agg if d['modell'] == modell and d['arm'] == arm]
            if len(sub) < 2:
                continue
            xs = [d['laenge'] for d in sub]
            ys = [d['rec_jsd'] for d in sub]
            werte = [bei_budget(xs, ys, b) for b in budgets]
            print(f"    {modell:<26}{arm:<12}" +
                  "".join(('  --  '.rjust(10) if v is None else f"{v:>10.4f}")
                          for v in werte))

    print()
    summarise([{k: z[k] for k in ('rec_jsd', 'rec_cov', 'laenge', 'abs_fehler')
                if k in z and np.isfinite(z[k])} for z in zeilen
               if np.isfinite(z.get('abs_fehler', np.nan))],
              label='Mean over all guided runs')
    return agg


# ── Abbildungen ──────────────────────────────────────────────────────────────
def _abbildungen(zeilen, agg, panels, leiter, out, a):
    import matplotlib.pyplot as plt

    modelle = sorted({z['modell'] for z in zeilen})
    arme = [x for x in ARME if x in {z['arm'] for z in zeilen}]
    gefuehrt = [x for x in arme if x != 'frei']
    L_frei = _mittel([z for z in zeilen if z['arm'] == 'frei'], 'laenge')

    # 1) Laengenkontrolle: angefordert gegen erreicht
    fig, axes = plt.subplots(1, len(modelle), squeeze=False,
                             figsize=(6.0 * len(modelle), 5.0), facecolor='white')
    lo, hi = min(leiter) - 1, max(leiter) + 1
    for j, modell in enumerate(modelle):
        ax = axes[0][j]
        ax.plot([lo, hi], [lo, hi], color='#B0B4B0', lw=1.2, ls='--',
                label='perfect control')
        ax.axhline(L_frei, color=P.C_GREY, lw=1, ls=':',
                   label=f'unguided (L~{L_frei:.1f})')
        for arm in gefuehrt:
            sub = [z for z in zeilen if z['modell'] == modell and z['arm'] == arm]
            ax.scatter([z['ziel'] for z in sub], [z['laenge'] for z in sub],
                       s=8, color=P.ARM_FARBEN.get(arm, '#888'), alpha=0.2, lw=0)
            mit = [d for d in agg if d['modell'] == modell and d['arm'] == arm]
            ax.plot([d['ziel'] for d in mit], [d['laenge'] for d in mit], '-o',
                    color=P.ARM_FARBEN.get(arm, '#888'), lw=2, ms=6, label=arm)
        P.achse(ax, f"'{modell}'", 'requested length', 'achieved length')
        ax.legend(frameon=True, fontsize=8, facecolor='white', edgecolor='#ddd')
    fig.suptitle("Path Length Control — requested vs. achieved "
                 f"(Holdout n={len({z['shape'] for z in zeilen})})",
                 fontsize=13, color=P.C_DARK)
    fig.tight_layout()
    P.save(fig, out, f"{a.tag}_kontrolle.png")
    plt.close(fig)

    # 2) Rekonstruktion ueber die ERREICHTE Laenge — die Kernabbildung
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor='white')
    for key, ax, yl in ((('rec_jsd'), axes[0], 'JSD(c \u2225 target density)  (lower is better)'),
                        (('E_erg'), axes[1], 'Ergodic error E_erg  (lower is better)')):
        for mi, modell in enumerate(modelle):
            for arm in arme:
                sub = [d for d in agg if d['modell'] == modell and d['arm'] == arm]
                if not sub:
                    continue
                sub = sorted(sub, key=lambda d: d['laenge'])
                if arm == 'frei':
                    ax.scatter([sub[0]['laenge']], [sub[0][key]], marker='*', s=140,
                               color=P.ARM_FARBEN['frei'], zorder=4,
                               label='unguided' if mi == 0 else None)
                    continue
                ax.plot([d['laenge'] for d in sub], [d[key] for d in sub],
                        P.linienart(mi), color=P.ARM_FARBEN.get(arm, '#888'),
                        lw=2, marker='o', ms=5, label=f"{arm} / {modell}")
        P.achse(ax, None, 'achieved length', yl)
        ax.legend(frameon=True, fontsize=7, facecolor='white', edgecolor='#ddd')
    fig.suptitle("Reconstruction vs. Path Length — plotted over *achieved* length\n"
                 "Colour = arm, line style = model",
                 fontsize=13, color=P.C_DARK)
    fig.tight_layout()
    P.save(fig, out, f"{a.tag}_rekonstruktion.png")
    plt.close(fig)

    # 3) Laengentreue und Nutzen je Wegeinheit
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), facecolor='white')
    P.boxen(axes[0], [z for z in zeilen if np.isfinite(z.get('abs_fehler', np.nan))],
            'abs_fehler', 'arm', 'modell', referenz=0.0, ref_label='exact',
            titel='Absolute relative path length error', yl='|L/L* - 1|')
    P.boxen(axes[1], [z for z in zeilen
                      if np.isfinite(z.get('nutzen_pro_weg', np.nan))],
            'nutzen_pro_weg', 'arm', 'modell', referenz=0.0,
            ref_label='no gain',
            titel='Reconstruction gain per additional path unit',
            yl='(rec_jsd_free - rec_jsd) / dL')
    fig.suptitle("Path Length Fidelity and Efficiency  (Box = holdout shapes)",
                 fontsize=13, color=P.C_DARK)
    fig.tight_layout()
    P.save(fig, out, f"{a.tag}_treue.png")
    plt.close(fig)

    # 4) Heatmaps: Modell x Arm gegen Ziellaenge
    reihen = [(m, arm) for m in modelle for arm in gefuehrt]
    if reihen:
        fig, axes = plt.subplots(1, 2, figsize=(7.0 * 2, 1.0 + 0.55 * len(reihen)),
                                 facecolor='white')
        for ax, key, titel, fmt in (
                (axes[0], 'laenge', 'Achieved length', '{:.1f}'),
                (axes[1], 'rec_jsd', 'rec_jsd (lower is better)', '{:.3f}')):
            mat = [[next((d[key] for d in agg if d['modell'] == m and d['arm'] == arm
                          and d['ziel'] == L), np.nan) for L in leiter]
                   for m, arm in reihen]
            P.heatmap(ax, mat, [f"{L:g}" for L in leiter],
                      [f"{m} / {arm}" for m, arm in reihen], titel=titel,
                      xl='requested length', fmt=fmt)
        fig.suptitle("Model x Arm vs. requested path length",
                     fontsize=13, color=P.C_DARK)
        fig.tight_layout()
        P.save(fig, out, f"{a.tag}_heatmap.png")
        plt.close(fig)

    # 5) Panels ueber alle Holdout-Formen
    for (modell, arm, L), eintraege in panels.items():
        P.panel(eintraege,
                f"Target Length L*={L:g} — Model '{modell}', Arm '{arm}'",
                out, f"{a.tag}_panel_{_slug(modell)}_{_slug(arm)}_L{L:g}.png",
                untertitel="Header: achieved length and relative error")


def _slug(s):
    return ''.join(ch if ch.isalnum() or ch in '-_' else '-' for ch in str(s))[:40]


if __name__ == '__main__':
    main()
