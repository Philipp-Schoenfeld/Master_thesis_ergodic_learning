r"""
rq2_plot.py
===========
RQ2 (Robustheit bei festem Zeitbudget) aus derselben `rq1_rq2_runtime.csv`
wie RQ1: ein Violinplot im Stil von Sun et al. Fig. 5 (ergodischer Fehler
nach festem Zeitbudget je Methode, eine Figur je Budget) plus eine
Erreichungsrate (Anteil Trials unter einer Qualitaetsschwelle).

Usage:
    python rq2_plot.py --out_tag smoke --budgets 0.1,0.5,2.0
"""
import argparse
import os

import numpy as np
import pandas as pd

import paper_style as ps


def E_at_budget(df, budget):
    """Je (method, trial): E beim letzten Checkpoint mit time_s<=budget
    (Treppenfunktion -- die tatsaechlich verfuegbare Information zum
    Zeitpunkt `budget`, kein Vorgriff auf spaetere Checkpoints)."""
    out = {}
    for m, sub in df.groupby('method'):
        vals = []
        for _, g in sub.groupby('trial'):
            g = g.sort_values('time_s')
            ok = g[g['time_s'] <= budget]
            vals.append(float(ok['E'].iloc[-1]) if len(ok) else float(g['E'].iloc[0]))
        out[m] = np.array(vals)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', type=str, required=True)
    ap.add_argument('--budgets', type=str, default='0.1,0.5,2.0',
                    help='Sekunden; 0.5 s ist Suns Wert in Benchmark Q2.A.')
    args = ap.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                           'results', args.out_tag)
    csv_path = os.path.join(out_dir, 'rq1_rq2_runtime.csv')
    df = pd.read_csv(csv_path)
    methods = [m for m in ps.METHOD_COLORS if m in df['method'].unique()]
    plot_dir = os.path.join(out_dir, 'plots')

    for b in (float(x) for x in args.budgets.split(',') if x):
        vals = E_at_budget(df, b)
        out_path = os.path.join(plot_dir, f'rq2_budget_{b:g}s.png')
        ps.violin(vals, f"Ergodic error after a {b:g}s budget", 'Ergodic error E (lower is better)',
                 out_path, methods=methods, log_y=True)
        print(f"[rq2_plot] budget={b:g}s -> {out_path}")
        for m in methods:
            v = vals[m]
            print(f"  {m:14s}  median E = {np.median(v):8.4g}   "
                 f"IQR = [{np.percentile(v, 25):.4g}, {np.percentile(v, 75):.4g}]")


if __name__ == '__main__':
    main()
