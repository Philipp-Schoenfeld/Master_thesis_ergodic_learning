r"""
rq1_plot.py
===========
RQ1 (Laufzeit bis zum Ziel-Niveau) aus `rq1_rq2_runtime.csv`: ein Violinplot
im Stil von Sun et al. Fig. 4 (Zeit bis Epsilon je Methode) und eine
Anytime-Kurve im Stil von Fig. 13 (Ergodik ueber der Wall-Clock-Zeit, Median
+/- Interquartilsbereich).

Epsilon-Wahl (siehe Thesis-Doc, RQ1): relativ zur Ground Truth statt Suns
absolutem Wert 0.005 -- `--eps_quantile` nimmt das gegebene Perzentil der
`sun_cold`-Endwerte (hoechstes gemessenes Iterationsbudget) als Ziel-Niveau,
`--eps` ueberschreibt mit einem festen Wert, falls ein absoluter Vergleich mit
Sun gewuenscht ist.

Usage:
    python rq1_plot.py --out_tag smoke
"""
import argparse
import os

import numpy as np
import pandas as pd

import paper_style as ps


def time_to_eps(df, eps):
    """Je (method, trial): kleinste Zeit, bei der E<=eps gemessen wurde
    (lineare Interpolation zwischen den beiden umgebenden Checkpoints);
    NaN = nie erreicht (zensiert)."""
    out = {}
    for m, sub in df.groupby('method'):
        vals = []
        for _, g in sub.groupby('trial'):
            g = g.sort_values('iters')
            e, t = g['E'].to_numpy(), g['time_s'].to_numpy()
            below = np.where(e <= eps)[0]
            if len(below) == 0:
                vals.append(np.nan)
                continue
            k = below[0]
            if k == 0:
                vals.append(t[0])
                continue
            # lineare Interpolation in E zwischen Checkpoint k-1 und k
            e0, e1, t0, t1 = e[k - 1], e[k], t[k - 1], t[k]
            frac = 0.0 if e0 == e1 else (e0 - eps) / (e0 - e1)
            vals.append(t0 + frac * (t1 - t0))
        out[m] = np.array(vals)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', type=str, required=True)
    ap.add_argument('--eps', type=float, default=None,
                    help='Festes Ziel-Niveau (Suns Wert fuer die Fourier-Ergodik: 0.005; '
                         'hier K=8, siehe exp_common.ergodic_error -- Skalen nicht 1:1 '
                         'vergleichbar, nur als Referenzgroesse gedacht).')
    ap.add_argument('--eps_quantile', type=float, default=0.5,
                    help="Ohne --eps: Ziel = dieses Perzentil von sun_cold's E beim "
                         "groessten Checkpoint (relativ zur Ground Truth, s. Doc RQ1).")
    args = ap.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                           'results', args.out_tag)
    csv_path = os.path.join(out_dir, 'rq1_rq2_runtime.csv')
    df = pd.read_csv(csv_path)
    methods = [m for m in ps.METHOD_COLORS if m in df['method'].unique()]

    if args.eps is not None:
        eps = args.eps
    else:
        kmax = df['iters'].max()
        ref = df[(df['method'] == 'sun_cold') & (df['iters'] == kmax)]['E']
        eps = float(ref.quantile(args.eps_quantile))
    print(f"[rq1_plot] eps = {eps:.3g} "
         f"({'fixed' if args.eps is not None else f'q{args.eps_quantile} of sun_cold @ max iters'})")

    tte = time_to_eps(df, eps)
    plot_dir = os.path.join(out_dir, 'plots')
    p1 = ps.violin(tte, f"Time to reach ergodic error ≤ {eps:.3g}",
                   'Time to target (s)', os.path.join(plot_dir, 'rq1_time_to_eps.png'),
                   methods=methods, log_y=True,
                   censored_note='not reached within the measured iteration budget: '
                                 + ', '.join(f"{ps.METHOD_LABELS.get(m, m)} "
                                             f"{int(np.isnan(tte[m]).sum())}/{len(tte[m])}"
                                             for m in methods if np.isnan(tte[m]).any()))
    print(f"[rq1_plot] wrote {p1}")

    p2 = ps.anytime_curve(df, 'Convergence over wall-clock time (median, IQR shaded)',
                          os.path.join(plot_dir, 'rq1_anytime_curve.png'),
                          methods=methods, log_x=True, log_y=True)
    print(f"[rq1_plot] wrote {p2}")

    for m in methods:
        n_nan = int(np.isnan(tte[m]).sum())
        ok = tte[m][~np.isnan(tte[m])]
        med = np.median(ok) if len(ok) else float('nan')
        print(f"  {m:14s}  median time-to-eps = {med:8.3f}s  "
             f"({n_nan}/{len(tte[m])} not reached)")


if __name__ == '__main__':
    main()
