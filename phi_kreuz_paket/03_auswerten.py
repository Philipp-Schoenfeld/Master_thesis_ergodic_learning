#!/usr/bin/env python3
"""Die Zellen des Phi-Kreuzes zu einer Tabelle zusammenfassen.

    python 03_auswerten.py [ergebnisverzeichnis]

Gemittelt wird ueber alle Formen und die glaubensgetriebenen Missionen.
`orakel` und `maeher` bleiben aussen vor: das Orakel kennt die Wahrheit und
ignoriert den Glauben, der Maeher faehrt ein festes Raster — beide duerfen
sich zwischen den Mustern nicht unterscheiden und dienen als Kontrolle.
"""
import csv, os, sys, collections, statistics as st

B = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'thesis_architecture',
    'exploration', 'results', 'phi_wahrheit')
MOD = [('lse','Niveaumenge'), ('mass','Massenanteil'), ('eid','Informationsdichte'),
       ('mi','Wechselseitige Info'), ('ucb','UCB mu+kappa*sigma'),
       ('stretch','Streckung'), ('ei','Erwarteter Gewinn')]
MUS = [('haelfte','halbe'), ('quadranten','Quadr.'), ('loch','Loch')]
GLAUBEN = ['glaube-1', 'glaube-R', 'zweistufig', 'glaube-D', 'B-warm']

if not os.path.isdir(B):
    sys.exit(f'Kein Ergebnisverzeichnis: {B}')

roh = collections.defaultdict(list)
for d in sorted(os.listdir(B)):
    p = os.path.join(B, d, 'metriken.csv')
    if not os.path.exists(p) or '_' not in d:
        continue
    mu, mo = d.split('_', 1)
    for r in csv.DictReader(open(p)):
        roh[(mu, mo, r['mission'])].append(r)

if not roh:
    sys.exit(f'Keine metriken.csv in {B}')

def mw(mu, mo, feld, missionen):
    v = [float(r[feld]) for m in missionen for r in roh.get((mu, mo, m), [])]
    return st.mean(v) if v else float('nan')

for feld, titel in (('coverage', 'Abdeckungsfehler'), ('belief_rmse', 'Belief-RMSE')):
    print(f'\n=== {titel} — kleiner ist besser ===')
    print('%-22s %8s %8s %8s %9s' % ('Zieldichte', *[k for _, k in MUS], 'Spanne'))
    zeilen = []
    for mo, lbl in MOD:
        v = [mw(mu, mo, feld, GLAUBEN) for mu, _ in MUS]
        da = [x for x in v if x == x]          # nan heraus
        if not da:
            continue
        zeilen.append((st.mean(da), lbl, v, da))
    for _, lbl, v, da in sorted(zeilen):
        # Eine Spanne ueber unvollstaendige Zellen waere irrefuehrend: sie
        # sieht klein aus, weil Werte fehlen, nicht weil das Verfahren robust ist.
        spanne = ('%9.4f' % (max(da) - min(da))) if len(da) == len(MUS) else '     (unv.)'
        print('%-22s %s %s' % (lbl, ' '.join(('%8.4f' % x) if x == x else '     — ' for x in v), spanne))
    if any(len(da) < len(MUS) for *_, da in zeilen):
        print('  (unv.) = noch nicht alle drei Muster gerechnet')

print('\n=== Missionen im Vergleich (Abdeckungsfehler, ueber alle Muster und Phi) ===')
missionen = sorted({k[2] for k in roh})
for m in missionen:
    v = [float(r['coverage']) for k, rs in roh.items() if k[2] == m for r in rs]
    l = [float(r['path_len']) for k, rs in roh.items() if k[2] == m for r in rs]
    if v:
        print('  %-12s %.4f   Weglaenge %6.2f   n=%d' % (m, st.mean(v), st.mean(l), len(v)))

print('\n=== Kontrolle: orakel und maeher duerfen sich zwischen Mustern nicht unterscheiden ===')
for m in ('orakel', 'maeher'):
    v = [mw(mu, 'ucb', 'coverage', [m]) for mu, _ in MUS]
    if all(x == x for x in v):
        print('  %-8s %s  Spanne %.6f'
              % (m, ' '.join('%.4f' % x for x in v), max(v) - min(v)))
