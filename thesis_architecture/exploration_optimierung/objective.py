r"""
objective.py
============
Was "gut" heisst — und warum das eine eigene Datei ist.

Gesucht ist "mit so wenig Ausfuehrungen wie moeglich die Zielverteilung so gut
wie moeglich abdecken". Das sind zwei Groessen, und ihr Verhaeltnis ist eine
*Setzung*, keine Messung. Deshalb steht die Setzung hier allein, getrennt von
der Mission, und arbeitet ausschliesslich auf den bereits gerechneten Spuren:

    J(Einstellung, n) = q(n)  +  lambda_len * n  +  lambda_time * t(n)

    q(n)  mittlerer *bezogener* Abdeckungsfehler nach n Ausfuehrungen,
          also cov / cov_blind, gemittelt ueber alle Holdout-Formen. 1,0 heisst
          "so gut wie ohne Bahn", 0,0 waere perfekt. Die Beziehung auf
          cov_blind ist noetig, damit grossflaechige Formen das Mittel nicht
          dominieren (siehe `mission.blind_coverage`).
    n     Zahl der Ausfuehrungen. Weil jede Ausfuehrung genau eine
          Laengeneinheit lang ist, **ist** n die gefahrene Strecke in
          Laengeneinheiten — der Term ist damit zugleich die Wegstrafe.
    t(n)  mittlere Rechenzeit einer Mission bis Runde n, in Sekunden. Nur im
          SVGD-Zweig von Bedeutung; ohne SVGD steht `lambda_time` auf 0.

Die beiden Gewichte sind Voreinstellungen mit Begruendung, keine Wahrheiten:

    lambda_len = 0.02   Eine zusaetzliche Ausfuehrung muss den bezogenen
                        Abdeckungsfehler um mindestens 2 Prozentpunkte senken,
                        um sich zu lohnen. In den Probelaeufen faellt q von
                        ~0,55 (n=1) auf ~0,10 (n=8); die ersten Runden bringen
                        je 5 bis 15 Punkte, die spaeten unter 1. Der Schnitt
                        liegt damit dort, wo die Kurve tatsaechlich abflacht.
    lambda_time = 0.004 Eine Mission, die 25 s mehr rechnet, muss 10
                        Prozentpunkte besser abdecken. Das stellt eine
                        SVGD-Verfeinerung ungefaehr einer zusaetzlichen
                        Ausfuehrung gegenueber, was die Frage ist, um die es im
                        zweiten Durchlauf geht.

**Diese Gewichte sind nachtraeglich aenderbar, ohne etwas neu zu rechnen.**
`optimize.py --reweight` liest die abgelegten Spuren und stellt die Rangfolge
mit anderen Gewichten neu auf. Zusaetzlich wird immer die vollstaendige
Pareto-Front (q gegen n) geschrieben — wer die Gewichte nicht teilt, liest die
Antwort dort direkt ab.
"""

import numpy as np

DEFAULT_LAMBDA_LEN = 0.02
DEFAULT_LAMBDA_TIME = 0.004

#: Gueteterm. 'cov' ist die Abdeckung (Frage: wurde ueberall hingefahren),
#: 'erg' der ergodische Fehler gegen die Wahrheit (Frage: stimmt auch die
#: Verweildauer). Die Aufgabe ist als Abdeckung gestellt, deshalb 'cov' als
#: Voreinstellung — 'erg' laeuft in jeder Tabelle als Kontrollspalte mit,
#: damit sichtbar bleibt, ob eine Einstellung die Abdeckung auf Kosten der
#: Ergodizitaet kauft.
QUALITY_KEYS = ('cov', 'erg')


def per_n(rows, quality='cov'):
    """Spur -> je Rundenzahl ein Mittel ueber die Formen.

    -> Liste von dicts, aufsteigend nach n_exec.
    """
    if not rows:
        return []
    qkey = {'cov': 'cov_norm', 'erg': 'erg_truth'}[quality]
    by_n = {}
    for r in rows:
        by_n.setdefault(int(r['n_exec']), []).append(r)

    n_max = max(int(r['n_exec']) for r in rows)
    out = []
    for n in sorted(by_n):
        grp = by_n[n]
        # Die Zeitspalten stehen als Gesamtzeit der *ganzen* Spur in jeder
        # Zeile; bis Runde n ist davon der Anteil n/n_max angefallen. Das ist
        # exakt, weil jede Runde denselben Aufwand hat: eine Planung und eine
        # Verfeinerung, beide unabhaengig von der Rundennummer.
        share = n / max(n_max, 1)
        rec = {
            'n_exec': n,
            'q': float(np.mean([x[qkey] for x in grp])),
            'cov_norm': float(np.mean([x['cov_norm'] for x in grp])),
            'cov': float(np.mean([x['cov'] for x in grp])),
            'erg_truth': float(np.mean([x['erg_truth'] for x in grp])),
            'belief_rmse': float(np.mean([x['belief_rmse'] for x in grp])),
            'info_gain': float(np.mean([x['info_gain'] for x in grp])),
            'path_len': float(np.mean([x['path_len'] for x in grp])),
            'time_s': float(np.mean([x['plan_s'] + x['svgd_s'] for x in grp])) * share,
            'n_shapes': len(grp),
        }
        out.append(rec)
    return out


def score_trace(rows, lambda_len=DEFAULT_LAMBDA_LEN,
                lambda_time=DEFAULT_LAMBDA_TIME, quality='cov', n_min=1,
                n_max=None):
    """Beste Rundenzahl einer Spur und ihr J.

    -> (bestes dict mit 'J' und 'n_exec', vollstaendige Liste je n)
    """
    table = per_n(rows, quality=quality)
    for rec in table:
        rec['J'] = (rec['q'] + lambda_len * rec['n_exec']
                    + lambda_time * rec['time_s'])
    cand = [r for r in table
            if r['n_exec'] >= n_min and (n_max is None or r['n_exec'] <= n_max)]
    if not cand:
        return None, table
    best = min(cand, key=lambda r: r['J'])
    return best, table


def pareto(table, x='n_exec', y='q'):
    """Die nicht dominierten Punkte einer (n, q)-Tabelle — beide klein besser.

    Damit laesst sich die Gewichtsfrage umgehen: was hier steht, ist bei
    *jedem* nichtnegativen lambda_len die Antwort fuer irgendein Budget.
    """
    pts = sorted(table, key=lambda r: (r[x], r[y]))
    front, best_y = [], float('inf')
    for p in pts:
        if p[y] < best_y - 1e-12:
            front.append(p)
            best_y = p[y]
    return front


def rank(results, **kw):
    """Mehrere Spuren -> nach J sortierte Liste.

    `results` ist eine Liste von dicts mit 'key' (Einstellung) und 'rows'.
    """
    out = []
    for res in results:
        best, table = score_trace(res['rows'], **kw)
        if best is None:
            continue
        rec = dict(res['key'])
        rec.update({k: best[k] for k in
                    ('n_exec', 'J', 'q', 'cov', 'cov_norm', 'erg_truth',
                     'belief_rmse', 'info_gain', 'path_len', 'time_s')})
        rec['table'] = table
        out.append(rec)
    return sorted(out, key=lambda r: r['J'])
