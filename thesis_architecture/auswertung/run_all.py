#!/usr/bin/env python3
r"""
run_all.py
==========
**Ein Befehl fuer die gesamte metrische Auswertung.**

Startet nacheinander

    1) 01_exploration_exploitation/run_ee.py
    2) 02_length_reconstruction/run_length.py

fuer alle Checkpoints in `transfer/` (oder die per `--models` angegebenen),
zeigt einen Gesamtfortschritt mit Restzeitschaetzung im Terminal, und fasst am
Ende beide Ergebnisordner in `results/` zusammen.

Modelle bereitstellen
----------------------
Checkpoints (`.pt`) nach `transfer/` legen (der Ordner der Projektwurzel, eine
Ebene ueber `thesis_architecture/`). Kurznamen fuer Abbildungen und Tabellen
werden automatisch aus dem unterscheidenden Teil der Dateinamen gebildet
(siehe `evalkit/models.py`) — bei zwei Checkpoints, die sich nur im
Zusatz-Loss unterscheiden, ist das genau dieser Zusatz.

    python run_all.py                              # alles, volle Holdout-Menge
    python run_all.py --quick                       # Rauchtest (Minuten, nicht Stunden)
    python run_all.py --only ee                      # nur Exploration/Exploitation
    python run_all.py --only length                  # nur Laengen-Rekonstruktion
    python run_all.py --models basis=a.pt zusatz=b.pt --ee_args --kappa 0 2 4 8

Zusatzargumente an ein einzelnes Experiment durchreichen: `--ee_args ...` bzw.
`--length_args ...` nehmen alles bis zum naechsten `--..._args` oder Zeilenende
und geben es unveraendert an den jeweiligen Runner weiter.

Fortschritt und Laufzeit
-------------------------
Jedes Teilexperiment meldet seinen eigenen Balken (Formen x Szenarien x kappa
bzw. Formen x Arme x Laengen); dieses Skript fuegt eine Gesamtzeile mit den
zwei Experimenten davor und schreibt nach jedem eine Laufzeit. So ist im
Terminal jederzeit sichtbar, wie weit der Gesamtlauf ist und wie lange er noch
braucht.
"""
import argparse
import os
import runpy
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evalkit.progress import Uhr, schreibe                    # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTE = [
    ('ee', 'Exploration & Exploitation',
     os.path.join(_here, '01_exploration_exploitation', 'run_ee.py')),
    ('length', 'Path Length & Reconstruction',
     os.path.join(_here, '02_length_reconstruction', 'run_length.py')),
]


def _split_passthrough(argv):
    """`--models ...` (global) von `--ee_args ...` / `--length_args ...` trennen.

    Alles nach `--ee_args` bis zum naechsten `--*_args` oder Ende gehoert dem
    ee-Runner, entsprechend fuer `--length_args`; der Rest ist global.
    """
    global_args, passthrough = [], {'ee': [], 'length': []}
    aktuell = None
    for tok in argv:
        if tok in ('--ee_args', '--length_args'):
            aktuell = tok[2:-5]
            continue
        (passthrough[aktuell] if aktuell else global_args).append(tok)
    return global_args, passthrough


def parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models', nargs='*', default=None,
                   help="name=pfad oder blosse Pfade; ohne Angabe wird transfer/ durchsucht")
    p.add_argument('--transfer', default=None, help="Ordner mit den Checkpoints")
    p.add_argument('--shapes', nargs='*', default=None,
                   help="an beide Experimente durchgereicht (Vorgabe: volle Holdout-Menge)")
    p.add_argument('--only', choices=['ee', 'length'], default=None,
                   help="nur eines der beiden Experimente laufen lassen")
    p.add_argument('--quick', action='store_true',
                   help="Rauchtest beider Experimente: wenige Formen, kurze Leiter")
    p.add_argument('--device', default=None)
    p.add_argument('--out', default=os.path.join(_here, 'results'))
    return p


def main(argv=None):
    global_argv, passthrough = _split_passthrough(sys.argv[1:] if argv is None else argv)
    a = parser().parse_args(global_argv)
    os.makedirs(a.out, exist_ok=True)

    gemeinsam = []
    if a.models:
        gemeinsam += ['--models'] + a.models
    if a.transfer:
        gemeinsam += ['--transfer', a.transfer]
    if a.shapes:
        gemeinsam += ['--shapes'] + a.shapes
    if a.quick:
        gemeinsam += ['--quick']
    if a.device:
        gemeinsam += ['--device', a.device]

    laeufe = [e for e in EXPERIMENTE if a.only is None or e[0] == a.only]
    print(f"\n{'#' * 78}\n#  Full Metric Evaluation  --  {len(laeufe)} experiment(s)\n"
          f"{'#' * 78}")

    t_start = time.perf_counter()
    ergebnisse = {}
    for i, (kurz, titel, skript) in enumerate(laeufe, 1):
        argv_lauf = gemeinsam + passthrough.get(kurz, [])
        with Uhr(f"[{i}/{len(laeufe)}] {titel}") as uhr:
            sys.argv = [skript] + argv_lauf
            modul = runpy.run_path(skript, run_name='__main__')
            ergebnisse[kurz] = modul.get('main')

    gesamt_min = (time.perf_counter() - t_start) / 60
    print(f"\n{'#' * 78}\n#  Done in {gesamt_min:.1f} min. Results are in:\n"
          f"#    01_exploration_exploitation/results/\n"
          f"#    02_length_reconstruction/results/\n"
          f"{'#' * 78}")


if __name__ == '__main__':
    main()
