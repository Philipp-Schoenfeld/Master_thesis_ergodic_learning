r"""
progress.py
===========
Fortschrittsbalken mit Restzeit, damit ein Lauf im Terminal lesbar bleibt.

Zwei Ebenen, bewusst nicht mehr:

    Ebene 0   der Gesamtlauf (`run_all.py`): welches Experiment von wie vielen
    Ebene 1   die Arbeitseinheiten *eines* Experiments

Die Arbeitseinheit ist immer eine **Generierung** (eine Form, ein Szenario, ein
kappa bzw. eine Ziellaenge, ein Arm), nicht eine Form. Das ist der Grund, warum
die Restzeit brauchbar ist: eine Form kostet je nach Zahl der Arme das Zehnfache
einer anderen, eine Generierung dagegen ungefaehr immer gleich viel.

`beschreibe()` schreibt die gerade laufende Kombination ins Postfix, damit bei
einem Abbruch sofort sichtbar ist, wo es stand.
"""

import os
import sys
import time

from tqdm.auto import tqdm

# Die Balkenzeichen von tqdm sind Unicode-Blockelemente. Nach dem
# `reconfigure` in `evalkit/__init__.py` steht stdout auf UTF-8, in einer alten
# cmd.exe kann das trotzdem schiefgehen -- dort auf ASCII zurueckfallen.
_ASCII = os.name == 'nt' and (getattr(sys.stdout, 'encoding', '') or '').lower() \
    not in ('utf-8', 'utf8')

_FORMAT = ('{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
           '[{elapsed}<{remaining}, {rate_fmt}]{postfix}')


def balken(total, desc, position=0, leave=True, unit='gen'):
    """Ein Fortschrittsbalken in einheitlichem Stil."""
    return tqdm(total=total, desc=desc, position=position, leave=leave,
                unit=unit, dynamic_ncols=True, ascii=_ASCII,
                bar_format=_FORMAT, file=sys.stdout, mininterval=0.3)


def beschreibe(bar, **felder):
    """Aktuelle Kombination ins Postfix des Balkens schreiben."""
    if bar is not None:
        bar.set_postfix(felder, refresh=False)


def schreibe(bar, text):
    """Zeile ausgeben, ohne den Balken zu zerreissen."""
    if bar is not None:
        bar.write(text)
    else:
        print(text)


class Uhr:
    """Kontextmanager, der die Laufzeit eines Abschnitts meldet."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        print(f"\n{'=' * 78}\n>>> {self.name}\n{'=' * 78}", flush=True)
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        print(f"<<< {self.name}  fertig in {dt / 60:.1f} min", flush=True)
        return False
