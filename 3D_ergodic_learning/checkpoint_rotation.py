r"""
checkpoint_rotation.py
======================
Je Lauf bleibt genau ein Checkpoint liegen.

Warum das eine eigene Datei ist: die Regel galt bisher in drei von zwoelf
Runnern, und in den dreien auch nur halb — nach dem Schreiben des `_final`
blieb der letzte `_ep`-Stand daneben stehen. Auf dem Cluster hatten sich so
**304 GB** in einem einzigen Verzeichnis angesammelt, bei rund einem Gigabyte
je Datei.

Zwei Funktionen, beide absichtlich vorsichtig:

`nach_zwischenstand(stem, run_str, neu, behalten=1)`
    Nach dem Schreiben eines `_ep####.pt` die aelteren desselben Laufs
    entfernen.

`nach_endstand(stem, run_str, final)`
    Nach dem Schreiben des `_final.pt` **alle** `_ep####.pt` desselben Laufs
    entfernen. Am Ende eines abgeschlossenen Laufs bleibt damit nur das
    `_final`; bricht der Job vorher ab, bleibt der letzte Zwischenstand.

Die Reihenfolge ist in beiden Faellen der Punkt: die neue Datei ist bereits
geschrieben und wird geprueft, *bevor* irgendetwas geloescht wird. Andersherum
waere ein fehlgeschlagener Schreibvorgang der Verlust des ganzen Trainings.

Geloescht wird ausschliesslich, was zum selben Lauf gehoert — gleicher
`run_str`, gleiches `_ep####.pt`-Muster. Checkpoints anderer Laeufe im selben
Verzeichnis bleiben unangetastet, auch aeltere.
"""
import glob
import os
import re

_EP = re.compile(r'_ep(\d+)\.pt$')
_MINDESTGROESSE = 1024


def _geschrieben(pfad):
    """Existiert die Datei und ist sie nicht leer?"""
    if not os.path.isfile(pfad) or os.path.getsize(pfad) < _MINDESTGROESSE:
        print(f"  [!] Neuer Stand fehlt oder ist leer ({pfad}) — "
              f"alte Staende bleiben stehen.")
        return False
    return True


def _staende(stem, run_str):
    aus = []
    for f in glob.glob(f"{stem}_{run_str}_ep*.pt"):
        m = _EP.search(f)
        if m:
            aus.append((int(m.group(1)), f))
    aus.sort()
    return aus


def _entfernen(pfade, neu=None):
    n = 0
    for p in pfade:
        if neu is not None and os.path.abspath(p) == os.path.abspath(neu):
            continue
        try:
            os.remove(p)
            print(f"  alter Stand entfernt: {os.path.basename(p)}")
            n += 1
        except OSError as e:
            print(f"  [!] konnte {p} nicht entfernen: {e}")
    return n


def nach_zwischenstand(stem, run_str, neu, behalten=1):
    """Aeltere Zwischenstaende desselben Laufs entfernen."""
    if not _geschrieben(neu):
        return 0
    behalten = max(1, int(behalten))
    alle = _staende(stem, run_str)
    return _entfernen([p for _, p in alle[:-behalten]], neu)


def nach_endstand(stem, run_str, final):
    """Nach dem Endstand alle Zwischenstaende desselben Laufs entfernen."""
    if not _geschrieben(final):
        return 0
    return _entfernen([p for _, p in _staende(stem, run_str)])
