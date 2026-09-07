r"""
budget.py
=========
Zeitbudget und geordneter Abbruch — damit ein langer Lauf Ergebnisse hat.

Die Kette aus Orakel, Option A, Option B und Auswertung laeuft ueber Stunden,
lokal wie auf dem Cluster. Zwei Dinge koennen sie beenden, bevor sie fertig
ist: das eigene Budget (`--max_minuten`, damit die Stufen zusammen in ein
Zeitfenster passen) und SLURM (`#SBATCH --signal=SIGTERM@120` schickt zwei
Minuten vor dem harten Limit ein SIGTERM).

In beiden Faellen ist "sofort abbrechen" die schlechteste Antwort: dann sind
Stunden Rechenzeit weg und es liegt nichts vor. Stattdessen setzt dieses Modul
nur eine Fahne; die Schleifen fragen sie zwischen zwei Runden ab, brechen dort
ab und schreiben, was schon da ist. Ein halber Orakel-Lauf ist ein
vollstaendiger Datensatz mit weniger Zeilen — jede Entscheidung darin ist fuer
sich gueltig, weil sie nur von ihrem eigenen Glaubenszustand abhaengt.

Dieselbe Idee wie der `TerminateInterrupt`-Handler in den Trainings-Runnern
des Projekts, hier nur klein und ohne Checkpoint-Datei.
"""

import signal
import time


class Zeitbudget:
    """Uhr und SIGTERM-Fahne in einem.

    Args:
        max_minuten: Budget in Minuten; None oder <= 0 heisst "kein Limit"
                     (dann bleibt nur die SIGTERM-Fahne wirksam).
        reserve_min: Sicherheitsabstand. `abgelaufen()` meldet schon so viel
                     frueher, dass die laufende Runde noch zu Ende gerechnet
                     und das Ergebnis geschrieben werden kann.
    """

    def __init__(self, max_minuten=None, reserve_min=3.0, name=''):
        self.t0 = time.perf_counter()
        self.max_sek = (float(max_minuten) * 60.0
                        if max_minuten and max_minuten > 0 else None)
        self.reserve_sek = float(reserve_min) * 60.0
        self.name = name
        self.signal_empfangen = False
        self._alt = {}
        for sig in (getattr(signal, 'SIGTERM', None),
                    getattr(signal, 'SIGINT', None)):
            if sig is None:
                continue
            try:
                self._alt[sig] = signal.signal(sig, self._auf_signal)
            except (ValueError, OSError):
                # Nicht im Hauptthread oder Plattform kennt das Signal nicht —
                # dann bleibt das Zeitbudget allein wirksam.
                pass

    def _auf_signal(self, signum, rahmen):
        self.signal_empfangen = True

    # -- Abfragen -----------------------------------------------------------
    def verbraucht(self):
        return time.perf_counter() - self.t0

    def rest(self):
        """Verbleibende Sekunden bis zum Budget (ohne Reserve); None ohne Limit."""
        if self.max_sek is None:
            return None
        return self.max_sek - self.verbraucht()

    def abgelaufen(self, naechster_schritt_sek=0.0):
        """Soll die Schleife jetzt geordnet aufhoeren?

        `naechster_schritt_sek` ist die geschaetzte Dauer des naechsten
        Schrittes: uebergibt eine Schleife ihre gemessene Rundendauer, hoert
        sie auf, *bevor* die Runde nicht mehr hineinpasst, statt mittendrin
        abgeschnitten zu werden.
        """
        if self.signal_empfangen:
            return True
        if self.max_sek is None:
            return False
        return (self.verbraucht() + naechster_schritt_sek + self.reserve_sek
                >= self.max_sek)

    def grund(self):
        if self.signal_empfangen:
            return 'SIGTERM/SIGINT empfangen'
        return f'Zeitbudget erschoepft ({self.verbraucht() / 60:.1f} min)'

    def bericht(self):
        rest = self.rest()
        teil = f", noch {rest / 60:.1f} min" if rest is not None else ''
        return f"{self.name or 'Lauf'}: {self.verbraucht() / 60:.1f} min{teil}"

    def loesen(self):
        """Signalbehandlung zuruecknehmen (fuer Selbsttests und Notebooks)."""
        for sig, alt in self._alt.items():
            try:
                signal.signal(sig, alt)
            except (ValueError, OSError):
                pass
        self._alt = {}
