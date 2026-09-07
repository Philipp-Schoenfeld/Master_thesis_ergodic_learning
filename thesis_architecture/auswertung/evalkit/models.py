r"""
models.py
=========
Checkpoints finden, benennen und laden.

Die Modelle werden in `transfer/` abgelegt, ohne dass die Auswertung ihre
Dateinamen vorher kennt. Dieses Modul findet sie dort, vergibt **kurze,
unterscheidende Namen** und laedt sie ueber das bestehende `model_zoo`.

Warum die Namen automatisch gebildet werden
-------------------------------------------
Checkpoint-Dateinamen dieses Projekts sind lang und zu 90 % identisch
(`cond_particles_crossattn_flow_matching_particle_ergodic_date_…`). Ein
Diagramm mit zwei solchen Beschriftungen ist unlesbar, und von Hand vergebene
Kuerzel veralten, sobald ein Modell ausgetauscht wird. Deshalb wird der
laengste gemeinsame Praefix und Suffix aller gefundenen Dateien abgeschnitten —
uebrig bleibt genau das, was die Modelle unterscheidet. Bei zwei Modellen, die
sich nur im Zusatz-Loss unterscheiden, ist das Ergebnis genau dieser Zusatz.

Ueberschreiben geht jederzeit mit `--models name=pfad`.

Was beim Laden geprueft wird
----------------------------
`beschreibe()` liest die Metadaten und meldet sie als Tabelle: `nxi`, `D`,
`n_particles`, `length_cond`, `log_ref/log_scale/length_freqs`, `lambda_erg`.
Diese Zeile ist nicht Deko — `length_freqs` entscheidet darueber, ob die
Laengenkonditionierung ueberhaupt wirken kann (siehe Kommentar in
`model_zoo.load_model`), und `log_ref` sagt, wo im trainierten Bereich eine
angeforderte Laenge liegt.
"""

import glob
import json
import os

import torch

from . import TRANSFER_DIR
from model_zoo import load_model        # noqa: E402  (Pfad-Bootstrap in __init__)

# Dateien, die im Transferordner liegen, aber keine 2D-Trajektoriennetze sind.
SKIP_HINTS = ('3d', 'flaechen', 'surf')


class Modell:
    """Ein geladener Checkpoint samt Name, Pfad und Metadaten."""

    def __init__(self, name, path, model, kind, meta):
        self.name, self.path = name, path
        self.model, self.kind, self.meta = model, kind, meta

    @property
    def length_cond(self):
        return bool(self.meta.get('length_cond', False))

    def __repr__(self):
        return f"Modell({self.name!r}, nxi={self.meta['nxi']}, length_cond={self.length_cond})"


# ── Finden ───────────────────────────────────────────────────────────────────
def finde_checkpoints(ordner=None, muster='*.pt'):
    """Alle Checkpoint-Dateien in `ordner` (Vorgabe: `transfer/`), sortiert."""
    ordner = ordner or TRANSFER_DIR
    treffer = sorted(glob.glob(os.path.join(ordner, muster)))
    return [p for p in treffer
            if not any(h in os.path.basename(p).lower() for h in SKIP_HINTS)]


def _kurznamen(pfade):
    """Kurze, unterscheidende Namen aus einer Liste langer Dateinamen.

    Gemeinsamen Praefix und Suffix abschneiden. Bleibt fuer eine Datei nichts
    uebrig (etwa weil nur ein Checkpoint vorliegt), wird der Dateiname ohne
    Endung benutzt.
    """
    stems = [os.path.splitext(os.path.basename(p))[0] for p in pfade]
    if len(stems) < 2:
        return [s[:40] for s in stems]

    def gemeinsam(seqs):
        n = 0
        while n < min(len(s) for s in seqs) and len({s[n] for s in seqs}) == 1:
            n += 1
        return n

    pre = gemeinsam(stems)
    suf = gemeinsam([s[::-1] for s in stems])
    namen = []
    for s in stems:
        kern = s[pre:len(s) - suf].strip('_-') or s
        namen.append(kern[:40])
    # Kollisionen (etwa nach dem Kuerzen auf 40 Zeichen) durchnummerieren.
    seen = {}
    out = []
    for n in namen:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}#{seen[n]}")
    return out


def waehle(specs=None, ordner=None, muster='*.pt'):
    """-> Liste (name, pfad).

    `specs` sind entweder `name=pfad`-Angaben von der Kommandozeile oder blosse
    Pfade; ohne `specs` wird `transfer/` durchsucht.
    """
    if specs:
        paare = []
        for s in specs:
            if '=' in s and not os.path.exists(s):
                name, pfad = s.split('=', 1)
            else:
                name, pfad = None, s
            if not os.path.isabs(pfad):
                kand = os.path.join(ordner or TRANSFER_DIR, pfad)
                pfad = kand if os.path.exists(kand) else os.path.abspath(pfad)
            paare.append((name, pfad))
        pfade = [p for _, p in paare]
        auto = _kurznamen(pfade)
        return [(n or a, p) for (n, p), a in zip(paare, auto)]

    pfade = finde_checkpoints(ordner, muster)
    return list(zip(_kurznamen(pfade), pfade))


# ── Laden ────────────────────────────────────────────────────────────────────
def lade(specs, device, ordner=None, muster='*.pt', nur_2d=True, verbose=True):
    """Alle gewaehlten Checkpoints laden. -> Liste[Modell]."""
    auswahl = waehle(specs, ordner, muster)
    if not auswahl:
        raise SystemExit(
            f"Keine Checkpoints gefunden in {ordner or TRANSFER_DIR}.\n"
            f"Modelle dort ablegen oder explizit angeben:\n"
            f"    --models basis=<pfad1>.pt zusatzloss=<pfad2>.pt")

    modelle = []
    for name, pfad in auswahl:
        if not os.path.exists(pfad):
            raise SystemExit(f"Checkpoint nicht gefunden: {pfad}")
        model, kind, meta = load_model(pfad, device)
        if nur_2d and meta.get('nd', 2) != 2:
            print(f"  [uebersprungen] {name}: nd={meta['nd']} (kein 2D-Netz)")
            continue
        modelle.append(Modell(name, pfad, model, kind, meta))
    if verbose:
        tabelle(modelle)
    return modelle


def tabelle(modelle):
    """Metadatentabelle der geladenen Modelle auf stdout."""
    if not modelle:
        print("  (keine Modelle)")
        return
    kopf = (f"  {'Name':<26}{'nxi':>5}{'D':>5}{'N':>6}{'ep':>6}"
            f"{'len_cond':>10}{'freqs':>9}{'log_ref':>9}{'log_sc':>8}{'erg_w':>8}")
    print("\n  Geladene Modelle")
    print(kopf)
    print("  " + "-" * (len(kopf) - 2))
    for m in modelle:
        mt = m.meta
        print(f"  {m.name:<26}{mt['nxi']:>5}{mt['D']:>5}{mt['n_particles']:>6}"
              f"{str(mt.get('epoch')):>6}{str(m.length_cond):>10}"
              f"{str(mt.get('length_freqs', '-')):>9}"
              f"{mt.get('log_ref', 0):>9.2f}{mt.get('log_scale', 0):>8.3f}"
              f"{mt.get('lambda_erg', 0) or 0:>8.4g}")
    print()


def manifest(modelle, out_dir, name='models.json'):
    """Welche Datei unter welchem Namen ausgewertet wurde — mit ins Ergebnis.

    Ohne das ist eine Metriktabelle drei Wochen spaeter nicht mehr zuzuordnen,
    weil die Kurznamen ja gerade *nicht* der Dateiname sind.
    """
    os.makedirs(out_dir, exist_ok=True)
    pfad = os.path.join(out_dir, name)
    with open(pfad, 'w', encoding='utf-8') as f:
        json.dump([{'name': m.name, 'datei': os.path.basename(m.path),
                    'pfad': m.path,
                    'meta': {k: v for k, v in m.meta.items()
                             if not k.startswith('_')
                             and isinstance(v, (int, float, str, bool, type(None)))}}
                   for m in modelle], f, indent=2, ensure_ascii=False)
    print(f"  Saved -> {pfad}")
    return pfad


def geraet(name=None):
    return torch.device(name if name else ('cuda' if torch.cuda.is_available() else 'cpu'))
