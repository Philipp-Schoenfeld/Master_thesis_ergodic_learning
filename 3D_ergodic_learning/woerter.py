r"""
woerter.py
==========
Fuenfundzwanzig kurze Wortkoerper — dieselbe Bauart wie `text_volumen.py`,
nur mit `baue()` auf mehr als ein Zeichen angewandt.

`text_volumen.baue(zeichen, ...)` war nie auf ein einzelnes Zeichen
beschraenkt: `TextPath` setzt jede Zeichenkette, und die Even-Odd-Verrechnung
der Ringe kuemmert sich nicht darum, ob die Ringe zu einem oder zu mehreren
Buchstaben gehoeren. Ein Wort ist als Zielflaeche etwas anderes als ein
Einzelbuchstabe: mehrere, oft getrennte Koerper (Buchstabenzwischenraum) statt
eines einzelnen zusammenhaengenden Volumens — die Projektion muss also mit
Luecken in der Silhouette umgehen, nicht nur mit Loechern darin.

Wortliste: kurze Begriffe aus der eigenen Arbeit (SE3, CFM, GMM, ...) statt
beliebigem Fliesstext — bewusst nicht laenger als noetig, weil laengere
Woerter nur mehr vom selben Fall (noch mehr getrennte Teile) brauchten, ohne
neue Silhouetten-Eigenschaft hinzuzufuegen. Schrift bleibt DejaVu Sans (siehe
`text_volumen.py`), keine neue Lizenzfrage.

Selbsttest:

    python woerter.py --pruefen
"""
import text_volumen

WOERTER = [
    'SE3', 'TSVEC', 'CFM', 'LBO', 'GMM', 'SVGD', 'MPD', 'WARM', 'FLOW',
    'ERGO', 'PHI', 'KAPPA', 'GAUSS', 'SPLINE', 'PRIOR', 'ADAM', 'RELU',
    'UNET', 'VITA', 'CROSS', 'ZERO', 'FILM', 'ATTN', 'COEFF', 'IAS',
]

_MIT_LOCH = {'GMM', 'WARM', 'FLOW', 'PRIOR', 'ADAM', 'UNET', 'CROSS', 'ZERO',
             'COEFF', 'KAPPA'}


def _eintrag(wort):
    notiz = (f'{len(wort)} Zeichen, mehrere getrennte Teilkoerper'
             f'{" mit Loch/Loechern" if wort in _MIT_LOCH else ""} '
             '— Luecken statt Kanten in der Silhouette')
    return (f'Wort {wort}', (lambda w=wort: text_volumen.baue(w)), notiz)


WOERTER_KOERPER = {f'wort_{w.lower()}': _eintrag(w) for w in WOERTER}


def baue(key):
    return WOERTER_KOERPER[key][1]()


def pruefen(verbose=True):
    fehler = []
    for k, (label, bauer, _) in WOERTER_KOERPER.items():
        try:
            m = bauer()
        except Exception as e:                              # noqa: BLE001
            fehler.append((k, f'Bau fehlgeschlagen: {type(e).__name__}: {e}'))
            continue
        grund = []
        if not m.is_watertight:
            grund.append('nicht wasserdicht')
        if not m.is_winding_consistent:
            grund.append('Umlaufsinn uneinheitlich')
        if m.volume <= 1e-9:
            grund.append(f'Volumen {m.volume:.3g}')
        if grund:
            fehler.append((k, ', '.join(grund)))
        if verbose:
            zeichen = 'ok ' if not grund else 'FEHLER'
            print(f'  {zeichen} {k:14s} {len(m.vertices):6d} Ecken '
                  f'{len(m.faces):6d} Dreiecke  V={m.volume:.4f}')
    return fehler


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pruefen', action='store_true')
    p.parse_args()
    print(f'{len(WOERTER_KOERPER)} Woerter\n')
    f = pruefen()
    print()
    if f:
        for k, g in f:
            print(f'  FEHLER {k}: {g}')
        raise SystemExit(1)
    print('Alle Woerter wasserdicht.')
