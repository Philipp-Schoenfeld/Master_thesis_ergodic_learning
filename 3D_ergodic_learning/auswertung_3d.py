#!/usr/bin/env python3
r"""
auswertung_3d.py
================
Die komplette 3D-Auswertung eines Checkpoints in einem Aufruf — und alles, was
dabei entsteht, in genau einem Ordner.

Bisher waren das fuenf Skripte, die man in der richtigen Reihenfolge mit
zueinander passenden Pfaden aufrufen musste; die Ergebnisse landeten je nach
Aufruf in `results/surfaces/`, `viz/` oder `visualisierungen/`. Hier laeuft die
Kette einmal durch:

    1  run_surface_eval.py       Bahnen erzeugen + Metriken     -> metriken.csv, bahnen.json
    2  export_meshes.py          Dreiecksnetze der Flaechen     -> meshes.json
    3  export_surface_viewer.py  kompakte Fassung fuer Betrachter -> viewer.json
    4  render_3d_snapshots.py    je Szene ein PNG *und* eine     -> bilder/<flaeche>/<form>.{png,html}
                                 drehbare Plotly-Seite
    5  plot_surface_metrics.py   Metrik-Diagramme               -> diagramme/
    6  (hier)                    Uebersichtsseite               -> index.html

Ergebnisordner (Standard `auswertungen/<name>/`):

    index.html          hier anfangen — Kacheln aller Szenen, jede klickbar
    diagramme/          01_pro_flaeche.png, 02_heatmap.png, 03_verteilung.png,
                        metriken.html (interaktiv), zusammenfassung.csv
    bilder/             je (Flaeche, Form) ein PNG und eine drehbare .html
    metriken.csv        eine Zeile je Form x Flaeche — die Rohzahlen
    bahnen.json         Bahnen, Oberflaechenpunkte, Gewichte (Nachrechnen ohne Netz)
    lauf.json           Checkpoint, Argumente, Laufzeiten
    log.txt             die vollstaendige Konsolenausgabe aller Schritte

Beispiel:

    python auswertung_3d.py --ckpt checkpoints/mein_neues_netz.pt --name mein_netz

Wiederholungslauf ohne die teure Inferenz (nur Diagramme und Seite neu bauen):

    python auswertung_3d.py --ckpt ... --name mein_netz --ab 5
"""
import argparse
import csv
import html
import json
import os
import subprocess
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

# The Windows console's default codepage (cp1252) cannot encode the box-
# drawing / arrow characters tqdm and the German docstrings use; without this
# the very first non-ASCII byte forwarded from a subprocess crashes the run.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCHRITTE = ['eval', 'meshes', 'viewer', 'bilder', 'diagramme', 'index']

BLAU = '#1565C0'
GRUEN = '#00C853'
TEXT = '#1A1A2E'
GRAU = '#555555'


# ── Hilfen ───────────────────────────────────────────────────────────────────
class Tee:
    """Konsolenausgabe gleichzeitig auf den Bildschirm und in log.txt."""

    def __init__(self, pfad):
        self.f = open(pfad, 'a', encoding='utf-8')

    def write(self, s):
        sys.stdout.write(s)
        sys.stdout.flush()
        self.f.write(s)
        self.f.flush()

    def close(self):
        self.f.close()


def lauf(cmd, tee, cwd=_here):
    """Ein Unterskript starten, Ausgabe durchreichen, bei Fehler abbrechen."""
    tee.write('\n$ ' + ' '.join(f'"{c}"' if ' ' in str(c) else str(c) for c in cmd) + '\n')
    t0 = time.perf_counter()
    umg = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                         errors='replace', env=umg, bufsize=1)
    # Chunked, not line-by-line: a tqdm bar updates via bare '\r' with no '\n',
    # so reading line-by-line would buffer the whole progress bar invisibly
    # until the step finishes. Small chunks keep it live in both the terminal
    # and log.txt.
    for stueck in iter(lambda: p.stdout.read(256), ''):
        tee.write(stueck)
    p.wait()
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        tee.write(f'\n!! Abbruch: Rueckgabewert {p.returncode} nach {dt:.0f} s\n')
        raise SystemExit(p.returncode)
    tee.write(f'   ... {dt:.0f} s\n')
    return dt


def checkpoint_pruefen(ckpt, tee):
    """Metadaten lesen und pruefen, dass die Gewichte zur Architektur passen.

    Ohne diese Pruefung endet ein unpassender Checkpoint mitten in Schritt 1 in
    einem `load_state_dict`-Stapelauszug — nach dem Laden von 1 GB und dem
    Aufbau aller Flaechen. Hier steht stattdessen sofort da, welche Schluessel
    fehlen oder zu viel sind.
    """
    import torch
    from flow_matching_cond_particles_crossattn import ParticleCrossAttnFlowNetwork

    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    meta = {k: ck[k] for k in ('epoch', 'loss', 'nxi', 'D', 'orientation', 'start_cond', 'args')
            if k in ck and not hasattr(ck[k], 'shape')}
    nxi, D = ck.get('nxi', 25), ck.get('D', 384)
    ori = bool(ck.get('orientation', False))
    start_cond = bool(ck.get('start_cond', False))
    tee.write(f'\nCheckpoint  {ckpt}\n')
    tee.write(f'  nxi={nxi}  D={D}  Orientierung={ori}  Start-Konditionierung={start_cond}  '
              f'Epoche={ck.get("epoch", "?")}  Verlust={ck.get("loss", float("nan"))}\n')
    if start_cond:
        tee.write('  Kein Startpunkt in der Auswertung bekannt -> Null-Start-Token '
                  '(neutrale Ablation, wie in der Trainingsplanung vorgesehen).\n')

    netz = ParticleCrossAttnFlowNetwork(nxi=nxi, nd=3, D=D, predict_orientation=ori,
                                        start_cond=start_cond)
    fehlt, zuviel = netz.load_state_dict(ck['model_state_dict'], strict=False)
    n_par = sum(p.numel() for p in netz.parameters())
    tee.write(f'  {n_par / 1e6:.1f} M Parameter\n')
    if fehlt or zuviel:
        tee.write(f'  !! fehlende Schluessel  ({len(fehlt)}): {list(fehlt)[:8]}\n')
        tee.write(f'  !! ueberzaehlige       ({len(zuviel)}): {list(zuviel)[:8]}\n')
        raise SystemExit(
            'Der Checkpoint passt nicht zu ParticleCrossAttnFlowNetwork in\n'
            'flow_matching_cond_particles_crossattn.py. Das ist kein Fehler der\n'
            'Auswertung — die Architektur muss zuerst angeglichen werden\n'
            '(z. B. Startpunkt-Konditionierung), sonst wertet man ein halb\n'
            'geladenes Netz aus.')
    tee.write('  Gewichte passen vollstaendig zur Architektur.\n')
    return dict(nxi=nxi, D=D, orientation=ori,
                epoch=ck.get('epoch'), loss=ck.get('loss'),
                params_mio=round(n_par / 1e6, 2),
                meta={k: (v if isinstance(v, (int, float, str, bool, type(None)))
                          else str(v)) for k, v in meta.items()})


# ── Uebersichtsseite ─────────────────────────────────────────────────────────
def index_bauen(ziel, name, info, tee):
    """index.html aus metriken.csv und den erzeugten Dateien zusammensetzen."""
    with open(os.path.join(ziel, 'metriken.csv'), newline='', encoding='utf-8') as f:
        zeilen = list(csv.DictReader(f))

    formen, flaechen = [], []
    for r in zeilen:
        if r['shape'] not in formen:
            formen.append(r['shape'])
        if r['surface'] not in flaechen:
            flaechen.append(r['surface'])

    try:
        import surfaces
        lab = {}
        for k in flaechen:
            try:
                lab[k] = surfaces.build(k).label
            except Exception:
                lab[k] = k
    except Exception:
        lab = {k: k for k in flaechen}

    zus_pfad = os.path.join(ziel, 'diagramme', 'zusammenfassung.json')
    gesamt = {}
    if os.path.exists(zus_pfad):
        gesamt = json.load(open(zus_pfad, encoding='utf-8')).get('gesamt', {})

    def zahl(r, k):
        try:
            return float(r[k])
        except (KeyError, TypeError, ValueError):
            return float('nan')

    werte = {(r['shape'], r['surface']): r for r in zeilen}

    # Kacheln: eine Zeile je Flaeche, darin je Form ein Bild mit Link auf die
    # drehbare Fassung. So liegt die Vergleichbarkeit dort, wo man sie braucht —
    # dieselbe Geometrie nebeneinander ueber alle Dichten.
    abschnitte = []
    for k in flaechen:
        kacheln = []
        for nm in formen:
            r = werte.get((nm, k))
            if r is None:
                continue
            png = f'bilder/{k}/{nm}.png'
            seite = f'bilder/{k}/{nm}.html'
            hat_png = os.path.exists(os.path.join(ziel, png.replace('/', os.sep)))
            hat_seite = os.path.exists(os.path.join(ziel, seite.replace('/', os.sep)))
            bild = (f'<img loading="lazy" src="{png}" alt="{html.escape(nm)}">'
                    if hat_png else '<div class="kein">no PNG</div>')
            innen = (f'<a href="{seite}" target="_blank">{bild}</a>' if hat_seite else bild)
            kacheln.append(
                f'<figure class="kachel">{innen}'
                f'<figcaption><b>{html.escape(nm)}</b>'
                f'<span>erg {zahl(r, "erg"):.4f} \u00b7 cov {zahl(r, "coverage"):.3f} \u00b7 '
                f'point {zahl(r, "pointing_deg"):.0f}\u00b0</span>'
                f'{"<em>interactive</em>" if hat_seite else ""}</figcaption></figure>')
        abschnitte.append(
            f'<section class="flaeche" data-key="{html.escape(k)}">'
            f'<h3>{html.escape(lab.get(k, k))} <small>{html.escape(k)}</small></h3>'
            f'<div class="gitter">{"".join(kacheln)}</div></section>')

    optionen = ''.join(f'<option value="{html.escape(k)}">{html.escape(lab.get(k, k))}</option>'
                       for k in flaechen)

    def kennzahl(schluessel, titel, fmt='{:.4f}'):
        g = gesamt.get(schluessel)
        if not g:
            return ''
        return (f'<div class="kz"><span>{titel}</span>'
                f'<b>{fmt.format(g["mittel"])}</b>'
                f'<small>Median {fmt.format(g["median"])} · '
                f'Max {fmt.format(g["max"])}</small></div>')

    diagramme = []
    for datei, titel in (('01_pro_flaeche.png', 'Means per target surface'),
                         ('02_heatmap.png', 'Shape \u00d7 Surface'),
                         ('03_verteilung.png', 'Distribution per surface')):
        if os.path.exists(os.path.join(ziel, 'diagramme', datei)):
            diagramme.append(f'<figure class="diagramm"><figcaption>{titel}</figcaption>'
                             f'<a href="diagramme/{datei}" target="_blank">'
                             f'<img loading="lazy" src="diagramme/{datei}" alt="{titel}"></a>'
                             f'</figure>')

    seite = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>3D Evaluation — {html.escape(name)}</title>
<style>
  :root {{ --blau:{BLAU}; --gruen:{GRUEN}; --text:{TEXT}; --grau:{GRAU}; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:#fff; color:var(--text);
         font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif; }}
  header {{ padding:26px 30px 18px; border-bottom:1px solid #eee; }}
  h1 {{ margin:0 0 4px; font-size:21px; letter-spacing:-.2px; }}
  .sub {{ color:var(--grau); font-size:13px; }}
  .kzr {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:16px; }}
  .kz {{ border:1px solid #e6e6e6; border-radius:8px; padding:9px 14px; min-width:150px; }}
  .kz span {{ display:block; color:var(--grau); font-size:11px;
              text-transform:uppercase; letter-spacing:.6px; }}
  .kz b {{ font-size:19px; color:var(--blau); }}
  .kz small {{ display:block; color:#888; font-size:11px; }}
  main {{ padding:22px 30px 60px; }}
  h2 {{ font-size:16px; margin:34px 0 12px; padding-bottom:6px;
        border-bottom:1px solid #eee; }}
  .diagramme {{ display:grid; gap:18px;
                grid-template-columns:repeat(auto-fit,minmax(430px,1fr)); }}
  .diagramm {{ margin:0; border:1px solid #eee; border-radius:8px; padding:10px; }}
  .diagramm figcaption {{ color:var(--grau); font-size:12px; margin-bottom:8px; }}
  .diagramm img {{ width:100%; display:block; }}
  .werkzeuge {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap;
                margin:6px 0 18px; }}
  select, .knopf {{ font:inherit; padding:6px 10px; border:1px solid #ddd;
                    border-radius:6px; background:#fff; color:var(--text);
                    text-decoration:none; }}
  .knopf:hover {{ border-color:var(--blau); color:var(--blau); }}
  .flaeche h3 {{ font-size:14px; margin:24px 0 10px; }}
  .flaeche h3 small {{ color:#aaa; font-weight:400; margin-left:8px; }}
  .gitter {{ display:grid; gap:12px;
             grid-template-columns:repeat(auto-fill,minmax(210px,1fr)); }}
  .kachel {{ margin:0; border:1px solid #eee; border-radius:8px; overflow:hidden;
             background:#fff; }}
  .kachel img {{ width:100%; display:block; background:#fff; }}
  .kachel:hover {{ border-color:var(--gruen); }}
  .kein {{ height:150px; display:flex; align-items:center; justify-content:center;
           color:#bbb; font-size:12px; }}
  figcaption {{ padding:7px 9px; font-size:11px; line-height:1.4; }}
  figcaption span {{ display:block; color:var(--grau); }}
  figcaption em {{ color:var(--gruen); font-style:normal; font-size:10px;
                   text-transform:uppercase; letter-spacing:.5px; }}
  footer {{ padding:18px 30px 50px; color:#999; font-size:12px; }}
  code {{ background:#f6f6f6; padding:1px 5px; border-radius:4px; font-size:12px; }}
</style></head><body>
<header>
  <h1>3D Surface Evaluation — {html.escape(name)}</h1>
  <div class="sub">
    {len(formen)} holdout shapes \u00d7 {len(flaechen)} target surfaces = {len(zeilen)} scenes \u00b7
    Checkpoint <code>{html.escape(str(info.get('ckpt', '')))}</code> \u00b7
    Epoch {html.escape(str(info.get('epoch', '?')))} \u00b7
    {html.escape(str(info.get('params_mio', '?')))} M parameters \u00b7
    Orientation {'yes' if info.get('orientation') else 'no'} \u00b7
    generated {html.escape(info.get('zeit', ''))}
  </div>
  <div class="kzr">
    {kennzahl('erg', 'Ergodic error \u2193', '{:.5f}')}
    {kennzahl('coverage', 'Coverage \u2193')}
    {kennzahl('standoff_err', 'Standoff error \u2193', '{:.3f}')}
    {kennzahl('pointing_deg', 'Pointing error \u2193', '{:.1f}\u00b0')}
    {kennzahl('path_len', 'Path length \u00b7', '{:.2f}')}
  </div>
</header>
<main>
  <h2>Metrics</h2>
  <div class="werkzeuge">
    <a class="knopf" href="diagramme/metriken.html" target="_blank">Interactive Metrics Page \u25b8</a>
    <a class="knopf" href="metriken.csv">metriken.csv (raw)</a>
    <a class="knopf" href="diagramme/zusammenfassung.csv">zusammenfassung.csv</a>
  </div>
  <div class="diagramme">{''.join(diagramme)}</div>

  <h2>Scenes — click image for interactive 3D view</h2>
  <div class="werkzeuge">
    <label for="filter" style="color:{GRAU}">Target surface</label>
    <select id="filter"><option value="">show all</option>{optionen}</select>
  </div>
  {''.join(abschnitte)}
</main>
<footer>
  Generated by <code>auswertung_3d.py</code>. Raw data for computing without network:
  <code>bahnen.json</code>, <code>meshes.json</code>, <code>viewer.json</code>.
  Full log in <code>log.txt</code>, configuration in <code>lauf.json</code>.
</footer>
<script>
  document.getElementById('filter').addEventListener('change', function (e) {{
    var k = e.target.value;
    document.querySelectorAll('section.flaeche').forEach(function (s) {{
      s.style.display = (!k || s.dataset.key === k) ? '' : 'none';
    }});
  }});
</script>
</body></html>"""

    pfad = os.path.join(ziel, 'index.html')
    with open(pfad, 'w', encoding='utf-8') as f:
        f.write(seite)
    tee.write(f'\n[html] {pfad}\n')
    return pfad


# ── Ablauf ───────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True, help='Pfad zum .pt-Checkpoint')
    p.add_argument('--name', default=None,
                   help='Ordnername unter --out_root (Standard: Dateiname des Checkpoints)')
    p.add_argument('--out_root', default=os.path.join(_here, 'auswertungen'))
    p.add_argument('--shapes', type=int, default=25,
                   help='Wie viele Holdout-Formen (Standard 25 = die volle Menge)')
    p.add_argument('--shape_names', nargs='+', default=None)
    p.add_argument('--surfaces', nargs='+', default=None,
                   help='Standard: alle Flaechen aus surfaces.KEYS')
    p.add_argument('--steps', type=int, default=100, help='ODE-Schritte bei der Erzeugung')
    p.add_argument('--cfg_weight', type=float, default=2.0)
    p.add_argument('--n_particles', type=int, default=512)
    p.add_argument('--standoff_target', type=float, default=0.12)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None)
    p.add_argument('--html_offline', action='store_true',
                   help='plotly.js in jede Szenenseite einbetten (~4,5 MB je Datei, '
                        'dafuer ohne Internet lesbar)')
    p.add_argument('--ab', type=int, default=1, choices=range(1, len(SCHRITTE) + 1),
                   help=f'Erst ab diesem Schritt laufen: {", ".join(f"{i+1}={s}" for i, s in enumerate(SCHRITTE))}')
    p.add_argument('--bis', type=int, default=len(SCHRITTE), choices=range(1, len(SCHRITTE) + 1))
    p.add_argument('--ohne_png', action='store_true',
                   help='Nur die drehbaren HTML-Szenen rendern, keine PNGs '
                        '(schneller, aber der Index hat dann keine Vorschaubilder)')
    a = p.parse_args()

    ckpt = os.path.abspath(a.ckpt)
    if not os.path.exists(ckpt):
        raise SystemExit(f'Checkpoint nicht gefunden: {ckpt}')
    name = a.name or os.path.splitext(os.path.basename(ckpt))[0]
    ziel = os.path.abspath(os.path.join(a.out_root, name))
    os.makedirs(ziel, exist_ok=True)

    tee = Tee(os.path.join(ziel, 'log.txt'))
    zeit = time.strftime('%Y-%m-%d %H:%M')
    tee.write('=' * 78 + f'\n3D-Auswertung "{name}"   {zeit}\nZiel: {ziel}\n' + '=' * 78 + '\n')

    py = sys.executable
    dauer, info = {}, dict(ckpt=ckpt, name=name, zeit=zeit, argumente=vars(a))

    if a.ab <= 1 <= a.bis:
        info.update(checkpoint_pruefen(ckpt, tee))

        cmd = [py, os.path.join(_here, 'run_surface_eval.py'),
               '--ckpt', ckpt, '--out_dir', ziel,
               '--shapes', a.shapes, '--steps', a.steps,
               '--cfg_weight', a.cfg_weight, '--n_particles', a.n_particles,
               '--seed', a.seed]
        if a.shape_names:
            cmd += ['--shape_names'] + a.shape_names
        if a.surfaces:
            cmd += ['--surfaces'] + a.surfaces
        if a.device:
            cmd += ['--device', a.device]
        dauer['1 eval'] = lauf(cmd, tee)

    if a.ab <= 2 <= a.bis:
        dauer['2 meshes'] = lauf([py, os.path.join(_here, 'export_meshes.py'),
                                  '--out', os.path.join(ziel, 'meshes.json')], tee)

    if a.ab <= 3 <= a.bis:
        dauer['3 viewer'] = lauf([py, os.path.join(_here, 'export_surface_viewer.py'),
                                  '--json', os.path.join(ziel, 'bahnen.json'),
                                  '--out', os.path.join(ziel, 'viewer.json')], tee)

    if a.ab <= 4 <= a.bis:
        cmd = [py, os.path.join(_here, 'render_3d_snapshots.py'),
               '--in_dir', ziel, '--out_dir', os.path.join(ziel, 'bilder'), '--html']
        if a.ohne_png:
            cmd.append('--html_only')
        if a.html_offline:
            cmd.append('--html_offline')
        dauer['4 bilder'] = lauf(cmd, tee)

    if a.ab <= 5 <= a.bis:
        dauer['5 diagramme'] = lauf(
            [py, os.path.join(_here, 'plot_surface_metrics.py'),
             '--csv', os.path.join(ziel, 'metriken.csv'),
             '--out_dir', os.path.join(ziel, 'diagramme'),
             '--standoff_target', a.standoff_target,
             '--titel', f'3D-Flaechenauswertung — {name}'], tee)

    if a.ab <= 6 <= a.bis:
        t0 = time.perf_counter()
        index_bauen(ziel, name, info, tee)
        dauer['6 index'] = time.perf_counter() - t0

    info['dauer_s'] = {k: round(v, 1) for k, v in dauer.items()}
    with open(os.path.join(ziel, 'lauf.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=1, default=str)

    tee.write('\n' + '=' * 78 + '\n')
    for k, v in dauer.items():
        tee.write(f'  {k:<14s} {v:7.0f} s\n')
    tee.write(f'  {"gesamt":<14s} {sum(dauer.values()):7.0f} s\n')
    tee.write('=' * 78 + '\n')
    tee.write(f'\nFertig. Hier anfangen:\n\n    {os.path.join(ziel, "index.html")}\n\n')
    tee.close()


if __name__ == '__main__':
    main()
