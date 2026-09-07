r"""
pruefe_warmstart.py
===================
Der Beweis, dass `--init_model` die Gewichte wirklich laedt.

Warum das eine eigene Pruefung braucht
--------------------------------------
Der urspruengliche Plan sah vor, den Verlust der ersten Epoche des
Feinabgleichs gegen den Endverlust des geladenen Laufs zu halten. Das kann
nicht aufgehen: der geladene Stand (`..._surfB_lang_..._ep1750.pt`, Endverlust
0,3344) wurde mit `--erg_on position`, ohne Orientierungsterm und mit
`w_cfm_rot 1.0` trainiert, der Feinabgleich laeuft mit `--erg_on footprint`,
`--lambda_ori 0.012` und `w_cfm_rot 0.5`. Das sind verschiedene Zielfunktionen;
zwei Zahlen daraus zu vergleichen wuerde nichts belegen — der Orientierungsterm
allein hebt einen Verlust von 3,10 auf 428.

Was stattdessen geprueft wird, ist dieselbe Aussage, nur an der richtigen
Stelle: das warmgestartete Netz wird **unter der Konfiguration des geladenen
Laufs, auf dessen Datenbank** ausgewertet. Kommt dabei wieder ~0,334 heraus,
sind die 87,5 Millionen Gewichte da, wo sie hingehoeren, und die neu
hinzugekommenen Bausteine (`start_emb`, `null_start_token`) tragen nichts bei.
Kommt etwas anderes heraus, ist der Warmstart kaputt — und zwar bevor
zweiundzwanzig Stunden Rechenzeit darauf gebaut werden.

Die zweite Haelfte der Aussage — dass der Null-Init exakt neutral ist — steht
in `test_3d_port.test_startpunkt` und ist dort bitgenau (`max |diff| = 0`).

    python pruefe_warmstart.py \
        --init_model checkpoints/surf_..._ep1750.pt \
        --db3d ergodic_dataset_3d_alt.db --erwartet 0.3344
"""
import argparse, os, sys, time

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from data_3d import augment_batch                                   # noqa: E402
from data_surfaces import load_surface_db, ziehgewichte, mix_parsen  # noqa: E402
from flow_matching_cond_particles_crossattn import (                # noqa: E402
    ParticleCrossAttnFlowNetwork, compute_particle_cfm_loss)
from flow_matching_runner_particles import _warmstart                # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--init_model', required=True)
    p.add_argument('--db3d', default=os.path.join(_here,
                                                  'ergodic_dataset_3d_alt.db'))
    p.add_argument('--db_splits', nargs='+', default=['train'])
    p.add_argument('--erwartet', type=float, default=None,
                   help='Endverlust des geladenen Laufs. Ohne Angabe wird der '
                        'Wert aus dem Checkpoint genommen.')
    p.add_argument('--toleranz', type=float, default=0.06,
                   help='zulaessige relative Abweichung. Der Trainingsverlust '
                        'ist ein Mittel ueber eine Epoche mit zufaelliger '
                        'Augmentierung und zufaelligem t, also selbst '
                        'streuend; exakte Gleichheit waere hier die falsche '
                        'Erwartung.')
    p.add_argument('--sigma', type=float, default=2.5,
                   help='Breite des Standardfehler-Bands in Vielfachen.')
    p.add_argument('--batches', type=int, default=40)
    p.add_argument('--mini_batch', type=int, default=32)
    p.add_argument('--nxi', type=int, default=25)
    p.add_argument('--nd', type=int, default=3)
    p.add_argument('--D', type=int, default=384)
    p.add_argument('--seed', type=int, default=0)
    # Die Konfiguration des geladenen Laufs. Voreinstellungen sind die aus
    # `..._surfB_lang_..._ep1750.pt` gelesenen Werte, nicht die des
    # Feinabgleichs.
    p.add_argument('--lambda_erg', type=float, default=100.0)
    p.add_argument('--erg_K', type=int, default=6)
    p.add_argument('--erg_pts', type=int, default=128)
    p.add_argument('--erg_t_power', type=float, default=2.0)
    p.add_argument('--erg_on', default='position')
    p.add_argument('--w_cfm_rot', type=float, default=1.0)
    p.add_argument('--p_drop', type=float, default=0.1)
    p.add_argument('--mu_thresh', type=float, default=0.5)
    p.add_argument('--noise_std', type=float, default=0.015)
    p.add_argument('--rot_range', type=float, default=20.0)
    p.add_argument('--scale_range', type=float, nargs=2, default=[0.75, 1.25])
    p.add_argument('--trans_range', type=float, default=0.08)
    p.add_argument('--rot_full', action='store_true', default=True)
    p.add_argument('--start_cond', action='store_true', default=True,
                   help='Netz mit Startpunkt-Kopf bauen — genau der Fall, den '
                        'der Feinabgleich benutzt.')
    # Orientierungsterm: ohne --lambda_ori > 0 bleibt orientation=None wie
    # bisher (die Konfiguration von ep1750). Fuer einen Checkpoint, der MIT
    # Orientierungsterm trainiert wurde (z.B. der 79-Flaechen-Feinabgleich),
    # muss diese Pruefung densselben Term bilden — sonst vergleicht sie zwei
    # verschiedene Zielfunktionen, genau der Fehler, den dieses Skript laut
    # eigenem Docstring vermeiden soll.
    p.add_argument('--orientation', action='store_true', default=False)
    p.add_argument('--lambda_ori', type=float, default=0.0)
    p.add_argument('--w_point', type=float, default=0.1)
    p.add_argument('--w_standoff', type=float, default=300.0)
    p.add_argument('--w_angsmooth', type=float, default=2.0)
    p.add_argument('--standoff_target', type=float, default=0.12)
    p.add_argument('--standoff_band', type=float, default=0.03)
    p.add_argument('--bspline_deg', type=int, default=5)
    p.add_argument('--mix', type=str, nargs='+', default=None,
                   help="Ziehgewichte je Gruppe wie im Runner, etwa "
                        "--mix ebene=0.25. Ohne Angabe wird gleichverteilt "
                        "gezogen — das ist NUR dann dieselbe Verteilung wie "
                        "beim geladenen Lauf, wenn der ebenfalls ohne --mix "
                        "trainiert wurde. Sonst vergleicht die Pruefung eine "
                        "andere Stichprobe als die, aus der der aufgezeichnete "
                        "Verlust stammt, und faellt durch, ohne dass am "
                        "Checkpoint etwas kaputt waere.")
    p.add_argument('--ebene_flach_anteil', type=float, default=0.5)
    a = p.parse_args()
    if a.lambda_ori > 0.0 and not a.orientation:
        raise SystemExit("--lambda_ori braucht --orientation (wie im Runner).")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    ckpt_verlust = None
    print(f'\nWarmstart-Pruefung  (Geraet: {device})')
    print(f'  Checkpoint: {os.path.basename(a.init_model)}')
    print(f'  Datenbank:  {os.path.basename(a.db3d)}  Splits {a.db_splits}')
    print(f'  Zielfunktion des geladenen Laufs: erg_on={a.erg_on}, '
          f'lambda_erg={a.lambda_erg:g}, erg_K={a.erg_K}, '
          f'w_cfm_rot={a.w_cfm_rot:g}, '
          + (f'lambda_ori={a.lambda_ori:g}' if a.lambda_ori > 0.0
             else 'kein Orientierungsterm') + '\n')

    model = ParticleCrossAttnFlowNetwork(
        nxi=a.nxi, nd=a.nd, D=a.D, predict_orientation=True,
        start_cond=a.start_cond).to(device)
    ckpt = _warmstart(model, a.init_model, device)
    ckpt_verlust = ckpt.get('loss')

    from ergodic_metric import ErgodicLoss
    ergodic = ErgodicLoss(
        nxi=a.nxi, K=a.erg_K, pts=a.erg_pts, deg=a.bspline_deg, weight=a.lambda_erg,
        t_power=a.erg_t_power, weighted_target=True, nd=a.nd,
        ergodic_on=a.erg_on, mu_thresh=a.mu_thresh).to(device)

    orientation_loss = None
    if a.lambda_ori > 0.0:
        from orientation_energy import OrientationLoss
        orientation_loss = OrientationLoss(
            nxi=a.nxi, pts=a.erg_pts, deg=a.bspline_deg, weight=a.lambda_ori,
            t_power=a.erg_t_power, w_point=a.w_point, w_standoff=a.w_standoff,
            w_angsmooth=a.w_angsmooth, standoff_target=a.standoff_target,
            standoff_band=a.standoff_band, mu_thresh=a.mu_thresh).to(device)

    eintraege = load_surface_db(a.db3d, nxi=a.nxi, splits=tuple(a.db_splits))
    if not eintraege:
        raise SystemExit(f'keine Eintraege in {a.db3d} fuer {a.db_splits}')
    print(f'  {len(eintraege)} Eintraege geladen')

    x1_all = torch.from_numpy(np.stack([e['x1'] for e in eintraege])).float()
    pa_all = torch.from_numpy(np.stack([e['parts'] for e in eintraege])).float()

    mix = mix_parsen(a.mix)
    ziehw = (ziehgewichte(eintraege, mix, ebene_flach_anteil=a.ebene_flach_anteil)
             if mix else None)
    if mix:
        print(f'  Ziehgewichte: {mix} (ebene_flach_anteil={a.ebene_flach_anteil:g})')

    rng = np.random.default_rng(a.seed)
    model.train()                       # derselbe Zustand wie beim Training
    werte, t0 = [], time.perf_counter()
    with torch.no_grad():
        for i in range(a.batches):
            idx = rng.choice(len(eintraege), size=a.mini_batch, replace=False,
                             p=ziehw)
            xb = x1_all[idx].to(device)
            pb = pa_all[idx].to(device)
            xa, pa_ = augment_batch(xb, pb, p_flip=0.0, rot_range=a.rot_range,
                                    scale_range=a.scale_range,
                                    trans_range=a.trans_range,
                                    noise_std=a.noise_std,
                                    rot_full=a.rot_full)
            verlust, _ = compute_particle_cfm_loss(
                model, xa, pa_, p_drop=a.p_drop, ergodic=ergodic,
                orientation=orientation_loss, w_cfm_rot=a.w_cfm_rot,
                p_drop_start=0.0)
            werte.append(float(verlust))
            if (i + 1) % 10 == 0:
                print(f'    {i+1}/{a.batches}  laufendes Mittel '
                      f'{np.mean(werte):.5f}  ({time.perf_counter()-t0:.0f} s)')

    w = np.array(werte)
    soll = a.erwartet if a.erwartet is not None else ckpt_verlust
    print(f'\n  Verlust ueber {len(w)} Stapel: Mittel {w.mean():.5f}  '
          f'sd {w.std():.5f}  Standardfehler {w.std()/np.sqrt(len(w)):.5f}')
    if soll is None:
        print('  Kein Sollwert bekannt — nichts zu vergleichen.')
        return 0
    # Das Kriterium muss die eigene Streuung kennen.
    #
    # Der Verlust je Stapel schwankt hier stark (sd in der Groessenordnung des
    # Mittelwerts): t wird je Beispiel neu gezogen, die Augmentierung dreht und
    # skaliert, und der ergodische Term haengt an beidem. Eine feste relative
    # Toleranz gegen den Mittelwert zu halten misst deshalb vor allem, wie
    # viele Stapel man gerechnet hat — mit acht Stapeln fiel die Pruefung bei
    # einem Standardfehler von 0,042 durch, obwohl der Sollwert weniger als
    # einen Standardfehler entfernt lag. Verglichen wird daher gegen das
    # groessere aus Standardfehler-Band und relativer Toleranz.
    se = float(w.std(ddof=1) / np.sqrt(len(w)))
    abw = abs(w.mean() - soll)
    band = max(a.sigma * se, a.toleranz * abs(soll))
    print(f'  Endverlust des geladenen Laufs: {soll:.5f}')
    print(f'  Abweichung {abw:.5f}  =  {abw/max(se,1e-12):.2f} Standardfehler'
          f'  =  {abw/max(abs(soll),1e-12)*100:.2f} %')
    print(f'  Zulaessiges Band: {band:.5f}  (das groessere aus '
          f'{a.sigma:g} x Standardfehler und {a.toleranz*100:.0f} %)')
    if len(w) < 20:
        print(f'  [!] nur {len(w)} Stapel — fuer eine belastbare Aussage '
              f'sollten es mindestens 20 sein.')
    if abw <= band:
        print('\n  BESTANDEN — die Gewichte sind geladen, und der '
              'Startpunkt-Kopf traegt nichts bei.')
        return 0
    print('\n  FEHLGESCHLAGEN — das warmgestartete Netz rechnet nicht das,\n'
          '  was der geladene Lauf zuletzt gerechnet hat. Bevor darauf 22 h\n'
          '  Rechenzeit gesetzt werden, gehoert das geklaert:\n'
          '    - passt die Zielfunktion (erg_on, erg_K, w_cfm_rot) wirklich\n'
          '      zu der des Checkpoints?\n'
          '    - passt die Datenbank (Augmentierung, Flaechen) dazu?\n'
          '    - meldet --init_model fehlende Schluessel ausser start_emb.*?')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
