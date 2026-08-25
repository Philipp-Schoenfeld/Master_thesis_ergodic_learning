r"""
run_surface_eval.py
===================
Das trainierte 3D-CFM+ErgLoss-Netz auf gekruemmte Oberflaechen loslassen.

Die Holdout-Formen werden nicht mehr auf die Trainingsebene gelegt, sondern
auf sieben verschiedene Oberflaechen projiziert (siehe `surfaces.py`). Das Netz
bleibt unangetastet — es konditioniert auf eine Partikelwolke (x, y, z, mu) und
sieht nicht, ob die aus einer Scheibe oder von einem Hasen stammt.

Gemessen wird viererlei:

    erg          ergodischer Fehler gegen die projizierte Zieldichte
    coverage     dichtegewichteter Abstand der Oberflaeche zur naechsten Bahn
    standoff     Abstand der Bahn zur Oberflaeche — trainiert auf 0,12
    pointing     Winkel zwischen Sensorachse und Richtung zur Oberflaeche

`standoff` und `pointing` sind die eigentliche Probe: sie pruefen, ob die im
Training auf einer Ebene gelernte SE(3)-Haltung auf einer gekruemmten Flaeche
ueberhaupt noch eine Bedeutung hat.

    python run_surface_eval.py --ckpt checkpoints/...ep0424.pt --shapes 12
"""
import argparse, json, os, sys, time
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import surfaces
from data_3d import load_pairs
from flow_matching_cond_particles_crossattn import (
    ParticleCrossAttnFlowNetwork, generate_particle_trajectories)
from ergodic_metric import (make_k_grid, trajectory_coeffs,
                            target_coeffs_from_particles)
from orientation import rot6d_to_matrix, sensor_axis
from orientation_energy import ParticleSurface
from obstacles import bspline_basis_matrix


def curve_from_cps(cps, pts=256, deg=5, device='cpu'):
    B = torch.from_numpy(bspline_basis_matrix(cps.shape[1], pts, deg)).float()
    return torch.einsum('pi,kid->kpd', B.to(device), cps.float())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--shapes', type=int, default=12)
    p.add_argument('--surfaces', nargs='+', default=surfaces.KEYS)
    p.add_argument('--n_particles', type=int, default=512)
    p.add_argument('--surface_points', type=int, default=20000)
    p.add_argument('--store_points', type=int, default=3500,
                   help='So viele Oberflaechenpunkte kommen zum Zeichnen in '
                        'die JSON. Die Partikelwolke allein ist zu duenn, um '
                        'eine Flaeche erkennbar zu machen.')
    p.add_argument('--dens_res', type=int, default=128)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--cfg_weight', type=float, default=2.0)
    p.add_argument('--pts', type=int, default=256)
    p.add_argument('--mu_thresh', type=float, default=0.5)
    p.add_argument('--erg_K', type=int, default=6)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None)
    p.add_argument('--out_dir', default=os.path.join(_here, 'results', 'surfaces'))
    a = p.parse_args()

    a.device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(a.seed)

    ck = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    nxi, D = ck.get('nxi', 25), ck.get('D', 384)
    ori = bool(ck.get('orientation', False))
    model = ParticleCrossAttnFlowNetwork(nxi=nxi, nd=3, D=D,
                                         predict_orientation=ori).to(a.device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    print(f"Netz: D={D} nxi={nxi} Orientierung={ori}  Epoche {ck.get('epoch','?')}"
          f"  Verlust {ck.get('loss', float('nan')):.4f}")
    print(f"Geraet: {a.device}")

    # ── Holdout-Formen: die 2D-Dichten, die projiziert werden ────────────
    from shape_library import pdf_on_grid
    # `load_pairs` liefert (Trajektorien, Dichte-Definitionen, Splits), jeweils
    # als Dict ueber den Formnamen. Gebraucht wird hier nur die Definition.
    _, defs, _ = load_pairs(nxi, splits=('val',))
    shapes_ = []
    for nm, df in defs.items():
        d2, _, _ = pdf_on_grid(df, resolution=a.dens_res)
        d2 = np.asarray(d2, dtype=np.float64)
        shapes_.append((nm, d2 / max(d2.max(), 1e-12)))
        if len(shapes_) >= a.shapes:
            break
    print(f"{len(shapes_)} Holdout-Formen: {', '.join(n for n, _ in shapes_)}")

    k_idx, Lam = make_k_grid(a.erg_K)
    k_idx = torch.tensor(k_idx, device=a.device)
    Lam = torch.tensor(Lam, dtype=torch.float32, device=a.device)

    surf = {k: surfaces.build(k) for k in a.surfaces}
    for k, s in surf.items():
        print(f"  {s.label:22s} {len(s.mesh.faces):6d} Dreiecke — {s.note}")

    rows, dump = [], {'meta': {k: v for k, v in vars(a).items()
                              if isinstance(v, (int, float, str, bool))},
                      'eintraege': []}
    t0 = time.perf_counter()

    for si, (name, d2) in enumerate(shapes_):
        print(f"\n[{si + 1}/{len(shapes_)}] {name}")
        for key in a.surfaces:
            s = surf[key]
            pts_s, nrm_s, w_s = surfaces.project(
                s, d2, n_points=a.surface_points, seed=a.seed + si)
            hit = float((w_s > 1e-3).mean())
            parts_np = surfaces.particles_from_projection(
                pts_s, w_s, a.n_particles, seed=a.seed + si)
            parts = torch.from_numpy(parts_np).to(a.device)

            g = torch.Generator(device=a.device).manual_seed(a.seed * 97 + si)
            cps, rot6 = generate_particle_trajectories(
                model, parts, num_samples=1, nxi=nxi, nd=3, steps=a.steps,
                device=str(a.device), cfg_weight=a.cfg_weight, generator=g)
            curve = curve_from_cps(cps, pts=a.pts, device=a.device)     # (1,T,3)

            # ── Ergodischer Fehler gegen die projizierte Dichte ──────────
            c = trajectory_coeffs(curve, k_idx)
            phi = target_coeffs_from_particles(parts.unsqueeze(0), k_idx, True)
            erg = float((Lam * (c - phi) ** 2).sum())

            # ── Abdeckung: getroffene Oberflaeche -> naechste Bahnstelle ─
            m = w_s > 1e-3
            if m.sum() > 4:
                tgt = torch.from_numpy(pts_s[m]).float().to(a.device)
                ww = torch.from_numpy(w_s[m]).float().to(a.device)
                dmin = torch.cdist(tgt, curve[0]).min(dim=1).values
                cov = float((dmin * ww).sum() / ww.sum().clamp(min=1e-9))
            else:
                cov = float('nan')

            # ── Standoff und Blickrichtung ──────────────────────────────
            ps = ParticleSurface(parts.unsqueeze(0), mu_thresh=a.mu_thresh)
            dist = ps.distance(curve)[0]                                # (T,)
            so_m, so_s = float(dist.mean()), float(dist.std())
            point_deg = float('nan')
            if rot6 is not None and rot6.numel():
                Rm = rot6d_to_matrix(rot6.reshape(-1, 6)).reshape(1, nxi, 3, 3)
                Bc = torch.from_numpy(
                    bspline_basis_matrix(nxi, a.pts, 5)).float().to(a.device)
                ax = sensor_axis(Rm, axis=2)                            # (1,nxi,3)
                axc = torch.einsum('pi,kid->kpd', Bc, ax)
                axc = axc / axc.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                tgt_dir = ps.direction(curve)
                cosang = (axc * tgt_dir).sum(-1).clamp(-1, 1)
                point_deg = float(torch.rad2deg(torch.acos(cosang)).mean())

            plen = float((curve[0, 1:] - curve[0, :-1]).norm(dim=-1).sum())
            rows.append(dict(shape=name, surface=key, erg=erg, coverage=cov,
                             standoff=so_m, standoff_sd=so_s,
                             pointing_deg=point_deg, path_len=plen,
                             hit_frac=hit))
            print(f"    {s.label:22s} erg={erg:.5f} cov={cov:.4f} "
                  f"standoff={so_m:.3f}±{so_s:.3f} "
                  f"blick={point_deg:6.1f}°  L={plen:.2f}")

            # Eine Auswahl der Oberflaeche zum Zeichnen — mit Vorrang fuer die
            # beschrifteten Punkte, damit die Dichte nicht wegsubsampelt wird.
            rs = np.random.default_rng(a.seed + si)
            lit = np.flatnonzero(w_s > 1e-3)
            dark = np.flatnonzero(w_s <= 1e-3)
            n_lit = min(len(lit), int(a.store_points * 0.6))
            n_dark = min(len(dark), a.store_points - n_lit)
            keep = np.concatenate([rs.choice(lit, n_lit, replace=False),
                                   rs.choice(dark, n_dark, replace=False)])
            dump['eintraege'].append(dict(
                shape=name, surface=key,
                bahn=curve[0].detach().cpu().numpy().round(4).tolist(),
                rot6=(rot6[0].detach().cpu().numpy().round(4).tolist()
                      if rot6 is not None else None),
                flaeche=pts_s[keep].round(4).tolist(),
                gewicht=w_s[keep].round(4).tolist(),
                partikel=parts_np[::8].round(4).tolist(),
                metrik=rows[-1]))

    import csv
    cp = os.path.join(a.out_dir, 'metriken.csv')
    with open(cp, 'w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)
    with open(os.path.join(a.out_dir, 'bahnen.json'), 'w') as f:
        json.dump(dump, f)
    print(f"\n  [csv] {cp}\n  [json] {a.out_dir}/bahnen.json")

    print("\n" + "=" * 78)
    print(f"{'Oberflaeche':22s} {'Ergodisch':>10s} {'Abdeckung':>10s} "
          f"{'Standoff':>10s} {'Blick':>8s} {'Laenge':>8s}")
    print("-" * 78)
    for key in a.surfaces:
        sel = [r for r in rows if r['surface'] == key]
        f = lambda k: float(np.nanmean([r[k] for r in sel]))
        print(f"{surf[key].label:22s} {f('erg'):10.5f} {f('coverage'):10.4f} "
              f"{f('standoff'):10.3f} {f('pointing_deg'):7.1f}° {f('path_len'):8.2f}")
    print("=" * 78)
    print(f"Gesamtzeit {time.perf_counter() - t0:.0f} s")


if __name__ == '__main__':
    main()
