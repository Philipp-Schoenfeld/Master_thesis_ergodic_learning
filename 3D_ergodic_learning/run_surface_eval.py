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
from tqdm import tqdm

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
from mesh_alignment import TrimeshSDF, SurfaceTangentAlignment, refine_with_alignment


def curve_from_cps(cps, pts=256, deg=5, device='cpu'):
    B = torch.from_numpy(bspline_basis_matrix(cps.shape[1], pts, deg)).float()
    return torch.einsum('pi,kid->kpd', B.to(device), cps.float())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--shapes', type=int, default=12)
    p.add_argument('--shape_names', nargs='+', default=None,
                   help='Nur diese Formen namentlich auswaehlen '
                        '(z.B. --shape_names A organic_0) statt der ersten '
                        '--shapes Formen in Datensatz-Reihenfolge.')
    p.add_argument('--splits', nargs='+', default=['val'],
                   help="Aus welchen Splits Formen geladen werden, z.B. "
                        "'--splits val train' um mit --shape_names auch "
                        "Trainings-Formen (z.B. einzelne Buchstaben) zu erreichen. "
                        "Nur fuer explorative Einzelbilder gedacht — fuer echte "
                        "Holdout-Metriken bei 'val' bleiben.")
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
    p.add_argument('--no_align', action='store_true',
                   help='SE(3)-Tangentenausrichtung (Constraint 5, siehe '
                        'mesh_alignment.py) abschalten -- liefert die reine '
                        'Netz-Bahn wie vor der Ausrichtungs-Nachbearbeitung.')
    p.add_argument('--w_surface', type=float, default=50.0,
                   help='Gewicht des Oberflaechen-Terms in der Ausrichtungs-Energie.')
    p.add_argument('--w_align', type=float, default=1.0,
                   help='Gewicht des Tangenten-Ausrichtungs-Terms.')
    p.add_argument('--align_iters', type=int, default=300)
    p.add_argument('--align_lr', type=float, default=0.2)
    p.add_argument('--align_max_force', type=float, default=0.5)
    a = p.parse_args()
    a.align = not a.no_align

    a.device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(a.seed)

    ck = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    nxi, D = ck.get('nxi', 25), ck.get('D', 384)
    ori = bool(ck.get('orientation', False))
    start_cond = bool(ck.get('start_cond', False))
    model = ParticleCrossAttnFlowNetwork(nxi=nxi, nd=3, D=D,
                                         predict_orientation=ori,
                                         start_cond=start_cond).to(a.device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    if start_cond:
        print("Start conditioning: on (no start point known -> null start token, neutral)")
    print(f"Network: D={D} nxi={nxi} orientation={ori}  Epoch {ck.get('epoch','?')}"
          f"  Loss {ck.get('loss', float('nan')):.4f}")
    print(f"Device: {a.device}")

    # ── Holdout-Formen: die 2D-Dichten, die projiziert werden ────────────
    from shape_library import pdf_on_grid
    # `load_pairs` liefert (Trajektorien, Dichte-Definitionen, Splits), jeweils
    # als Dict ueber den Formnamen. Gebraucht wird hier nur die Definition.
    _, defs, _ = load_pairs(nxi, splits=tuple(a.splits))
    shapes_ = []
    if a.shape_names:
        missing = [nm for nm in a.shape_names if nm not in defs]
        if missing:
            p.error(f"Unknown holdout shape(s): {', '.join(missing)}. "
                    f"Available: {', '.join(sorted(defs))}")
        for nm in a.shape_names:
            d2, _, _ = pdf_on_grid(defs[nm], resolution=a.dens_res)
            d2 = np.asarray(d2, dtype=np.float64)
            shapes_.append((nm, d2 / max(d2.max(), 1e-12)))
    else:
        for nm, df in defs.items():
            d2, _, _ = pdf_on_grid(df, resolution=a.dens_res)
            d2 = np.asarray(d2, dtype=np.float64)
            shapes_.append((nm, d2 / max(d2.max(), 1e-12)))
            if len(shapes_) >= a.shapes:
                break
    print(f"{len(shapes_)} holdout shapes: {', '.join(n for n, _ in shapes_)}")

    k_idx, Lam = make_k_grid(a.erg_K)
    k_idx = torch.tensor(k_idx, device=a.device)
    Lam = torch.tensor(Lam, dtype=torch.float32, device=a.device)

    surf = {k: surfaces.build(k) for k in a.surfaces}
    for k, s in surf.items():
        print(f"  {s.label:22s} {len(s.mesh.faces):6d} triangles \u2014 {s.note}")

    # -- SE(3)-Tangentenausrichtung (Constraint 5, mesh_alignment.py) --
    # Politur der vom Netz erzeugten Kontrollpunkte auf jede der Zielflaechen,
    # damit die Bahn nicht nur naeherungsweise, sondern bis auf numerische
    # Toleranz auf der Oberflaeche liegt und die Tangente in deren lokaler
    # Ebene bleibt (siehe mesh_alignment.py fuer die Herleitung).
    B_align = torch.from_numpy(
        bspline_basis_matrix(nxi, a.pts, 5)).float().to(a.device)
    con_by_surf = {}
    if a.align:
        con_by_surf = {k: SurfaceTangentAlignment(
            TrimeshSDF(s.mesh), w_surface=a.w_surface, w_align=a.w_align)
            for k, s in surf.items()}
        print(f"Ausrichtung an: w_surface={a.w_surface:g} w_align={a.w_align:g} "
              f"iters={a.align_iters} lr={a.align_lr:g}")
    else:
        print("Ausrichtung aus (--no_align): reine Netz-Bahn ohne Nachbearbeitung.")

    def score_curve(curve, rot6):
        """erg/coverage/standoff/pointing/path_len fuer eine dichte Bahn."""
        c = trajectory_coeffs(curve, k_idx)
        phi = target_coeffs_from_particles(parts.unsqueeze(0), k_idx, True)
        erg_ = float((Lam * (c - phi) ** 2).sum())
        m_ = w_s > 1e-3
        if m_.sum() > 4:
            tgt_ = torch.from_numpy(pts_s[m_]).float().to(a.device)
            ww_ = torch.from_numpy(w_s[m_]).float().to(a.device)
            dmin_ = torch.cdist(tgt_, curve[0]).min(dim=1).values
            cov_ = float((dmin_ * ww_).sum() / ww_.sum().clamp(min=1e-9))
        else:
            cov_ = float('nan')
        dist_ = ps.distance(curve)[0]
        so_m_, so_s_ = float(dist_.mean()), float(dist_.std())
        point_deg_ = float('nan')
        if rot6 is not None and rot6.numel():
            Rm_ = rot6d_to_matrix(rot6.reshape(-1, 6)).reshape(1, nxi, 3, 3)
            ax_ = sensor_axis(Rm_, axis=2)
            axc_ = torch.einsum('pi,kid->kpd', B_align, ax_)
            axc_ = axc_ / axc_.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            tgt_dir_ = ps.direction(curve)
            cosang_ = (axc_ * tgt_dir_).sum(-1).clamp(-1, 1)
            point_deg_ = float(torch.rad2deg(torch.acos(cosang_)).mean())
        plen_ = float((curve[0, 1:] - curve[0, :-1]).norm(dim=-1).sum())
        return dict(erg=erg_, coverage=cov_, standoff=so_m_, standoff_sd=so_s_,
                    pointing_deg=point_deg_, path_len=plen_)

    rows, dump = [], {'meta': {k: v for k, v in vars(a).items()
                              if isinstance(v, (int, float, str, bool))},
                      'eintraege': []}
    t0 = time.perf_counter()

    szenen = [(si, name, d2, key)
             for si, (name, d2) in enumerate(shapes_) for key in a.surfaces]
    balken = tqdm(szenen, desc='Scenes', unit='scene',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                             '[{elapsed}<{remaining}, {rate_fmt}]{postfix}')
    for si, name, d2, key in balken:
            balken.set_postfix_str(f'{name}/{key}', refresh=False)
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
            curve_base = curve_from_cps(cps, pts=a.pts, device=a.device)  # (1,T,3)

            # ── Standoff/Blickrichtung brauchen `ps` schon in score_curve ───
            ps = ParticleSurface(parts.unsqueeze(0), mu_thresh=a.mu_thresh)

            # ── SE(3)-Tangentenausrichtung: Politur auf die echte Oberflaeche ──
            if a.align:
                con = con_by_surf[key]
                cps_al = refine_with_alignment(
                    cps, con, B_align, iters=a.align_iters, lr=a.align_lr,
                    max_force=a.align_max_force)
                curve = curve_from_cps(cps_al, pts=a.pts, device=a.device)
                sdf_b, cos_b = con.report(curve_base)
                sdf_a, cos_a = con.report(curve)
            else:
                curve = curve_base
                sdf_b = sdf_a = cos_b = cos_a = float('nan')

            m_basis = score_curve(curve_base, rot6)
            m_align = score_curve(curve, rot6)
            so_m, so_s = m_align['standoff'], m_align['standoff_sd']
            point_deg, plen = m_align['pointing_deg'], m_align['path_len']

            rows.append(dict(
                shape=name, surface=key, hit_frac=hit,
                sdf_max_basis=sdf_b, sdf_max_align=sdf_a,
                cos_tn_basis=cos_b, cos_tn_align=cos_a,
                erg_basis=m_basis['erg'], erg=m_align['erg'],
                coverage_basis=m_basis['coverage'], coverage=m_align['coverage'],
                standoff_basis=m_basis['standoff'], standoff=so_m, standoff_sd=so_s,
                pointing_deg_basis=m_basis['pointing_deg'], pointing_deg=point_deg,
                path_len_basis=m_basis['path_len'], path_len=plen))
            balken.write(f"  [{name:14s}] {s.label:22s} "
                        f"|SDF| {sdf_b:.4f}->{sdf_a:.4f}  "
                        f"|cos| {cos_b:.3f}->{cos_a:.3f}  "
                        f"erg {m_basis['erg']:.5f}->{m_align['erg']:.5f}  "
                        f"cov {m_basis['coverage']:.4f}->{m_align['coverage']:.4f}  "
                        f"standoff={so_m:.3f}±{so_s:.3f} "
                        f"point={point_deg:6.1f}°  L={plen:.2f}")

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
    print(f"{'Surface':22s} {'Ergodic':>10s} {'Coverage':>10s} "
          f"{'Standoff':>10s} {'Point':>8s} {'Length':>8s}")
    print("-" * 78)
    for key in a.surfaces:
        sel = [r for r in rows if r['surface'] == key]
        f = lambda k: float(np.nanmean([r[k] for r in sel]))
        print(f"{surf[key].label:22s} {f('erg'):10.5f} {f('coverage'):10.4f} "
              f"{f('standoff'):10.3f} {f('pointing_deg'):7.1f}\u00b0 {f('path_len'):8.2f}")
    print("=" * 78)
    if a.align:
        print(f"\n{'Surface':22s} {'|SDF| vorher':>13s} {'|SDF| nachher':>14s} "
              f"{'|cos(t,n)| vorher':>18s} {'|cos(t,n)| nachher':>19s}")
        print("-" * 78)
        for key in a.surfaces:
            sel = [r for r in rows if r['surface'] == key]
            f = lambda k: float(np.nanmean([r[k] for r in sel]))
            print(f"{surf[key].label:22s} {f('sdf_max_basis'):13.4f} "
                  f"{f('sdf_max_align'):14.4f} {f('cos_tn_basis'):18.3f} "
                  f"{f('cos_tn_align'):19.3f}")
        print("=" * 78)
    print(f"Total time {time.perf_counter() - t0:.0f} s")


if __name__ == '__main__':
    main()
