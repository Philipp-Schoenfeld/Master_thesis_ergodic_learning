r"""
compare_unknown_structured_cluster.py
=======================================
Cluster (GPU) old-vs-new checkpoint comparison on the val split of
ergodic_dataset_improved_final.db: same ergodic Fourier error (K=10) as
compare_checkpoints_cluster.py, but splits the 72 val target densities into
"unknown-region variant" (shape_name contains '#unk') vs "plain" (base shape,
no injected unknown region) and reports aggregate stats for each group
separately, plus the full per-item table as JSON.

--new_ckpt lets this be pointed at any checkpoint (e.g. an intermediate
pilot-run checkpoint), so it doubles as a cheap validation gate: rerun it
against the latest checkpoint of a finetune to check whether the plain-shape
regression seen in the FTSHORT15x200 run (see compare_unknown_structured_results.json)
has been fixed before committing to a long run.

    python compare_unknown_structured_cluster.py --new_ckpt checkpoints/..._ep0040.pt --tag pilot_ep40
"""
import argparse
import sys, json, sqlite3, os, time
import numpy as np
import torch

PROJ = '/home/stud_schonfeld/Master_thesis/thesis_architecture'
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, 'ergodic_dataset_generator'))

from model_zoo import load_model, generate
from shape_library import pdf_on_grid
from ergodic_energy_torch import make_k_grid, target_coeffs_from_grid, coeffs_from_points
from obstacles import bspline_basis_matrix
from flow_matching_runner_start import sample_particles

DB = os.path.join(PROJ, 'ergodic_dataset_generator', 'ergodic_dataset_improved_final.db')
OLD_CKPT_DEFAULT = os.path.join(PROJ, 'checkpoints',
    'cond_particles_crossattn_flow_matching_particle_ergodic_date_08_26_04h30min_nxi25_D384_N256_C75_flip0.0_START_FLAT400_L500_ERGLOSS-SINKHORN-w1300-blur0.05-tp2_final.pt')
NEW_CKPT_DEFAULT = os.path.join(PROJ, 'checkpoints',
    'cond_particles_crossattn_flow_matching_particle_ergodic_date_10_07_21h32min_nxi25_D384_N256_C5_flip0.0_START_FLAT1200_FTSHORT15x200_ERGLOSS-SINKHORN-w1300-blur0.05-tp2_final.pt')

N_SAMPLES = 3
STEPS = 100
K = 10
GRID_RES = 128
DEVICE = 'cuda'


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--old_ckpt', type=str, default=OLD_CKPT_DEFAULT)
    p.add_argument('--new_ckpt', type=str, default=NEW_CKPT_DEFAULT)
    p.add_argument('--tag', type=str, default=None,
                   help='Suffix for the output JSON filename, e.g. "pilot_ep40". '
                        'Default: derived from --new_ckpt basename.')
    args = p.parse_args()

    new_ckpt = args.new_ckpt if os.path.isabs(args.new_ckpt) else os.path.join(PROJ, args.new_ckpt)
    old_ckpt = args.old_ckpt if os.path.isabs(args.old_ckpt) else os.path.join(PROJ, args.old_ckpt)
    tag = args.tag or os.path.splitext(os.path.basename(new_ckpt))[0]
    out_json = os.path.join(PROJ, f'compare_unknown_structured_results_{tag}.json')

    device = torch.device(DEVICE)
    print("Loading checkpoints...", flush=True)
    print(f"  old_ckpt: {old_ckpt}")
    print(f"  new_ckpt: {new_ckpt}")
    model_old, kind_old, meta_old = load_model(old_ckpt, device)
    model_new, kind_new, meta_new = load_model(new_ckpt, device)
    print(f"  old: start_cond={meta_old['start_cond']} length_cond={meta_old['length_cond']}")
    print(f"  new: start_cond={meta_new['start_cond']} length_cond={meta_new['length_cond']}")

    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT shape_name, density_params, x0 FROM ergodic_pairs WHERE split='val'"
    ).fetchall()
    print(f"  {len(rows)} val target densities", flush=True)

    k_idx, Lambda = make_k_grid(K)
    k_idx_t = torch.from_numpy(k_idx).float().to(device)
    Lambda_t = torch.from_numpy(Lambda).float().to(device)
    basis = torch.from_numpy(bspline_basis_matrix(25, 256, 5)).float().to(device)

    def err_for(model, kind, meta, particles, start, phi_k):
        cps, _ = generate(model, kind, particles, N_SAMPLES, meta, STEPS, device, seed=0, start=start)
        curve = torch.einsum('ti,bid->btd', basis, cps.float())
        c = coeffs_from_points(curve, k_idx_t)
        diff = c - phi_k.unsqueeze(0)
        e = 0.5 * (Lambda_t * diff.pow(2)).sum(dim=-1)
        return e.mean().item()

    results = []
    t0 = time.time()
    for i, (name, dp_str, x0_str) in enumerate(rows):
        shape_def = json.loads(dp_str)
        x0 = json.loads(x0_str)

        d_map, _, _ = pdf_on_grid(shape_def, resolution=GRID_RES)
        phi_k = target_coeffs_from_grid(
            torch.from_numpy(np.asarray(d_map)).float().to(device), k_idx_t)

        grid_t = torch.from_numpy(np.asarray(d_map)).float().unsqueeze(0)
        idx_t = torch.tensor([0], dtype=torch.long)
        particles = sample_particles(grid_t, idx_t, 256, device='cpu', mode='uniform')[0]
        start_t = torch.tensor(x0, dtype=torch.float32)

        e_old = err_for(model_old, kind_old, meta_old, particles, start_t, phi_k)
        e_new = err_for(model_new, kind_new, meta_new, particles, start_t, phi_k)
        is_unknown = '#unk' in name
        results.append({'name': name, 'old': e_old, 'new': e_new, 'unknown': is_unknown})
        dt = time.time() - t0
        print(f"  [{i+1}/{len(rows)}] {name:<28} old={e_old:.4f}  new={e_new:.4f}  "
              f"{'new better' if e_new < e_old else 'old better'}  "
              f"({'unk' if is_unknown else 'plain'})  [{dt:.0f}s]", flush=True)

    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full per-item results to {out_json}")

    def report(label, subset):
        old_e = np.array([r['old'] for r in subset])
        new_e = np.array([r['new'] for r in subset])
        n_better = int((new_e < old_e).sum())
        rel = (new_e.mean() - old_e.mean()) / old_e.mean()
        print(f"\n=== {label} (n={len(subset)}) ===")
        print(f"  old: mean={old_e.mean():.4f}  median={np.median(old_e):.4f}")
        print(f"  new: mean={new_e.mean():.4f}  median={np.median(new_e):.4f}")
        print(f"  new better in {n_better}/{len(subset)} ({n_better/len(subset):.0%})")
        print(f"  relative change (new vs old): {rel:+.1%}")

    report("ALL", results)
    report("UNKNOWN-REGION variants (#unk)", [r for r in results if r['unknown']])
    report("PLAIN (no injected unknown region)", [r for r in results if not r['unknown']])


if __name__ == '__main__':
    main()
