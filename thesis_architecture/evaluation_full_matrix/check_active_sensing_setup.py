"""Pre-flight check for the active-sensing evaluation jobs (comparison / timing / convergence).

Run it once on the machine that will start the jobs (login node is fine, it only reads files and
loads checkpoints on the CPU if no GPU is visible). It reports, for every model the jobs need,
whether the checkpoint can be found where the code looks for it, and with `--link` it tries to
fill gaps by symlinking matching files from `thesis_architecture/checkpoints/` (the cluster
training output folder) into `<repo>/transfer/` (where `find_checkpoint` searches).

    python check_active_sensing_setup.py            # report only
    python check_active_sensing_setup.py --link     # also create symlinks for missing checkpoints
    python check_active_sensing_setup.py --load     # additionally load every planner once (slow-ish)
"""
import argparse
import glob
import os
import re
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.insert(0, _arch)

#: models needed by the three job families (comparison: selfsup/old500; sweep+timing: also dit630,
#: unet630; optional extras only used by the timing test)
REQUIRED = ('unet630', 'dit630', 'old500', 'selfsup')
OPTIONAL = ('unet0', 'unet1890', 'lm1300')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--link', action='store_true',
                   help='Symlink missing checkpoints from thesis_architecture/checkpoints/ into <repo>/transfer/.')
    p.add_argument('--load', action='store_true', help='Also instantiate every planner once.')
    a = p.parse_args()

    import sample_annealed_mission_overview as ov
    import selfsup_planner
    from exploration.common import net_zoo

    transfer = net_zoo.TRANSFER
    print(f'repo transfer folder : {transfer} (exists: {os.path.isdir(transfer)})')
    ckpt_dir = os.path.join(_arch, 'checkpoints')
    print(f'training checkpoints : {ckpt_dir} (exists: {os.path.isdir(ckpt_dir)})')
    ok = True

    for net in REQUIRED + OPTIONAL:
        required = net in REQUIRED
        if net == 'selfsup':
            path = selfsup_planner.DEFAULT_SELFSUP_CKPT
            found = os.path.exists(path)
            if not found and a.link:
                name = os.path.basename(path)
                hits = glob.glob(os.path.join(ckpt_dir, name))
                if hits:
                    os.makedirs(transfer, exist_ok=True)
                    os.symlink(hits[0], os.path.join(transfer, name))
                    path, found = os.path.join(transfer, name), True
        else:
            try:
                path, found = ov.find_checkpoint(net), True
            except FileNotFoundError:
                path, found = None, False
                if a.link and os.path.isdir(ckpt_dir):
                    pat = re.compile(ov.NETS[net][1])
                    hits = [f for f in glob.glob(os.path.join(ckpt_dir, '*.pt'))
                            if pat.search(os.path.basename(f)) and 'flow3d' not in os.path.basename(f)]
                    if hits:
                        best = max(hits, key=os.path.getmtime)
                        os.makedirs(transfer, exist_ok=True)
                        dst = os.path.join(transfer, os.path.basename(best))
                        if not os.path.exists(dst):
                            os.symlink(best, dst)
                        path, found = ov.find_checkpoint(net), True
        tag = 'OK     ' if found else ('MISSING' if required else 'missing (optional)')
        print(f'  {net:9s} {tag}  {os.path.basename(path) if path else ""}')
        ok &= found or not required

    # shapes / datasets / spline library
    try:
        from sample_annealed_mission import load_truth, rsc  # noqa: F401
        names, _ = load_truth(n=3, split='val', resolution=rsc.TRUTH_RES, device='cpu')
        print(f'  holdout shapes loadable: {list(names)[:3]} ...')
    except Exception as e:                                                        # noqa: BLE001
        ok = False
        print(f'  holdout shapes NOT loadable: {e!r}')
    try:
        import bsplinax.bspline  # noqa: F401
        print('  bsplinax import: OK')
    except Exception as e:                                                        # noqa: BLE001
        ok = False
        print(f'  bsplinax import FAILED: {e!r}')
    import torch
    print(f'  torch {torch.__version__}, cuda available: {torch.cuda.is_available()}')

    if a.load and ok:
        import argparse as ap
        import run_exploration_comparison as rec
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for net in REQUIRED:
            if net in ('unet630', 'dit630', 'old500', 'selfsup'):
                rec.build_planner(net, ap.Namespace(selfsup_ckpt=None), dev)
                print(f'  planner {net}: loaded')
    print('SETUP OK' if ok else 'SETUP INCOMPLETE - see MISSING lines')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
