#!/usr/bin/env python3
"""
eval_dfl_checkpoint.py
=======================
Standalone holdout evaluation for a flow_matching_runner_ergodic.py checkpoint
(SpectralCrossAttnFlowNetwork, incl. the DFL-trained lambda0 head) -- no
training loop, just load + generate + plot, reusing the exact same
`_save_viz`/`visualise_set` code path the training script itself uses
periodically, so the output is identical in style to the in-training plots.

`visualize_checkpoint.py` / `model_zoo.py` do NOT support this architecture
(they only dispatch particle-conditioned checkpoints) -- this script exists
because of that gap, not as a replacement for them.

Usage:
  python eval_dfl_checkpoint.py --checkpoint checkpoints/cond_spectral_crossattn_ergodic_dfl_..._ep0433.pt
"""
import argparse
import os

import torch

from flow_matching_runner_ergodic import (
    _load_shapes, _save_viz, trajectory_to_spectral,
)
from flow_matching_cond_spectral_crossattn import SpectralCrossAttnFlowNetwork


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tag', default='checkpoint_eval',
                   help="Suffix for the output filename, so this never "
                        "overwrites an in-training periodic viz for the same epoch")
    p.add_argument('--n_gen', type=int, default=5)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--cfg_weight', type=float, default=2.0)
    p.add_argument('--bspline_pts', type=int, default=512)
    p.add_argument('--bspline_deg', type=int, default=5)
    p.add_argument('--copies_per_char', type=int, default=75,
                   help="Only used for the output filename (matches the "
                        "training run's naming), has no effect on evaluation")
    p.add_argument('--p_flip', type=float, default=0.0,
                   help="Only used for the output filename")
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  epoch={ckpt.get('epoch')}  loss={ckpt.get('loss')}  "
          f"dfl_weight={ckpt.get('dfl_weight')}")

    # Pull architecture hyperparams straight from the checkpoint, so this
    # always matches what was actually trained, not a guessed default.
    args.nxi = ckpt['nxi']; args.nd = ckpt['nd']; args.D = ckpt['D']; args.S = ckpt['S']
    args.n_lambda = ckpt['n_lambda']; args.predict_lambda = ckpt['predict_lambda']
    args.p_drop = ckpt.get('p_drop', 0.1)

    model = SpectralCrossAttnFlowNetwork(
        nxi=args.nxi, nd=args.nd, D=args.D, S=args.S,
        n_lambda=args.n_lambda, predict_lambda=args.predict_lambda,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  nxi={args.nxi} D={args.D} S={args.S} n_lambda={args.n_lambda} "
          f"predict_lambda={args.predict_lambda}")

    _, holdout_shapes, _, holdout_densities = _load_shapes(args.nxi)
    holdout_spec = {lbl: trajectory_to_spectral(base, S=args.S)
                    for lbl, base in holdout_shapes.items()}
    print(f"  {len(holdout_shapes)} holdout shapes")

    ep_num = ckpt.get('epoch', -1) + 1
    viz_path = _save_viz(
        model, holdout_shapes, holdout_spec, holdout_densities,
        f"Holdout Evaluation — checkpoint epoch {ep_num}", ep_num, args, device,
        tag=args.tag,
    )
    print(f"Saved -> {viz_path}")


if __name__ == '__main__':
    main()
