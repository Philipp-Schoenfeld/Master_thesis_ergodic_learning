r"""
net_zoo.py
==========
Registry of the trained 2D planner networks that the interactive GUI can switch between,
each with a short description.

Checkpoints are looked up in ``transfer/`` and its dated sub-folders (``transfer/2026-10-28/``)
by file-name pattern, so a freshly pulled run folder is picked up without touching the code.
A checkpoint that matches no entry (e.g. ``--ckpt`` pointing elsewhere) is still usable: it gets
a generic entry whose description is built from the metadata stored in the file.

All entries are 2D, 25 control points, 256 particles (what ``apply_cfm_belief.CfmPlanner`` and
the GUI assume); ``CfmPlanner`` picks the architecture from the flags in the checkpoint.
"""

import glob
import os
import re
from dataclasses import dataclass

_arch = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRANSFER = os.path.join(os.path.dirname(_arch), 'transfer')


@dataclass(frozen=True)
class NetEntry:
    key: str
    label: str            # short name for the radio list
    pattern: str          # regex searched in the checkpoint's file name
    description: str      # one or two lines for the selection window


ENTRIES = (
    NetEntry(
        'lm1300', 'U-Net, λ=1300 (default)',
        r'LMTEACHER|netz2d_startpunkt_lm_inpaint',
        'Start-conditioned U-Net (87M params), start held fixed by inpainting. Trained 600 epochs\n'
        'on the LM-teacher DB (Levenberg-Marquardt trajectories as 25 control points) with the\n'
        'Sinkhorn ergodic loss, weight 1300. Longest-trained model; reference for the ablations.'),
    NetEntry(
        'lam210', 'U-Net, λ=210 (~25 % erg.)',
        r'^(?!.*DIT).*LAMERG25',
        'Lambda ablation: as the default, but 180 epochs and ergodic-loss weight 210, i.e. about\n'
        '25 % of the loss on the ergodic term. Compare with the 630 run (same schedule).'),
    NetEntry(
        'lam630', 'U-Net, λ=630 (~50 % erg.)',
        r'^(?!.*DIT).*LAMERG50',
        'Lambda ablation: 180 epochs, ergodic-loss weight 630 (about 50 % of the loss on the\n'
        'ergodic term). U-Net counterpart of the DiT run below.'),
    NetEntry(
        'dit630', 'DiT, λ=630',
        r'DIT-d12-h6',
        'Diffusion transformer (adaLN-Zero, 12 blocks, 6 heads, 40M params, ~4x faster than the\n'
        'U-Net) with particle cross-attention. Same data, schedule and loss (λ=630) as the U-Net\n'
        'above, so the pair isolates the backbone.'),
    NetEntry(
        'belief_ak', 'U-Net, AK-belief clouds',
        r'BELIEF-AK',
        'Not start-conditioned. Trained on particle clouds sampled from AK-GP belief posteriors\n'
        '(32-128 noisy measurements) instead of the true density, Fourier ergodic loss (w=300),\n'
        '500 epochs. Fits the AK belief kernel best.'),
    NetEntry(
        'belief_np', 'U-Net, NP-belief clouds',
        r'BELIEF-NP',
        'Not start-conditioned. Same as the AK-belief net, but the training beliefs come from the\n'
        'neural-process belief (belief_np_date_10_23 checkpoint). Fits the NP belief kernel best.'),
    NetEntry(
        'start_old', 'U-Net, start-cond. (older)',
        r'^netz2d_startpunkt\.pt$',
        'Earlier start-conditioned U-Net (160 epochs, ergodic loss 1300) trained on the SVGD\n'
        'start DB, without start inpainting and without the LM-teacher data.'),
)


def _candidate_files():
    files = glob.glob(os.path.join(TRANSFER, '*.pt')) + glob.glob(os.path.join(TRANSFER, '*', '*.pt'))
    # old_models/ and gross/ hold superseded or non-2D-planner networks
    return sorted(f for f in files if os.path.basename(os.path.dirname(f)) not in ('old_models', 'gross'))


def resolve(entry):
    """Path of the checkpoint for ``entry`` (the newest match), or None if it is not on disk."""
    hits = [f for f in _candidate_files() if re.search(entry.pattern, os.path.basename(f))]
    return max(hits, key=os.path.getmtime) if hits else None


def available():
    """``[(entry, path)]`` for the registered networks that exist locally."""
    out = []
    for e in ENTRIES:
        p = resolve(e)
        if p is not None:
            out.append((e, p))
    return out


def entry_for(path):
    """Registered entry a checkpoint path belongs to, or None."""
    name = os.path.basename(path)
    for e in ENTRIES:
        if re.search(e.pattern, name):
            return e
    return None


def custom_entry(path):
    """Generic entry for a checkpoint that is not registered (description from its metadata)."""
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    bits = []
    if ck.get('start_cond'):
        bits.append('start-conditioned' + (' (inpainted)' if ck.get('start_inpaint') else ''))
    if ck.get('backbone') == 'dit':
        bits.append('DiT backbone')
    if ck.get('lambda_erg'):
        bits.append('ergodic loss %g' % ck['lambda_erg'])
    if ck.get('epochs') or ck.get('epoch') is not None:
        bits.append('%s epochs' % (ck.get('epochs') or ck['epoch'] + 1))
    desc = 'Custom checkpoint %s\n%s' % (os.path.basename(path), ', '.join(bits) or 'no metadata')
    return NetEntry('custom', 'Custom (--ckpt)', re.escape(os.path.basename(path)), desc)
