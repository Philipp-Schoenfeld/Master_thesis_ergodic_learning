r"""
net_zoo3d.py
============
Registry of the trained 3D planner networks that the control centre can switch between,
each with a short description (3D counterpart of ``exploration/common/net_zoo.py``).

Checkpoints are looked up by file-name pattern in ``transfer/`` (and its sub-folders) and in
``3D_ergodic_learning/checkpoints/``, so a freshly pulled checkpoint is picked up without touching
the code; an entry whose file is not on disk is simply not offered.  A checkpoint that matches no
entry (e.g. ``main.py some.pt``) still works: it gets a generic entry whose description is built
from the metadata stored in the file.

All entries are 3D SE(3) networks (25 control points, 512 surface particles) and share the
``Planner3D`` interface.  They were trained on different datasets and losses, so the choice changes
the plans for the same target (see the descriptions).
"""

import glob
import os
import re
from dataclasses import dataclass

import paths3d


@dataclass(frozen=True)
class NetEntry:
    key: str
    label: str            # short name for the radio list
    pattern: str          # regex searched in the checkpoint's file name
    description: str      # two or three lines for the selection window


ENTRIES = (
    NetEntry(
        'nooffset200', 'U-Net D384, no-offset-200 (default)',
        r'^nooffset200_flow3d.*_ep0300\.pt$',
        'Start-conditioned U-Net (D384) with SE(3) orientation head, 300 epochs on the no-offset-200\n'
        'surface DB, Fourier ergodic loss (w=100, K=6) at the beam footprint. Best model of the\n'
        'earlier 3D comparison on that DB (the GUI default).'),
    NetEntry(
        'oriw3x', 'U-Net D384, orientation loss x3',
        r'oriw3x',
        'As the no-offset-200 net but with three times the orientation loss weight (0.036; run\n'
        'stopped early): the sensor axis should follow the surface normal more strictly.'),
    NetEntry(
        'startfl22', 'U-Net D384, start-cond., mixed shapes',
        r'startfl22',
        'Start-conditioned U-Net trained on a surface mix that includes plane entries\n'
        '(MIX-ebene=0.25), Fourier ergodic loss. Variant of the no-offset-200 family.'),
    NetEntry(
        'surfb', 'U-Net D384, no start (long run)',
        r'surfB_lang',
        'Not start-conditioned (plans from the null-start token), 1750 epochs on the surface DB,\n'
        'Fourier ergodic loss (w=100, K=6). Longest-trained 3D net; the first control point is not\n'
        'pinned to the robot position.'),
    NetEntry(
        'flaechen_old', 'U-Net, first surface net (old)',
        r'^netz3d_flaechen\.pt$',
        'The first network trained on surface particles (no start conditioning). Kept as the\n'
        'oldest reference.'),
    NetEntry(
        'lm630unet', 'U-Net D512, LM DB, Sinkhorn λ=630',
        r'lm630unet.*ep0750',
        'U-Net (D512) trained 750 epochs on the projected LM-teacher DB with the debiased Sinkhorn\n'
        'loss (w=630) at the footprint, start held by inpainting, control-point targets (LSQ fit).'),
    NetEntry(
        'lm630dit', 'DiT-B (D768), LM DB, Sinkhorn λ=630',
        r'lm630dit.*ep0750',
        'Diffusion transformer (adaLN-Zero, 12 blocks, 12 heads, D768), same data / loss / schedule\n'
        'as the U-Net above, so the pair isolates the backbone. 750 epochs.'),
    NetEntry(
        'dits_old', 'DiT-S (D384), old LM DB',
        r'dbcmpold.*_ep0150',
        'Small DiT (12 blocks, 6 heads, D384), 150 epochs on the projected LM DB (teacher paths on\n'
        'the surface). Together with the U-Net D512, the best active-sensing network on 4.11.'),
    NetEntry(
        'dits_ref', 'DiT-S (D384), refined LM DB',
        r'dbcmpref.*_ep0150',
        'Same DiT-S, trained on the refined DB (teacher paths re-optimised on the training\n'
        'objective, standoff 0.12). Lowest offline Sinkhorn error but longer paths and worse pointing.'),
)


def _candidate_files():
    t = os.path.join(paths3d.ROOT, 'transfer')
    files = (glob.glob(os.path.join(t, '*.pt')) + glob.glob(os.path.join(t, '*', '*.pt'))
             + glob.glob(os.path.join(paths3d.D3, 'checkpoints', '*.pt')))
    # old_models/ holds superseded copies; the dated folders hold the 2D nets (no pattern matches)
    return sorted(f for f in files if os.path.basename(os.path.dirname(f)) != 'old_models')


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
    if ck.get('epoch') is not None:
        bits.append('%s epochs' % (ck['epoch'] + 1))
    desc = 'Custom checkpoint %s\n%s' % (os.path.basename(path), ', '.join(bits) or 'no metadata')
    return NetEntry('custom', 'Custom (command line)', re.escape(os.path.basename(path)), desc)
