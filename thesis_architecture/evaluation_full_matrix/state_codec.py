r"""
state_codec.py -- compact storage of logged SVGD states (numpy + zlib only)
===========================================================================
Control points are quantised to 16 bit over [Q_LO, Q_HI] (step 3.1e-5, far below
any sensor / solver scale), the first state is stored absolute, the following
ones as int16 successive differences (int32 if one does not fit), then zlib.
Measured on real SVGD logs: ~10x smaller than float32 (about 29 KB for the
1501 states of one candidate).

`pack_states(..., wide_range=True)` quantises over [Q_LO_WIDE, Q_HI_WIDE]
instead (step ~1.7e-4, still far below sensor/solver scale): early-stopped
candidates (`run_mission_eval_budget_matched.py`, which can log states from
iteration ~50 instead of only the converged ~1500th) are not yet pulled back
into [0, 1] by the SVGD boundary term and can briefly swing well past
[Q_LO, Q_HI] -- silently clipped by the narrow range otherwise. The range
used is encoded in the blob's header byte, so `unpack_states` decodes old and
new blobs alike without needing to be told which was used.
"""
import zlib

import numpy as np

#: control points are quantised to 16 bit over this range (step 3.1e-5, i.e. far
#: below any sensor/solver scale). CFM control points can leave [0, 1] slightly.
Q_LO, Q_HI = -0.5, 1.5
#: wider range for not-yet-converged (early-stopped) states, see module docstring.
Q_LO_WIDE, Q_HI_WIDE = -3.0, 4.0


def pack_states(cps, wide_range=False):
    """(n_states, nxi, 2) float -> bytes. 16-bit quantisation, first state
    absolute, then int16 successive differences (int32 if one does not fit),
    zlib. Reconstruction error <= (q_hi - q_lo) / 65535 / 2 per coordinate."""
    q_lo, q_hi = (Q_LO_WIDE, Q_HI_WIDE) if wide_range else (Q_LO, Q_HI)
    a = np.asarray(cps, dtype=np.float64)
    q = np.rint((np.clip(a, q_lo, q_hi) - q_lo) / (q_hi - q_lo) * 65535.0).astype(np.int32)
    d = np.diff(q, axis=0)
    wide_diff = bool(d.size and (d.min() < -32768 or d.max() > 32767))
    dd = d.astype(np.int32 if wide_diff else np.int16)
    head = np.array([(2 if wide_range else 0) + (1 if wide_diff else 0)], dtype=np.uint8).tobytes()
    return head + zlib.compress(q[0].astype(np.uint16).tobytes() + dd.tobytes(), 6)


def unpack_states(blob, n_states, nxi):
    raw = zlib.decompress(blob[1:])
    head = blob[0]
    wide_diff = bool(head & 1)
    q_lo, q_hi = (Q_LO_WIDE, Q_HI_WIDE) if head & 2 else (Q_LO, Q_HI)
    n0 = nxi * 2 * 2
    q0 = np.frombuffer(raw[:n0], dtype=np.uint16).astype(np.int32).reshape(nxi, 2)
    d = np.frombuffer(raw[n0:], dtype=np.int32 if wide_diff else np.int16).astype(np.int32)
    d = d.reshape(n_states - 1, nxi, 2)
    q = np.concatenate([q0[None], q0[None] + np.cumsum(d, axis=0)], axis=0)
    return (q.astype(np.float64) / 65535.0 * (q_hi - q_lo) + q_lo).astype(np.float32)
