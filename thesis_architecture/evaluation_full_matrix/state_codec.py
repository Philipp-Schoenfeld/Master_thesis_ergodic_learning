r"""
state_codec.py -- compact storage of logged SVGD states (numpy + zlib only)
===========================================================================
Control points are quantised to 16 bit over [Q_LO, Q_HI] (step 3.1e-5, far below
any sensor / solver scale), the first state is stored absolute, the following
ones as int16 successive differences (int32 if one does not fit), then zlib.
Measured on real SVGD logs: ~10x smaller than float32 (about 29 KB for the
1501 states of one candidate).
"""
import zlib

import numpy as np

#: control points are quantised to 16 bit over this range (step 3.1e-5, i.e. far
#: below any sensor/solver scale). CFM control points can leave [0, 1] slightly.
Q_LO, Q_HI = -0.5, 1.5


def pack_states(cps):
    """(n_states, nxi, 2) float -> bytes. 16-bit quantisation, first state
    absolute, then int16 successive differences (int32 if one does not fit),
    zlib. Reconstruction error <= (Q_HI - Q_LO) / 65535 / 2 per coordinate."""
    a = np.asarray(cps, dtype=np.float64)
    q = np.rint((np.clip(a, Q_LO, Q_HI) - Q_LO) / (Q_HI - Q_LO) * 65535.0).astype(np.int32)
    d = np.diff(q, axis=0)
    wide = bool(d.size and (d.min() < -32768 or d.max() > 32767))
    dd = d.astype(np.int32 if wide else np.int16)
    head = np.array([1 if wide else 0], dtype=np.uint8).tobytes()
    return head + zlib.compress(q[0].astype(np.uint16).tobytes() + dd.tobytes(), 6)


def unpack_states(blob, n_states, nxi):
    raw = zlib.decompress(blob[1:])
    wide = blob[0] == 1
    n0 = nxi * 2 * 2
    q0 = np.frombuffer(raw[:n0], dtype=np.uint16).astype(np.int32).reshape(nxi, 2)
    d = np.frombuffer(raw[n0:], dtype=np.int32 if wide else np.int16).astype(np.int32)
    d = d.reshape(n_states - 1, nxi, 2)
    q = np.concatenate([q0[None], q0[None] + np.cumsum(d, axis=0)], axis=0)
    return (q.astype(np.float64) / 65535.0 * (Q_HI - Q_LO) + Q_LO).astype(np.float32)
