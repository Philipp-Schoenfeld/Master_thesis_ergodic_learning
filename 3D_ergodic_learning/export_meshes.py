r"""
export_meshes.py
================
Die sieben Oberflaechen als vereinfachte Dreiecksnetze fuer den Betrachter.

Bisher zeigte der Betrachter nur Punktwolken — man sah durch die Formen
hindurch, und Zieldichte wie Bahn verloren sich im Gewimmel. Mit einem
gefuellten weissen Koerper liegt beides auf einer Flaeche statt in einer Wolke.

Die Netze sind je Oberflaeche *einmal* gespeichert, nicht je Szene: alle 84
Szenen einer Flaeche teilen sich dasselbe Netz. Deshalb kostet der gefuellte
Koerper fast nichts an Seitengroesse.
"""
import argparse, json, os, sys
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
import surfaces                                            # noqa: E402


def vereinfache(mesh, ziel):
    """Auf etwa `ziel` Dreiecke bringen. Ebenen und Wuerfel bleiben, wie sie sind."""
    import open3d as o3d
    if len(mesh.faces) <= ziel:
        return np.asarray(mesh.vertices), np.asarray(mesh.faces)
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces)))
    m = m.simplify_quadric_decimation(int(ziel))
    m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()
    return np.asarray(m.vertices), np.asarray(m.triangles)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ziel', type=int, default=2200)
    p.add_argument('--out', default='results/surfaces/meshes.json')
    a = p.parse_args()

    out = {}
    for k in surfaces.KEYS:
        s = surfaces.build(k)
        # Ebenen brauchen kein feines Netz — zwei Dreiecke reichen fuer die
        # Flaeche, aber ein grobes Gitter haelt die Beleuchtung ruhiger.
        ziel = 200 if k.startswith('ebene') else a.ziel
        V, F = vereinfache(s.mesh, ziel)
        q = np.clip(np.round(V * 1000), 0, 1000).astype(int)
        out[k] = dict(v=q.flatten().tolist(), f=F.flatten().tolist(),
                      n=len(V), t=len(F))
        print(f'  {k:16s} {len(s.mesh.faces):7d} → {len(F):5d} Dreiecke')

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'\n[json] {a.out}  ({os.path.getsize(a.out)/2**20:.2f} MiB)')


if __name__ == '__main__':
    main()
