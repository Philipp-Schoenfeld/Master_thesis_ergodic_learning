r"""Static PNG snapshots of the 3D surface-coverage holdout results.

Reads the already-generated bahnen.json / meshes.json (no re-inference —
the checkpoint's weights are unchanged, verified byte-identical to
surf_flow3d_particle_ergodic_surfB_lang_..._ep1750.pt) and renders one PNG
per (shape, surface) entry: white shaded mesh at 50% opacity, target-density
points colored by weight (white->inferno), generated trajectory as a thick
neon-green line, matching the project's visualization style guide. Uses
Plotly/Kaleido rather than matplotlib's mplot3d, whose Poly3DCollection does
not composite semi-transparent faces correctly (points and lines behind a
50%-alpha mesh render as fully hidden instead of showing through).
"""
import argparse
import json
import os

import numpy as np
import plotly.graph_objects as go

WHITE_INFERNO = [
    [0.0, '#ffffff'], [0.25, '#fee0b6'], [0.5, '#fc8d59'],
    [0.75, '#b30000'], [1.0, '#1A1A2E'],
]


def orthogonal_camera(v, f, flaeche, gewicht, dist=2.05):
    """Camera looking straight-on (orthogonal) at where most of the target
    density sits, instead of a fixed angle per surface type.

    Uses the mesh's own local surface normal near the density-weighted
    centroid, not the direction from the mesh center to that centroid — for
    a flat plane those disagree completely: the centroid lies *in* the
    plane, so a center-to-centroid vector is edge-on (viewing direction
    orthogonal to what the camera should look orthogonal to). The true local
    normal, found from face normals near the centroid, is constant across a
    flat surface and radially outward on a convex one, giving a genuinely
    face-on view of the coverage target in both cases.
    """
    w = gewicht / max(gewicht.sum(), 1e-9)
    centroid = (flaeche * w[:, None]).sum(axis=0)

    tris = v[f]
    face_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    fn_norm = np.linalg.norm(face_normals, axis=1, keepdims=True)
    fn_norm[fn_norm == 0] = 1.0
    face_normals /= fn_norm
    face_centers = tris.mean(axis=1)

    d2 = ((face_centers - centroid) ** 2).sum(axis=1)
    sigma2 = max(np.median(d2), 1e-9) * 0.15
    kernel = np.exp(-d2 / (2 * sigma2))
    ksum = kernel.sum()
    if ksum < 1e-9:
        direction = np.array([0.85, -0.85, 0.5])
    else:
        direction = (face_normals * kernel[:, None]).sum(axis=0) / ksum
        n = np.linalg.norm(direction)
        direction = direction / n if n > 1e-6 else np.array([0.85, -0.85, 0.5])
        # The mesh normal can point either way; face the side the density
        # (and therefore the trajectory) actually sits on.
        mesh_ctr = v.mean(axis=0)
        if np.dot(centroid - mesh_ctr, direction) < 0:
            direction = -direction

    eye = dict(x=float(direction[0]) * dist, y=float(direction[1]) * dist,
               z=float(direction[2]) * dist)
    # A near-vertical view direction makes the default z-up vector
    # degenerate (it would point almost straight at the camera).
    up = (dict(x=0, y=1, z=0) if abs(direction[2]) > 0.9
          else dict(x=0, y=0, z=1))
    return eye, up


def render_entry(entry, mesh, png_path=None, html_path=None,
                  html_offline=False, surf_alpha=0.5, width=980, height=980):
    v = np.array(mesh['v'], dtype=np.float64).reshape(-1, 3) / 1000.0
    f = np.array(mesh['f'], dtype=np.int64).reshape(-1, 3)
    bahn = np.array(entry['bahn'], dtype=np.float64).reshape(-1, 3)
    flaeche = np.array(entry['flaeche'], dtype=np.float64).reshape(-1, 3)
    gewicht = np.array(entry['gewicht'], dtype=np.float64)

    order = np.argsort(gewicht)
    sizes = 2.2 + 7.5 * (gewicht[order] / max(gewicht.max(), 1e-9)) ** 1.3

    fig = go.Figure()

    fig.add_trace(go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color='#e7e9f0', opacity=surf_alpha,
        lighting=dict(ambient=0.4, diffuse=0.95, specular=0.45,
                       roughness=0.5, fresnel=0.2),
        lightposition=dict(x=200, y=-150, z=250),
        flatshading=False,
        name='oberflaeche', showscale=False, hoverinfo='skip',
    ))

    edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    boundary = {}
    for a, b in edges:
        key = (min(a, b), max(a, b))
        boundary[key] = boundary.get(key, 0) + 1
    outline = [k for k, c in boundary.items() if c == 1]
    if outline:
        xs, ys, zs = [], [], []
        for a, b in outline:
            xs += [v[a, 0], v[b, 0], None]
            ys += [v[a, 1], v[b, 1], None]
            zs += [v[a, 2], v[b, 2], None]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='lines',
            line=dict(color='#c7cbd6', width=2.5),
            hoverinfo='skip', name='umriss',
        ))

    fig.add_trace(go.Scatter3d(
        x=flaeche[order, 0], y=flaeche[order, 1], z=flaeche[order, 2],
        mode='markers',
        marker=dict(size=sizes, color=gewicht[order], colorscale=WHITE_INFERNO,
                    opacity=0.9, line=dict(width=0)),
        hoverinfo='skip', name='zieldichte',
    ))

    fig.add_trace(go.Scatter3d(
        x=bahn[:, 0], y=bahn[:, 1], z=bahn[:, 2],
        mode='lines',
        line=dict(color='#00C853', width=9),
        hoverinfo='skip', name='trajektorie',
    ))
    fig.add_trace(go.Scatter3d(
        x=[bahn[0, 0]], y=[bahn[0, 1]], z=[bahn[0, 2]],
        mode='markers', marker=dict(size=6, color='#1565C0'),
        hoverinfo='skip', name='start',
    ))

    allpts = np.vstack([v, bahn])
    lo, hi = allpts.min(0), allpts.max(0)
    ctr = (lo + hi) / 2
    rng = max((hi - lo).max(), 1e-6) * 0.62

    eye, up = orthogonal_camera(v, f, flaeche, gewicht)

    m = entry['metrik']
    title = (f"<b>{entry['shape']}</b> @ {entry['surface']}"
             f"&nbsp;&nbsp;&nbsp;erg={m['erg']:.4f}  cov={m['coverage']:.3f}"
             f"  pointing={m['pointing_deg']:.0f}&deg;")

    fig.update_layout(
        width=width, height=height,
        paper_bgcolor='white', plot_bgcolor='white',
        showlegend=False,
        margin=dict(l=0, r=0, t=54, b=0),
        title=dict(text=title, font=dict(family='IBM Plex Mono, monospace',
                                          size=15, color='#1A1A2E'),
                   x=0.04, xanchor='left'),
        scene=dict(
            xaxis=dict(visible=False, range=[ctr[0] - rng, ctr[0] + rng]),
            yaxis=dict(visible=False, range=[ctr[1] - rng, ctr[1] + rng]),
            zaxis=dict(visible=False, range=[ctr[2] - rng, ctr[2] + rng]),
            aspectmode='cube',
            bgcolor='white',
            camera=dict(eye=eye, up=up),
        ),
    )
    if png_path is not None:
        fig.write_image(png_path, engine='kaleido')
    if html_path is not None:
        fig.write_html(html_path, include_plotlyjs=(True if html_offline else 'cdn'),
                       full_html=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir', required=True)
    p.add_argument('--out_dir', default=None)
    p.add_argument('--surf_alpha', type=float, default=0.5)
    p.add_argument('--surfaces', nargs='+', default=None,
                   help='Nur diese Oberflaeche(n) rendern, z.B. --surfaces kugel')
    p.add_argument('--shapes', nargs='+', default=None,
                   help='Nur diese Form(en), z.B. --shapes A digit_5')
    p.add_argument('--png', action='store_true', default=None,
                   help='PNG schreiben (Standard: an, ausser --html_only)')
    p.add_argument('--html', action='store_true',
                   help='Zusaetzlich eine eigenstaendige, interaktive .html je Kombination schreiben')
    p.add_argument('--html_only', action='store_true',
                   help='Nur .html schreiben, kein PNG (impliziert --html)')
    p.add_argument('--html_offline', action='store_true',
                   help='plotly.js in die .html einbetten (~4.5MB/Datei, kein Internet noetig). '
                        'Ohne dieses Flag laedt die Seite plotly.js von einem CDN.')
    args = p.parse_args()
    out_dir = args.out_dir or os.path.join(args.in_dir, 'bilder')
    write_png = not args.html_only if args.png is None else args.png
    write_html = args.html or args.html_only
    os.makedirs(out_dir, exist_ok=True)

    bahnen = json.load(open(os.path.join(args.in_dir, 'bahnen.json'), encoding='utf-8'))
    meshes = json.load(open(os.path.join(args.in_dir, 'meshes.json'), encoding='utf-8'))

    entries = bahnen['eintraege']
    if args.surfaces:
        entries = [e for e in entries if e['surface'] in args.surfaces]
    if args.shapes:
        entries = [e for e in entries if e['shape'] in args.shapes]

    n = len(entries)
    for i, entry in enumerate(entries):
        surf = entry['surface']
        shape = entry['shape']
        sub = os.path.join(out_dir, surf)
        os.makedirs(sub, exist_ok=True)
        png_path = os.path.join(sub, f'{shape}.png') if write_png else None
        html_path = os.path.join(sub, f'{shape}.html') if write_html else None
        render_entry(entry, meshes[surf], png_path=png_path, html_path=html_path,
                     html_offline=args.html_offline, surf_alpha=args.surf_alpha)
        if (i + 1) % 10 == 0 or i == n - 1:
            print(f'[{i+1}/{n}] {surf}/{shape}')

    print('done ->', out_dir)


if __name__ == '__main__':
    main()
