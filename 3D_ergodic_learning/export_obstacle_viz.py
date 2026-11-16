#!/usr/bin/env python3
"""Export an interactive HTML viewer (Plotly) for an obstacle-conditioned checkpoint.

For each selected holdout entry the model generates paths
  (a) without an obstacle (null token) and
  (b) conditioned on a sphere placed on the stored path at several positions.
The viewer lets you pick the shape, the sphere position and the condition, and
rotate / zoom the 3D scene. Sphere penetration and surface distance are shown
per sample.

    python export_obstacle_viz.py --ckpt checkpoints/obs3d_*.pt \
        --db3d ergodic_dataset_3d_lm.db --out viz/obstacle_viewer.html
"""
import argparse
import json

import numpy as np
import torch

from data_surfaces import load_surface_db
from eval_lm3d_compare import load_model
from flow_matching_cond_particles_crossattn import generate_particle_trajectories
from obstacle_cond import deterministic_obstacle, sphere_penetration
from obstacles import OBSTACLE_MARGIN, SphereObstacle, basis_torch
from orientation_energy import ParticleSurface

HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Obstacle-conditioned trajectories</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.0/plotly.min.js"></script>
<style>
 body{margin:0;background:#fff;color:#1A1A2E;font-family:system-ui,sans-serif}
 #bar{display:flex;flex-wrap:wrap;gap:16px;align-items:center;padding:10px 16px;border-bottom:1px solid #ccc}
 label{font-size:13px;color:#555}
 select,button{font-size:13px;padding:3px 6px}
 #plot{width:100%;height:calc(100vh - 150px);min-height:420px}
 #stats{padding:6px 16px;font-size:13px;color:#555;white-space:pre-wrap;font-family:ui-monospace,monospace}
</style></head><body>
<div id="bar">
 <label>Shape <select id="item"></select></label>
 <label>Sphere position <select id="var"></select></label>
 <span id="methods">
  <label><input type="checkbox" id="m_null" checked> <b style="color:#E65100">no obstacle</b></label>
  <label><input type="checkbox" id="m_cond" checked> <b style="color:#00C853">trained conditioning</b></label>
  <label><input type="checkbox" id="m_cstr"> <b style="color:#7B1FA2">inference constraint</b></label>
  <label><input type="checkbox" id="m_both"> <b style="color:#00838F">conditioning + constraint</b></label>
 </span>
 <span id="samples"><label>Samples</label> </span>
 <label><input type="checkbox" id="gt" checked> ground truth</label>
 <label><input type="checkbox" id="parts" checked> target distribution</label>
</div>
<div id="plot"></div>
<div id="stats"></div>
<script>
const DATA = __DATA__;
const GT_C = '#1565C0';
const METHODS = [
  {id: 'm_null', name: 'no obstacle', color: '#E65100', gen: (it, v) => it.gen_null, pen: (it, v) => v.pen_null, smin: (it, v) => it.smin_null},
  {id: 'm_cond', name: 'trained conditioning', color: '#00C853', gen: (it, v) => v.gen_cond, pen: (it, v) => v.pen_cond, smin: (it, v) => v.smin_cond},
  {id: 'm_cstr', name: 'inference constraint', color: '#7B1FA2', gen: (it, v) => v.gen_cstr, pen: (it, v) => v.pen_cstr, smin: (it, v) => v.smin_cstr},
  {id: 'm_both', name: 'conditioning + constraint', color: '#00838F', gen: (it, v) => v.gen_both, pen: (it, v) => v.pen_both, smin: (it, v) => v.smin_both}];
const itemSel = document.getElementById('item'), varSel = document.getElementById('var');
DATA.items.forEach((it, i) => itemSel.add(new Option(it.name, i)));
function buildSamples() {
  const box = document.getElementById('samples'); box.querySelectorAll('.s').forEach(e => e.remove());
  const n = DATA.items[0].gen_null.length;
  for (let i = 0; i < n; i++) {
    const l = document.createElement('label'); l.className = 's';
    l.innerHTML = '<input type="checkbox" class="sc" data-i="' + i + '" checked> #' + (i + 1);
    l.firstChild.onchange = draw; box.appendChild(l);
  }
  const all = document.createElement('button'); all.className = 's'; all.textContent = 'all';
  all.onclick = () => { document.querySelectorAll('.sc').forEach(c => c.checked = true); draw(); };
  const none = document.createElement('button'); none.className = 's'; none.textContent = 'none';
  none.onclick = () => { document.querySelectorAll('.sc').forEach(c => c.checked = false); draw(); };
  box.appendChild(all); box.appendChild(none);
}
function pick(cs) {
  const idx = [...document.querySelectorAll('.sc')].filter(c => c.checked).map(c => +c.dataset.i);
  return idx.map(i => cs[i]).filter(Boolean);
}
function fillVars() {
  const it = DATA.items[+itemSel.value]; varSel.innerHTML = '';
  it.variants.forEach((v, j) => varSel.add(new Option('frac ' + v.frac + ' (r=' + v.sphere[3] + ')', j)));
}
function sphereMesh(c, r, color) {
  const n = 24, x = [], y = [], z = [], I = [], J = [], K = [];
  for (let a = 0; a <= n; a++) for (let b = 0; b < n; b++) {
    const th = Math.PI * a / n, ph = 2 * Math.PI * b / n;
    x.push(c[0] + r * Math.sin(th) * Math.cos(ph)); y.push(c[1] + r * Math.sin(th) * Math.sin(ph)); z.push(c[2] + r * Math.cos(th));
  }
  for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
    const p = a * n + b, q = a * n + (b + 1) % n, p2 = (a + 1) * n + b, q2 = (a + 1) * n + (b + 1) % n;
    I.push(p, q); J.push(q, q2); K.push(p2, p2);
  }
  return {type: 'mesh3d', x, y, z, i: I, j: J, k: K, color, opacity: 0.45, name: 'obstacle', hoverinfo: 'name', showlegend: true};
}
function line(curves, color, name, dash) {
  return curves.map((c, i) => ({type: 'scatter3d', mode: 'lines', x: c.map(p => p[0]), y: c.map(p => p[1]), z: c.map(p => p[2]),
    line: {color, width: i === 0 ? 6 : 3, dash: dash || 'solid'}, opacity: i === 0 ? 0.95 : 0.35,
    name, legendgroup: name, showlegend: i === 0, hoverinfo: 'name'}));
}
const RED = '#D50000';
function insideSegments(curves, c, r, name) {
  // one trace; segments of consecutive points inside the sphere, separated by null gaps
  const x = [], y = [], z = [];
  curves.forEach(cv => {
    let open = false;
    cv.forEach(p => {
      const d = Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]);
      if (d < r) { x.push(p[0]); y.push(p[1]); z.push(p[2]); open = true; }
      else if (open) { x.push(null); y.push(null); z.push(null); open = false; }
    });
    if (open) { x.push(null); y.push(null); z.push(null); }
  });
  return {type: 'scatter3d', mode: 'lines', x, y, z, line: {color: RED, width: 10},
    name: 'inside obstacle', legendgroup: 'inside', showlegend: name, hoverinfo: 'name'};
}
const densCache = new Map();
function density(parts, key) {
  // neighbour count in a 3x3x3 block of 0.06-wide cells, normalised to [0, 1]
  if (densCache.has(key)) return densCache.get(key);
  const h = 0.06, cell = p => [Math.floor(p[0] / h), Math.floor(p[1] / h), Math.floor(p[2] / h)];
  const grid = new Map(), id = c => c[0] + ',' + c[1] + ',' + c[2];
  parts.forEach(p => { const k = id(cell(p)); grid.set(k, (grid.get(k) || 0) + 1); });
  const d = parts.map(p => { const c = cell(p); let n = 0;
    for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) for (let k = -1; k <= 1; k++)
      n += grid.get(id([c[0] + i, c[1] + j, c[2] + k])) || 0;
    return n; });
  const m = Math.max(...d); const out = d.map(v => v / m);
  densCache.set(key, out); return out;
}
const WHITE_INFERNO = [[0, '#FFFFFF'], [0.15, '#4B0C6B'], [0.4, '#BB3754'], [0.7, '#F98E09'], [1, '#FCFFA4']];
function draw() {
  const it = DATA.items[+itemSel.value], v = it.variants[+varSel.value];
  const tr = [];
  if (document.getElementById('parts').checked)
    tr.push({type: 'scatter3d', mode: 'markers', x: it.parts.map(p => p[0]), y: it.parts.map(p => p[1]), z: it.parts.map(p => p[2]),
      marker: {size: 3.2, color: density(it.parts, itemSel.value), colorscale: WHITE_INFERNO, cmin: 0, cmax: 1, opacity: 0.55,
               colorbar: {title: 'target density', thickness: 10, len: 0.4, x: 1.0}},
      name: 'target distribution', hoverinfo: 'skip'});
  const sc = v.sphere.slice(0, 3), sr = v.sphere[3];
  let first = true; const red = cs => { tr.push(insideSegments(cs, sc, sr, first)); first = false; };
  if (document.getElementById('gt').checked) { tr.push(...line([it.gt], GT_C, 'ground truth')); red([it.gt]); }
  METHODS.forEach(m => {
    if (!document.getElementById(m.id).checked || !m.gen(it, v)) return;
    const g = pick(m.gen(it, v)); tr.push(...line(g, m.color, m.name)); red(g);
  });
  tr.push({type: 'scatter3d', mode: 'markers', x: [it.start[0]], y: [it.start[1]], z: [it.start[2]],
    marker: {size: 6, color: '#00C853', symbol: 'circle-open', line: {width: 3}}, name: 'conditioned start'});
  tr.push(sphereMesh(v.sphere.slice(0, 3), v.sphere[3], '#777777'));
  Plotly.react('plot', tr, {
    paper_bgcolor: '#fff', margin: {l: 0, r: 0, t: 0, b: 0}, legend: {x: 0, y: 1},
    scene: {aspectmode: 'cube', xaxis: {range: [0, 1], title: 'x'}, yaxis: {range: [0, 1], title: 'y'}, zaxis: {range: [0, 1], title: 'z'}}
  }, {responsive: true});
  const f = a => a.map(x => x.toFixed(3)).join('  ');
  let txt = 'sphere centre (' + sc.map(x => x.toFixed(2)).join(', ') + ')  radius ' + sr + '   (values per sample #1..#n)\n';
  METHODS.forEach(m => { if (!m.gen(it, v)) return;
    txt += (m.name + ':').padEnd(28) + 'penetration ' + f(m.pen(it, v)) + '  | min. surface dist. ' + f(m.smin(it, v)) + '\n'; });
  txt += 'ground truth min. surface dist. ' + it.smin_gt.toFixed(3);
  document.getElementById('stats').textContent = txt;
}
itemSel.onchange = () => { fillVars(); draw(); };
['var', 'm_null', 'm_cond', 'm_cstr', 'm_both', 'gt', 'parts'].forEach(id => document.getElementById(id).onchange = draw);
buildSamples(); fillVars(); draw();
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=None)
    p.add_argument('--db3d', default='ergodic_dataset_3d_lm.db')
    p.add_argument('--n_items', type=int, default=10)
    p.add_argument('--fracs', type=float, nargs='+', default=[0.3, 0.55, 0.8])
    p.add_argument('--radius', type=float, default=0.08)
    p.add_argument('--n_gen', type=int, default=4)
    p.add_argument('--constraint_weight', type=float, default=500.0,
                   help='strength of the inference-time obstacle repulsion (planner3d.OBSTACLE_WEIGHT)')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--max_particles', type=int, default=3000)
    p.add_argument('--rebuild_from', default=None,
                   help='re-render an existing viewer HTML with the current template (no model run)')
    p.add_argument('--out', default='viz/obstacle_viewer.html')
    a = p.parse_args()

    if a.rebuild_from:
        import re
        old = open(a.rebuild_from).read()
        data = re.search(r'const DATA = (.*?);\nconst GT_C', old, re.S).group(1)
        with open(a.out, 'w') as f:
            f.write(HTML.replace('__DATA__', data))
        print(f"rebuilt {a.out}")
        return

    if not a.ckpt:
        raise SystemExit('--ckpt is required unless --rebuild_from is given')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, info = load_model(a.ckpt, dev)
    if not getattr(model, 'obstacle_cond', False):
        raise SystemExit('checkpoint has no obstacle conditioning')
    print(f"model: {info}  device={dev}")
    entries = load_surface_db(a.db3d, nxi=25, splits=('val_form', 'val_flaeche', 'val_beides'),
                              target_cps=True, deg=5)
    step = max(len(entries) // a.n_items, 1)
    sel = entries[::step][:a.n_items]
    B = basis_torch(25, 256, 5, dev)
    rng = np.random.default_rng(0)

    def curves(gen):
        c = torch.einsum('pi,bid->bpd', B, gen[..., :3].float())
        return c

    out = dict(items=[])
    for n, e in enumerate(sel):
        x1 = torch.tensor(e['x1']).to(dev).float()
        parts = torch.tensor(e['parts']).to(dev).float()
        start = torch.tensor(e['start']).to(dev).float()
        surf = ParticleSurface(parts[None].expand(a.n_gen, -1, -1).contiguous(), 0.5)
        gt_curve = torch.einsum('pi,id->pd', B, x1[..., :3])
        smin_gt = ParticleSurface(parts[None], 0.5).distance(gt_curve[None]).amin().item()
        kw = dict(num_samples=a.n_gen, nxi=25, nd=3, steps=a.steps, device=str(dev),
                  cfg_weight=2.0, start=start if getattr(model, 'start_cond', False) else None)
        def gen(**extra):
            # identical initial noise for every method: sample #i is the same
            # noise draw everywhere, so the methods are directly comparable
            g = torch.Generator(device=dev).manual_seed(1000 + n)
            return generate_particle_trajectories(model, parts, generator=g, **kw, **extra)[0]

        gen_null = gen(obs_cond=None)
        cn = curves(gen_null)
        smin_null = surf.distance(cn).amin(dim=1).tolist()
        item = dict(name=f"{n:02d} {e.get('name', e.get('label', e.get('split', '')))}"[:80],
                    parts=parts[rng.permutation(len(parts))[:a.max_particles]].cpu().numpy()[:, :3].round(4).tolist(),
                    gt=gt_curve.cpu().numpy().round(4).tolist(),
                    start=start.cpu().numpy()[:3].round(4).tolist(),
                    gen_null=cn.cpu().numpy().round(4).tolist(), smin_null=smin_null,
                    smin_gt=smin_gt, variants=[])
        for fr in a.fracs:
            obs = deterministic_obstacle(x1, B, radius=a.radius, frac=fr)
            cc = curves(gen(obs_cond=obs))
            # inference-time constraint (obstacles.SphereObstacle repulsion in the flow
            # + final polish; the network itself never sees the sphere)
            sph = SphereObstacle(center=obs[:3].tolist(), radius=float(obs[3]),
                                 margin=OBSTACLE_MARGIN)
            cs = curves(gen(obs_cond=None, obstacle=sph, obstacle_weight=a.constraint_weight))
            cb = curves(gen(obs_cond=obs, obstacle=sph, obstacle_weight=a.constraint_weight))
            pen = lambda c: sphere_penetration(c, obs[None].expand(a.n_gen, -1)).amax(dim=1).tolist()
            item['variants'].append(dict(
                frac=fr, sphere=[round(v, 4) for v in obs.tolist()],
                gen_cond=cc.cpu().numpy().round(4).tolist(),
                pen_cond=sphere_penetration(cc, obs[None].expand(a.n_gen, -1)).amax(dim=1).tolist(),
                pen_null=sphere_penetration(cn, obs[None].expand(a.n_gen, -1)).amax(dim=1).tolist(),
                smin_cond=surf.distance(cc).amin(dim=1).tolist(),
                gen_cstr=cs.cpu().numpy().round(4).tolist(), pen_cstr=pen(cs),
                smin_cstr=surf.distance(cs).amin(dim=1).tolist(),
                gen_both=cb.cpu().numpy().round(4).tolist(), pen_both=pen(cb),
                smin_both=surf.distance(cb).amin(dim=1).tolist()))
        out['items'].append(item)
        print(f"  [{n + 1}/{len(sel)}] {item['name']}", flush=True)

    with open(a.out, 'w') as f:
        f.write(HTML.replace('__DATA__', json.dumps(out, separators=(',', ':'))))
    print(f"wrote {a.out}")


if __name__ == '__main__':
    main()
