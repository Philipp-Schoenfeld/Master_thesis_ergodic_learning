#!/usr/bin/env python3
"""Interactive 3D HTML viewer from the trajectories.npz of eval_lm3d_compare.py --dump_npz.

One self-contained file (plotly from cdnjs): pick a holdout entry, toggle the DB
teacher and each learned model, see particles (size ~ mu), the sampled
trajectories and the sensor axis along the path.

    python build_dbcmp_html.py --npz results/dbcmp_viz_old/trajectories.npz --title "old DB holdout"
"""
import argparse
import json
import os

import numpy as np
import torch

from obstacles import bspline_basis_matrix
from orientation import rot6d_to_matrix

COLORS = {'teacher': '#1565C0', 0: '#00C853', 1: '#F57C00', 2: '#8E24AA', 3: '#00838F'}


def curve(cps, rot6, B, n_ax=14):
    pos = np.einsum('pi,...id->...pd', B, cps)
    r = torch.from_numpy(np.einsum('pi,...ic->...pc', B, rot6))
    ax = rot6d_to_matrix(r)[..., 2].numpy()
    idx = np.linspace(0, B.shape[0] - 1, n_ax).astype(int)
    return pos, pos[..., idx, :], ax[..., idx, :]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', required=True)
    p.add_argument('--title', default='Holdout viewer')
    p.add_argument('--out', default=None)
    p.add_argument('--max_entries', type=int, default=300)
    a = p.parse_args()
    z = np.load(a.npz)
    labels = [str(x) for x in z['labels']]
    B = bspline_basis_matrix(25, 128, 5)
    r = lambda x: np.round(x, 3).tolist()
    entries = []
    for i in range(min(len(z['names']), a.max_entries)):
        parts = z['parts'][i]
        keep = parts[:, 3] > 0.05 * parts[:, 3].max()
        e = dict(name=str(z['names'][i]), split=str(z['splits'][i]), group=str(z['groups'][i]),
                 surface=str(z['surfaces'][i]),
                 parts=[r(parts[keep, 0]), r(parts[keep, 1]), r(parts[keep, 2]),
                        r(parts[keep, 3] / parts[:, 3].max())])
        pos, _, _ = curve(z['db_x1'][i][:, :3], z['db_x1'][i][:, 3:], B)
        e['teacher'] = [r(pos[:, 0]), r(pos[:, 1]), r(pos[:, 2])]
        e['gen'] = []
        for j in range(len(labels)):
            pos, pts, ax = curve(z[f'cps_{j}'][i], z[f'rot6_{j}'][i], B)
            e['gen'].append([dict(xyz=[r(c[:, 0]), r(c[:, 1]), r(c[:, 2])],
                                  arrows=[r(q) for q in (pts_[:, 0], pts_[:, 1], pts_[:, 2],
                                                         ax_[:, 0], ax_[:, 1], ax_[:, 2])])
                             for c, pts_, ax_ in zip(pos, pts, ax)])
        entries.append(e)
    html = HTML.replace('__TITLE__', a.title).replace('__DATA__', json.dumps(entries)) \
               .replace('__LABELS__', json.dumps(labels)).replace('__COLORS__', json.dumps(
                   [COLORS[j] for j in range(len(labels))])).replace('__TCOL__', COLORS['teacher'])
    out = a.out or os.path.join(os.path.dirname(a.npz), 'viewer_3d.html')
    open(out, 'w').write(html)
    print(f"{len(entries)} entries, {len(labels)} models -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"></script>
<style>body{font-family:system-ui,sans-serif;background:#fff;color:#1A1A2E;margin:0 auto;max-width:1200px;padding:16px}
h1{font-size:20px}select{font-size:14px;padding:3px;margin:4px 12px 4px 0}label{margin-right:14px;font-size:14px}
#p{width:100%;height:680px}.sw{display:inline-block;width:12px;height:12px;margin-right:4px;vertical-align:-1px}</style></head><body>
<h1>__TITLE__</h1>
<div>Split <select id="s"></select>Group <select id="g"></select>Entry <select id="e"></select>
<button id="prev">&lt;</button><button id="next">&gt;</button></div>
<div id="tg"></div><div id="p"></div>
<p style="color:#555;font-size:13px">Grey dots: target particles (size ~ mu). Blue: DB teacher. Other colours: first generated sample bold, the others faint. Short sticks: sensor axis along the path.</p>
<script>
const E=__DATA__,LAB=__LABELS__,COL=__COLORS__,TC='__TCOL__';
const sel=(id)=>document.getElementById(id),uniq=a=>['all',...new Set(a)];
sel('s').innerHTML=uniq(E.map(e=>e.split)).map(x=>`<option>${x}</option>`).join('');
sel('g').innerHTML=uniq(E.map(e=>e.group)).sort().map(x=>`<option>${x}</option>`).join('');
const names=[['DB teacher',TC],...LAB.map((l,i)=>[l,COL[i]])];
sel('tg').innerHTML=names.map((n,i)=>`<label><input type="checkbox" id="c${i}" checked><span class="sw" style="background:${n[1]}"></span>${n[0]}</label>`).join('');
let cur=[];
function fillEntries(){const s=sel('s').value,g=sel('g').value;cur=E.map((e,i)=>i).filter(i=>(s=='all'||E[i].split==s)&&(g=='all'||E[i].group==g));
 sel('e').innerHTML=cur.map(i=>`<option value="${i}">${E[i].name} / ${E[i].surface}</option>`).join('');draw();}
function draw(){const e=E[+sel('e').value];if(!e)return;const T=[];
 T.push({type:'scatter3d',mode:'markers',x:e.parts[0],y:e.parts[1],z:e.parts[2],hoverinfo:'skip',
  marker:{size:e.parts[3].map(m=>1.5+3*m),color:'#444',opacity:.3}});
 if(sel('c0').checked)T.push({type:'scatter3d',mode:'lines',x:e.teacher[0],y:e.teacher[1],z:e.teacher[2],line:{color:TC,width:6},name:'DB teacher'});
 LAB.forEach((l,j)=>{if(!sel('c'+(j+1)).checked)return;
  e.gen[j].forEach((g,k)=>{T.push({type:'scatter3d',mode:'lines',x:g.xyz[0],y:g.xyz[1],z:g.xyz[2],
   line:{color:COL[j],width:k?2:6},opacity:k?.35:1,name:l,showlegend:k==0,legendgroup:l});});
  const g=e.gen[j][0].arrows,x=[],y=[],z=[];
  for(let q=0;q<g[0].length;q++){x.push(g[0][q],g[0][q]+.06*g[3][q],null);y.push(g[1][q],g[1][q]+.06*g[4][q],null);z.push(g[2][q],g[2][q]+.06*g[5][q],null);}
  T.push({type:'scatter3d',mode:'lines',x,y,z,line:{color:COL[j],width:2},hoverinfo:'skip',showlegend:false,legendgroup:l});});
 Plotly.react('p',T,{paper_bgcolor:'#fff',margin:{t:10,l:0,r:0,b:0},scene:{aspectmode:'data',camera:window.cam||undefined},
  legend:{x:0,y:1}});}
sel('p').on&&0;
['s','g'].forEach(i=>sel(i).onchange=fillEntries);sel('e').onchange=draw;
names.forEach((_,i)=>sel('c'+i).onchange=draw);
sel('prev').onclick=()=>{sel('e').selectedIndex=Math.max(0,sel('e').selectedIndex-1);draw();};
sel('next').onclick=()=>{sel('e').selectedIndex=Math.min(sel('e').options.length-1,sel('e').selectedIndex+1);draw();};
fillEntries();
</script></body></html>"""

if __name__ == '__main__':
    main()
