#!/usr/bin/env python3
"""Comparative analysis of per_entry.csv written by eval_lm3d_compare.py.

Writes PNG plots, a markdown table per surface group and a self-contained
interactive HTML report (plotly from cdnjs, data embedded) into --out_dir.

    python analyze_dbcmp.py --csv results/dbcmp_compare/per_entry.csv
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ['sinkhorn', 'fourier', 'standoff_pca', 'pointing_pca', 'path_len', 'cp_rmse']
LABEL = {'sinkhorn': 'Sinkhorn (footprint)', 'fourier': 'Fourier (footprint)',
         'standoff_pca': '|standoff - 0.12| (PCA plane)', 'pointing_pca': 'pointing angle [deg] (PCA)',
         'path_len': 'control polygon length', 'cp_rmse': 'control point RMSE to DB target'}
LOGM = {'sinkhorn', 'fourier'}
COLORS = ['#1565C0', '#00C853', '#F57C00', '#8E24AA', '#D81B60', '#555555']
REF = 'DB teacher'
NEW, OLD = 'DiT-S, refined DB', 'DiT-S, old DB'


def style(ax):
    ax.set_facecolor('white')
    ax.grid(alpha=0.2)
    for s in ax.spines.values():
        s.set_color('#ccc')
    ax.tick_params(colors='#555')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='results/dbcmp_compare/per_entry.csv')
    p.add_argument('--out_dir', default=None)
    a = p.parse_args()
    out = a.out_dir or os.path.dirname(a.csv)
    df = pd.read_csv(a.csv)
    models = list(dict.fromkeys(df['model']))
    col = dict(zip(models, COLORS))

    # 1. distributions per metric
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor='white')
    for ax, m in zip(axes.ravel(), METRICS):
        data = [df[df.model == k][m].values for k in models]
        if m in LOGM:
            data = [np.log10(np.clip(d, 1e-7, None)) for d in data]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, k in zip(bp['boxes'], models):
            patch.set_facecolor(col[k]); patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(models) + 1))
        ax.set_xticklabels([k.replace(' 750 ep', '\n750 ep').replace(', ', '\n') for k in models],
                           fontsize=8)
        ax.set_title(LABEL[m] + (' [log10]' if m in LOGM else ''), color='#1A1A2E', fontsize=10)
        style(ax)
    fig.suptitle('Per-entry metric distributions (%d holdout entries per model)'
                 % (len(df) // len(models)), color='#1A1A2E')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'cmp_distributions.png'), dpi=130)
    plt.close(fig)

    # 2. paired comparison: refined vs old DiT-S, win rate and ratio per metric
    piv = {m: df.pivot_table(index='name', columns='model', values=m, aggfunc='mean') for m in METRICS}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor='white')
    wins = {}
    for m in METRICS:
        wins[m] = float((piv[m][NEW] < piv[m][OLD]).mean() * 100)
    ax = axes[0]
    ax.barh(range(len(METRICS)), [wins[m] for m in METRICS], color='#00C853', alpha=0.8)
    ax.axvline(50, color='#555', lw=1)
    ax.set_yticks(range(len(METRICS))); ax.set_yticklabels([LABEL[m] for m in METRICS], fontsize=8)
    ax.set_xlabel('% of entries where refined-DB model is lower (better)'); ax.set_xlim(0, 100)
    ax.set_title('Paired win rate: refined vs old DB', color='#1A1A2E'); style(ax)
    for ax, m in zip(axes[1:], ['sinkhorn', 'pointing_pca']):
        x, y = piv[m][OLD].values, piv[m][NEW].values
        ax.scatter(x, y, s=6, alpha=0.4, color='#00C853')
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color='#555', lw=1)
        if m in LOGM:
            ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(f'{OLD}'); ax.set_ylabel(f'{NEW}')
        ax.set_title(f'{LABEL[m]} per entry (below diagonal = refined better)', fontsize=9, color='#1A1A2E')
        style(ax)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'cmp_paired.png'), dpi=130); plt.close(fig)

    # 3. by surface group
    groups = sorted(df.gruppe.unique())
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), facecolor='white')
    for ax, m in zip(axes, ['sinkhorn', 'pointing_pca', 'path_len']):
        w = 0.8 / len(models)
        for j, k in enumerate(models):
            v = [df[(df.model == k) & (df.gruppe == g)][m].median() for g in groups]
            ax.bar(np.arange(len(groups)) + j * w, v, w, label=k, color=col[k], alpha=0.85)
        ax.set_xticks(np.arange(len(groups)) + 0.4 - w / 2)
        ax.set_xticklabels(groups, rotation=30, ha='right', fontsize=8)
        if m in LOGM:
            ax.set_yscale('log')
        ax.set_title('Median ' + LABEL[m] + ' by surface group', fontsize=10, color='#1A1A2E'); style(ax)
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'cmp_by_group.png'), dpi=130); plt.close(fig)

    # 4. trade-off: coverage vs pointing, one marker per model and group
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor='white')
    for k in models:
        d = df[df.model == k].groupby('gruppe')[['sinkhorn', 'pointing_pca']].median()
        ax.scatter(d.pointing_pca, d.sinkhorn, s=45, color=col[k], label=k, alpha=0.85)
    ax.set_yscale('log'); ax.set_xlabel('median pointing angle [deg] (PCA)')
    ax.set_ylabel('median Sinkhorn (footprint)'); ax.legend(fontsize=8)
    ax.set_title('Coverage vs pointing trade-off (one point per surface group)', color='#1A1A2E')
    style(ax); fig.tight_layout(); fig.savefig(os.path.join(out, 'cmp_tradeoff.png'), dpi=130)
    plt.close(fig)

    # markdown: median per group and model
    lines = ['| group | model | ' + ' | '.join(METRICS) + ' |', '|---|---|' + '---|' * len(METRICS)]
    for g in groups:
        for k in models:
            d = df[(df.model == k) & (df.gruppe == g)]
            lines.append(f'| {g} | {k} | ' + ' | '.join(f'{d[m].median():.3g}' for m in METRICS) + ' |')
    lines += ['', 'Paired win rate (refined < old, %): ' + ', '.join(f'{m} {wins[m]:.0f}' for m in METRICS)]
    open(os.path.join(out, 'cmp_by_group.md'), 'w').write('\n'.join(lines) + '\n')

    # HTML
    recs = df[['name', 'split', 'surface', 'gruppe', 'model'] + METRICS].round(6).to_dict('list')
    html = HTML.replace('__DATA__', json.dumps(recs)).replace('__COLORS__', json.dumps(col)) \
               .replace('__METRICS__', json.dumps(METRICS)).replace('__LABEL__', json.dumps(LABEL)) \
               .replace('__LOG__', json.dumps(sorted(LOGM))).replace('__MODELS__', json.dumps(models)) \
               .replace('__NEW__', json.dumps(NEW)).replace('__OLD__', json.dumps(OLD))
    open(os.path.join(out, 'report.html'), 'w').write(html)
    print('wrote plots, cmp_by_group.md and report.html to', out)


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DB comparison</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"></script>
<style>body{font-family:system-ui,sans-serif;background:#fff;color:#1A1A2E;margin:0 auto;max-width:1100px;padding:16px}
h1{font-size:20px}h2{font-size:16px;margin-top:28px}select{font-size:14px;padding:3px;margin-right:12px}
.row{margin:8px 0}.plot{width:100%;height:480px}</style></head><body>
<h1>Learned models vs DB teacher - holdout comparison</h1>
<div class="row">Metric <select id="m"></select>Split <select id="s"><option>all</option></select>Group <select id="g"><option>all</option></select></div>
<h2>Distribution per model</h2><div id="box" class="plot"></div>
<h2>Per entry: old-DB model (x) vs refined-DB model (y) - below the diagonal the refined DB is better</h2><div id="sc" class="plot"></div>
<h2>Median by surface group</h2><div id="bar" class="plot"></div>
<script>
const D=__DATA__,C=__COLORS__,MET=__METRICS__,LAB=__LABEL__,LOG=new Set(__LOG__),MOD=__MODELS__,NEW=__NEW__,OLD=__OLD__;
const n=D.name.length,uniq=a=>[...new Set(a)];
const fill=(id,v)=>{const e=document.getElementById(id);v.forEach(x=>e.add(new Option(x)));};
fill('m',MET.map(m=>m));document.getElementById('m').innerHTML=MET.map(m=>`<option value="${m}">${LAB[m]}</option>`).join('');
fill('s',uniq(D.split));fill('g',uniq(D.gruppe).sort());
const L={paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{color:'#1A1A2E'},margin:{t:20}};
function sel(){const s=document.getElementById('s').value,g=document.getElementById('g').value;
 const idx=[];for(let i=0;i<n;i++)if((s=='all'||D.split[i]==s)&&(g=='all'||D.gruppe[i]==g))idx.push(i);return idx;}
function draw(){const m=document.getElementById('m').value,idx=sel(),log=LOG.has(m)?'log':'linear';
 Plotly.react('box',MOD.map(k=>({type:'box',name:k,y:idx.filter(i=>D.model[i]==k).map(i=>D[m][i]),marker:{color:C[k]},boxpoints:false})),
  {...L,yaxis:{type:log,title:LAB[m],gridcolor:'#eee'}});
 const byName={};idx.forEach(i=>{(byName[D.name[i]]=byName[D.name[i]]||{})[D.model[i]]=D[m][i];});
 const nm=Object.keys(byName).filter(k=>byName[k][NEW]!==undefined&&byName[k][OLD]!==undefined);
 const x=nm.map(k=>byName[k][OLD]),y=nm.map(k=>byName[k][NEW]),lo=Math.min(...x,...y),hi=Math.max(...x,...y);
 Plotly.react('sc',[{type:'scatter',mode:'markers',x,y,text:nm,marker:{color:'#00C853',size:5,opacity:.5}},
  {type:'scatter',mode:'lines',x:[lo,hi],y:[lo,hi],line:{color:'#555',width:1},hoverinfo:'skip'}],
  {...L,showlegend:false,xaxis:{type:log,title:OLD,gridcolor:'#eee'},yaxis:{type:log,title:NEW,gridcolor:'#eee'}});
 const gs=uniq(D.gruppe).sort(),med=a=>{a=a.sort((p,q)=>p-q);return a.length?a[Math.floor(a.length/2)]:null;};
 const s=document.getElementById('s').value;
 Plotly.react('bar',MOD.map(k=>({type:'bar',name:k,x:gs,marker:{color:C[k]},
  y:gs.map(g=>med(D.model.map((_,i)=>i).filter(i=>D.model[i]==k&&D.gruppe[i]==g&&(s=='all'||D.split[i]==s)).map(i=>D[m][i])))})),
  {...L,barmode:'group',yaxis:{type:log,title:LAB[m],gridcolor:'#eee'}});}
['m','s','g'].forEach(i=>document.getElementById(i).onchange=draw);draw();
</script></body></html>"""

if __name__ == '__main__':
    main()
