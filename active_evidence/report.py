"""Build bounded Chinese summaries and plots for a verified active-evidence run."""
from __future__ import annotations
import argparse, base64, hashlib, html, json, re, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'active_evidence'
DOCS = ROOT / 'docs' / 'active_evidence'
PAIRS = [('gain_final','final'), ('gain_final','gate'), ('gain_gate','final'), ('gain_gate','gate'), ('stationary','final'),
         ('sweep','final'), ('random','final'), ('ibe_reference','final')]
LABEL = {'gain_final':'Gain / final','gain_gate':'Joint gate','stationary':'Stationary','sweep':'Sweep','random':'Random','ibe_reference':'IBE reference'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def load_ready(out):
    results=json.loads((out/'results.json').read_text(encoding='utf8'))
    selection=json.loads((out/'selection.json').read_text(encoding='utf8'))
    verification=json.loads((out/'verification.json').read_text(encoding='utf8'))
    if results.get('complete') is not True: raise ValueError('results.json is incomplete')
    if verification.get('passed') is not True: raise ValueError('verification.json has not passed')
    if verification.get('results_sha256') != sha(out/'results.json') or verification.get('selection_sha256') != sha(out/'selection.json'):
        raise ValueError('Verified result/selection identity no longer matches')
    return results,selection,verification

def ci(values):
    a=np.asarray(values,dtype=float); n=len(a); mean=float(a.mean())
    if n<2:return {'mean':mean,'sd':None,'n':n,'ci95':None}
    sd=float(a.std(ddof=1)); critical={2:12.706204736,3:4.30265273,4:3.182446305,5:2.776445105}[n]; half=critical*sd/np.sqrt(n)
    return {'mean':mean,'sd':sd,'n':n,'ci95':[mean-half,mean+half]}

def index_rows(rows):
    return {(r['seed'],r['split'],r['scene'],r['controller'],r['budget_m'],r['readout']):r for r in rows}

def scene_rows(rows,split,controller,budget,readout):
    return [r for r in rows if r['scene']=='all' and r['split']==split and r['controller']==controller and r['budget_m']==budget and r['readout']==readout]

def seed_summary(rows,metric,deterministic=False):
    vals=[r[metric] for r in rows]
    if deterministic and vals: vals=vals[:1]
    return ci(vals)

def make_plots(rows,docs):
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.grid':True,'grid.alpha':.25,'figure.dpi':130})
    colors={'gain_final':'#276FBF','gain_gate':'#D1495B','stationary':'#555555','sweep':'#2A9D8F','random':'#E09F3E','ibe_reference':'#8064A2'}
    specs=PAIRS
    for split in ('simple','hard'):
        fig,axs=plt.subplots(1,2,figsize=(12,4.6))
        for ctrl,readout in specs:
            points=[]
            for budget in (0,10,20,40,60):
                rr=scene_rows(rows,split,ctrl,budget,readout)
                if rr: points.append((budget,float(np.mean([r['accuracy'] for r in rr])),float(np.mean([r['distance_m'] for r in rr]))))
            if points:
                x,a,d=map(np.asarray,zip(*points))
                line='--' if ctrl.startswith('gain') and readout=='final' else '-'
                label=('Gain controller' if ctrl=='gain_final' else 'Joint controller' if ctrl=='gain_gate' else LABEL[ctrl])+' + '+readout
                axs[0].plot(x,a*100,marker='o',linestyle=line,label=label,color=colors[ctrl])
                axs[1].plot(d,a*100,marker='o',linestyle=line,label=label,color=colors[ctrl])
        axs[0].set(xlabel='Hard distance cap (m)',ylabel='Accuracy (%)',title=f'{split.title()}: accuracy vs cap')
        axs[1].set(xlabel='Mean realized distance (m)',ylabel='Accuracy (%)',title=f'{split.title()}: accuracy vs actual distance')
        axs[0].legend(fontsize=8); axs[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(docs/f'{split}_accuracy.png',bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8))
    for split,style in (('simple','-'),('hard','--')):
        for ctrl,color in [('gain_final','#276FBF'),('gain_gate','#D1495B')]:
            budgets=[]; rescue=[]; harm=[]
            for budget in (0,10,20,40,60):
                rr=scene_rows(rows,split,ctrl,budget,'gate')
                if rr:
                    budgets.append(budget);rescue.append(np.mean([r['rescue_rate'] for r in rr])*100);harm.append(np.mean([r['harm_rate'] for r in rr])*100)
            if budgets:
                ax.plot(budgets,rescue,'o'+style,color=color,label=f'{split} {LABEL[ctrl]} rescue')
                ax.plot(budgets,harm,'x'+style,color=color,alpha=.8,label=f'{split} {LABEL[ctrl]} harm')
    ax.set(xlabel='Hard distance cap (m)',ylabel='Episode share (%)',title='Gate readout: rescue and harm vs final-view prediction')
    ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(docs/'gate_rescue_harm.png',bbox_inches='tight');plt.close(fig)

def portable(value):
    if isinstance(value,dict): return {portable(k):portable(v) for k,v in value.items()}
    if isinstance(value,list): return [portable(v) for v in value]
    if isinstance(value,str) and (Path(value).is_absolute() or len(value)>2 and value[1:3]==':\\' or value.startswith('\\\\')):
        p=Path(value)
        try: return p.resolve().relative_to(ROOT.resolve()).as_posix()
        except (ValueError,OSError): return p.name
    return value

def build(out=OUT,docs=DOCS):
    results,selection,verification=load_ready(out); docs.mkdir(parents=True,exist_ok=True)
    rows=[r for r in results['rows'] if r['scene']=='all']
    indexed=index_rows(rows)
    make_plots(rows,docs)
    summaries={}
    for split in ('simple','hard'):
        seeds=sorted({r['seed'] for r in rows if r['split']==split})
        summaries[split]={}
        for ctrl,readout in PAIRS:
            summaries[split][f'{ctrl}/{readout}']={}
            for b in (0,10,20,40,60):
                rr=scene_rows(rows,split,ctrl,b,readout)
                if rr: summaries[split][f'{ctrl}/{readout}'][str(b)]={m:seed_summary(rr,m,ctrl in ('stationary','sweep') and readout in ('final','max_confidence','mean')) for m in ('accuracy','distance_m','moves','unknown_rate','rescue_rate','harm_rate')}
        # Paired gate-minus-final effects on identical paths, averaged over scenes per seed.
        effects={}
        for ctrl in ('gain_final','gain_gate'):
            effects[ctrl]={}
            for b in (0,10,20,40,60):
                diffs=[]
                for seed in seeds:
                    f=indexed.get((seed,split,'all',ctrl,b,'final'))
                    g=indexed.get((seed,split,'all',ctrl,b,'gate'))
                    if f and g:diffs.append(g['accuracy']-f['accuracy'])
                if diffs: effects[ctrl][str(b)]=ci(diffs)
        summaries[split]['paired_gate_minus_final']=effects
    cal_raw=selection.get('calibration',{})
    calibration={seed:{name:{'brier':value.get('brier'),'scope':value.get('scope')} for name,value in entries.items()} for seed,entries in cal_raw.items()}
    calibration_ranges={}
    for name in ('gain_final','gain_gate'):
        vals=[entries[name]['brier'] for entries in cal_raw.values() if name in entries and entries[name].get('brier') is not None]
        if vals: calibration_ranges[name]={'min':min(vals),'max':max(vals),'mean':float(np.mean(vals)),'n':len(vals)}
    metrics={'schema':'active-evidence-report-v1','selection':portable(selection.get('selected',{})),
      'config':portable(selection.get('config',{})), 'selection_candidates':portable(selection.get('candidates',[])),
      'calibration':portable({'by_seed':calibration,'brier_ranges':calibration_ranges}), 'verification':portable(verification),
      'complete':results.get('complete'),'rows':summaries,'artifacts':[]}
    for p in sorted(docs.glob('*.png')):metrics['artifacts'].append({'path':p.name,'sha256':sha(p)})
    (docs/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    md=['# 主动证据预测评估报告','', '评估仅衡量当前观测下的一步收益预测与证据门控。预算上限不代表各策略实际行驶距离相同；图中同时报告上限与实耗。结果不构成 SOTA 或普遍稳定优势的主张。','', '## 结果图','', '| 测试划分 | 准确率与预算/实耗 | 门控救回与误伤 |','|---|---|---|']
    for s in ('simple','hard'): md.append(f'| {s} | ![{s} accuracy]({s}_accuracy.png) | ![gate rescue/harm](gate_rescue_harm.png) |')
    md += ['', '## 验证与模型选择','', f"独立核验通过：`{verification.get('passed')}`。五个新模块训练种子为 42–46，并共享冻结的上游分类器、ResNet50 特征与 SAM 分解。simple/hard 划分已有过往实验探索，因此本次是复评，不视为未触碰的确认集。成本系数在 validation 选择，最终测试只报告冻结选择。准确率、距离和未知率区间是五个种子上的普通 t 区间（df=4），不是配对区间；n=1 的确定性结果没有区间。stationary 与 sweep 的 final/max_confidence/mean 读出确定性去重；它们的 gate 读出仍随 gate 种子变化。IBE reference 是五个不同种子的历史策略检查点，保留五种子；random 同样保留五种子。", '', '## 主要比较','', '| 划分 | 控制器/读出 | 预算(m) | 准确率 均值±SD，95%普通t区间 | 实际距离(m) 均值±SD | 未知率 |', '|---|---|---:|---|---|---|']
    ix=md.index('## 主要比较')
    md[ix:ix]=[f"内部 validation 初始状态 gain Brier（五种子范围、均值）：gain_final {calibration_ranges['gain_final']['min']:.4f}–{calibration_ranges['gain_final']['max']:.4f}，均值 {calibration_ranges['gain_final']['mean']:.4f}。", f"gain_gate {calibration_ranges['gain_gate']['min']:.4f}–{calibration_ranges['gain_gate']['max']:.4f}，均值 {calibration_ranges['gain_gate']['mean']:.4f}；这是一步收益标签概率校准，不等于最终分类准确率。", '']
    for s in ('simple','hard'):
        for c,r in PAIRS:
            key=f'{c}/{r}'
            for b,v in summaries[s].get(key,{}).items():
                a=v['accuracy'];d=v['distance_m'];u=v['unknown_rate'];ci_txt='—' if a['ci95'] is None else f"[{a['ci95'][0]*100:.2f}, {a['ci95'][1]*100:.2f}]";ds='—' if d['sd'] is None else f"{d['sd']:.2f}";asd='—' if a['sd'] is None else f"{a['sd']*100:.2f}";md.append(f"| {s} | {key} | {b} | {a['mean']*100:.2f}% ± {asd} {ci_txt} | {d['mean']:.2f} ± {ds} | {u['mean']*100:.2f}% |")
    md += ['', '## Hard split 重点预算','', '| 控制器/读出 | cap (m) | 准确率均值 (%) | 实际距离均值 (m) |', '|---|---:|---:|---:|']
    for ctrl,readout in PAIRS:
        for b in (20,40,60):
            v=summaries['hard'].get(f'{ctrl}/{readout}',{}).get(str(b))
            if v: md.append(f"| {ctrl}/{readout} | {b} | {v['accuracy']['mean']*100:.2f} | {v['distance_m']['mean']:.2f} |")
    md += ['', '## 同路径 gate 与 final 配对差值','', '| 划分 | 控制器 | cap (m) | gate − final 准确率百分点，配对95% t区间 |', '|---|---|---:|---|']
    for sp in ('simple','hard'):
        for ctrl in ('gain_final','gain_gate'):
            for b,v in summaries[sp]['paired_gate_minus_final'][ctrl].items():
                interval='—' if v['ci95'] is None else f"[{v['ci95'][0]*100:.2f}, {v['ci95'][1]*100:.2f}]"
                md.append(f"| {sp} | {ctrl} | {b} | {v['mean']*100:.2f}pp; {interval} |")
    observations=[]
    for b in (20,40,60):
        candidates=[(k,v[str(b)]['accuracy']['mean'],v[str(b)]['distance_m']['mean']) for k,v in summaries['hard'].items() if k in {f'{c}/{r}' for c,r in PAIRS} and str(b) in v]
        if candidates:
            top=max(candidates,key=lambda x:x[1]); observations.append(f"cap {b} m 时，列出的读出中最高准确率为 {top[0]}（{top[1]*100:.2f}%，平均实耗 {top[2]:.2f} m）")
    paired=[(sp,c,b,v) for sp in ('simple','hard') for c in ('gain_final','gain_gate') for b,v in summaries[sp]['paired_gate_minus_final'][c].items()]
    if paired:
        low=min(paired,key=lambda x:x[3]['mean']); high=max(paired,key=lambda x:x[3]['mean'])
        observations.append(f"gate−final 配对差值范围为 {low[3]['mean']*100:.2f} 到 {high[3]['mean']*100:.2f} 个百分点（最小值 {low[0]}/{low[1]} cap {low[2]} m，最大值 {high[0]}/{high[1]} cap {high[2]} m）")
    md += ['', '## 按实测结果生成的观察','', *[f'- {x}。' for x in observations], '', '普通准确率区间基于种子均值；只有 gate−final 的区间是同路径配对区间。gain_final 与 gain_gate 的路径可能不同，二者横向差异不代表同路径因果效应。一步预测不规划必须先经过低收益视点的多步路线。预算上限不等于相同实耗。','']
    report='\n'.join(md)
    (docs/'report.md').write_text(report,encoding='utf8')
    # Local interactive-free HTML bundle; same bounded report, charts embedded.
    blocks=[];in_table=False
    for line in report.splitlines():
        if '![' in line: continue
        if line.startswith('|'):
            if all(c in '|-: ' for c in line):continue
            if not in_table:blocks.append('<div class="scroll"><table>');in_table=True
            blocks.append('<tr>'+''.join('<td>'+html.escape(c.strip())+'</td>' for c in line.strip('|').split('|'))+'</tr>')
            continue
        if in_table:blocks.append('</table></div>');in_table=False
        if not line:continue
        match=re.match(r'^(#{1,6}) (.*)$',line)
        blocks.append(f'<h{len(match[1])}>{html.escape(match[2])}</h{len(match[1])}>' if match else '<p>'+html.escape(line)+'</p>')
    if in_table:blocks.append('</table></div>')
    plots=''.join('<img alt="'+p.stem+'" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode('ascii')+'">' for p in sorted(docs.glob('*.png')))
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>主动证据预测评估</title><style>body{font:16px/1.7 system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#233040}img{max-width:100%}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td{border-bottom:1px solid #dbe2e7;padding:.45rem;white-space:nowrap}tr:first-child{font-weight:bold;background:#e7f2f3}h2{margin-top:2rem}</style><body>'+plots+''.join(blocks)+'</body></html>'
    out.mkdir(parents=True,exist_ok=True)
    for p in docs.glob('*.png'): shutil.copy2(p,out/p.name)
    (out/'report.md').write_text(report,encoding='utf8');(out/'report.html').write_text(page,encoding='utf8')
    print(f'Reports written; publishable docs in {docs}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);p.add_argument('--docs',type=Path,default=DOCS);a=p.parse_args();build(a.out,a.docs)
