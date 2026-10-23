"""Publish bounded five-seed ablation summaries after independent verification."""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .report import ci, portable, sha

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44, 45, 46]
BUDGETS = [0, 10, 20, 40, 60]
LABELS = {'final': 'Final frame', 'max_confidence': 'Max confidence', 'mean': 'Mean',
          'rule_latest': 'Latest visible', 'rule_agree': 'Agreement rule',
          'gate2_full': 'Binary gate', 'gate3_full': 'Three-way gate',
          'gate2_noembed': 'Binary / no embeddings', 'gate3_noembed': 'Three-way / no embeddings'}


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def pct(value):
    return f'{value * 100:.2f}'


def effect_text(effect):
    lo, hi = effect['ci95']
    return f"{effect['mean'] * 100:+.3f} [{lo * 100:+.3f}, {hi * 100:+.3f}]"


def build(out, docs):
    results, selection = load(out / 'results.json'), load(out / 'selection.json')
    verification, replay = load(out / 'verification.json'), load(out / 'model_replay.json')
    if results.get('complete') is not True:
        raise ValueError('Incomplete ablation results')
    for check in (verification, replay):
        if check.get('passed') is not True or check.get('results_sha256') != sha(out / 'results.json'):
            raise ValueError('Missing/stale verification or neural replay')
        if check.get('selection_sha256') != sha(out / 'selection.json'):
            raise ValueError('Selection verification changed')
    docs.mkdir(parents=True, exist_ok=True)
    ix = {(r['seed'], r['split'], r['controller'], r['budget_m'], r['scene'], r['readout']): r for r in results['rows']}

    def values(split, controller, budget, readout, metric='accuracy', scene='all'):
        return np.array([ix[seed, split, controller, budget, scene, readout][metric] for seed in SEEDS])

    summaries = []
    fields = ['accuracy', 'distance_m', 'moves', 'unknown_rate', 'rescue_rate', 'harm_rate']
    for split in ('simple', 'hard'):
        for controller in results['controllers']:
            for budget in BUDGETS:
                for readout in results['readouts']:
                    summaries.append({'split': split, 'controller': controller, 'budget_m': budget, 'readout': readout,
                                      **{metric: ci(values(split, controller, budget, readout, metric)) for metric in fields}})

    # Every contrast is fixed in the protocol; include both benchmark splits and all caps.
    gate_pairs = [('binary_vs_latest', 'gate2_full', 'rule_latest'),
                  ('binary_vs_agreement', 'gate2_full', 'rule_agree'),
                  ('three_vs_binary', 'gate3_full', 'gate2_full'),
                  ('three_vs_latest', 'gate3_full', 'rule_latest'),
                  ('three_vs_agreement', 'gate3_full', 'rule_agree'),
                  ('three_vs_final', 'gate3_full', 'final'),
                  ('binary_embeddings', 'gate2_full', 'gate2_noembed'),
                  ('three_embeddings', 'gate3_full', 'gate3_noembed')]
    effects = []
    for split in ('simple', 'hard'):
        scenes = sorted({r['scene'] for r in results['rows'] if r['split'] == split and r['scene'] != 'all'})
        for controller in results['controllers']:
            for budget in BUDGETS:
                for name, first, second in gate_pairs:
                    differences = values(split, controller, budget, first) - values(split, controller, budget, second)
                    by_scene = [{'scene': scene, 'delta': float(np.mean(values(split, controller, budget, first, scene=scene)
                                                                 - values(split, controller, budget, second, scene=scene)))}
                                for scene in scenes]
                    effects.append({'split': split, 'controller': controller, 'budget_m': budget, 'contrast': name,
                                    'first': first, 'second': second, 'accuracy_delta': ci(differences),
                                    'per_seed_delta': differences.tolist(), 'by_scene': by_scene,
                                    'same_path': True})
    controller_effects = []
    for split in ('simple', 'hard'):
        for budget in BUDGETS:
            for other in ('gain_noembed', 'gain_noembed_fixed_cost'):
                for first, second in [('final', 'final'), ('gate3_full', 'gate3_full'), ('gate3_full', 'gate3_noembed')]:
                    controller_effects.append({'split': split, 'budget_m': budget, 'first_controller': 'gain_full',
                                              'second_controller': other, 'first_readout': first, 'second_readout': second,
                                              'accuracy_delta': ci(values(split, 'gain_full', budget, first) - values(split, other, budget, second)),
                                              'distance_delta': ci(values(split, 'gain_full', budget, first, 'distance_m') - values(split, other, budget, second, 'distance_m')),
                                              'same_path': False})

    summary_ix = {(r['split'], r['controller'], r['budget_m'], r['readout']): r for r in summaries}
    effect_ix = {(r['split'], r['controller'], r['budget_m'], r['contrast']): r for r in effects}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 140, 'axes.grid': True, 'grid.alpha': .25})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for split, ax in zip(('simple', 'hard'), axes):
        for readout in ('final', 'rule_latest', 'rule_agree', 'gate2_full', 'gate3_full', 'gate3_noembed'):
            ys = [summary_ix[split, 'gain_full', b, readout]['accuracy']['mean'] * 100 for b in BUDGETS]
            ax.plot(BUDGETS, ys, marker='o', label=LABELS[readout])
        ax.set(title=f'{split.title()}: identical gain-controller paths', xlabel='Distance cap (m)', ylabel='Accuracy (%)')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(docs / 'gate_ablations.png', bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = ['#276FBF', '#D1495B', '#2A9D8F']
    for split, ax in zip(('simple', 'hard'), axes):
        for index, (controller, label, color) in enumerate(zip(('gain_full', 'gain_noembed', 'gain_noembed_fixed_cost'),
                                           ('Full embeddings', 'No embeddings / selected cost', 'No embeddings / fixed cost'), colors)):
            points = [summary_ix[split, controller, b, 'final'] for b in BUDGETS]
            ax.plot([p['distance_m']['mean'] for p in points], [p['accuracy']['mean'] * 100 for p in points],
                    marker='o', color=color, label=label)
            for b, p in zip(BUDGETS, points):
                if b:
                    ax.annotate(str(b), (p['distance_m']['mean'], p['accuracy']['mean'] * 100),
                                fontsize=7, color=color, xytext=[(4, -12), (-12, 7), (4, 7)][index], textcoords='offset points')
        ax.set(title=f'{split.title()}: gain controller, final-frame readout', xlabel='Mean actual distance (m)', ylabel='Accuracy (%)')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(docs / 'controller_embeddings.png', bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    names = [('three_vs_binary', 'Three-way minus binary'), ('three_vs_latest', 'Three-way minus latest'),
             ('three_embeddings', 'Three-way full minus no embeddings')]
    for split, ax in zip(('simple', 'hard'), axes):
        for j, (name, label) in enumerate(names):
            es = [effect_ix[split, 'gain_full', b, name]['accuracy_delta'] for b in BUDGETS[1:]]
            ys = np.array([e['mean'] for e in es]) * 100
            errors = np.array([[e['mean'] - e['ci95'][0], e['ci95'][1] - e['mean']] for e in es]).T * 100
            ax.errorbar(np.array(BUDGETS[1:]) + (j - 1) * 1.5, ys, yerr=errors, fmt='o-', capsize=3, label=label)
        ax.axhline(0, color='black', lw=.8)
        ax.set(title=f'{split.title()}: paired effects on identical paths', xlabel='Distance cap (m)', ylabel='Accuracy difference (percentage points)')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(docs / 'paired_effects.png', bbox_inches='tight')
    plt.close(fig)

    published = {'schema': 'active-evidence-ablations-v1', 'config': results['config'], 'selection': selection['selected'],
                 'selection_candidates': selection['candidates'], 'verification': portable(verification),
                 'model_replay': portable(replay), 'summaries': summaries, 'paired_gate_effects': effects,
                 'controller_effects': controller_effects,
                 'artifacts': [{'path': p.name, 'sha256': sha(p)} for p in sorted(docs.glob('*.png'))]}
    (docs / 'metrics.json').write_text(json.dumps(published, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    md = ['# 门控与额外视觉特征消融', '',
          '沿用 90 DEVELOPMENT / 10 VALIDATION，最终在 100 TRAIN 场景重训；种子 42–46。'
          '五种子均逐权重复现原有完整模型，然后共享样本、初始化和批次顺序训练消融版本。', '',
          '无 embedding 版本同时删除 raw/SAM 的 128 维特征，保留分类概率、目标框和位置；不能称为完全无视觉。'
          '二路门控只学习保留/替换，三路增加半半融合。两者使用相同的混合历史训练分布。', '',
          f"验证集选择成本：完整控制器 {selection['selected']['gain_full']} /m；无 embedding 控制器 "
          f"{selection['selected']['gain_noembed']} /m。另保留无 embedding、固定完整控制器成本 "
          f"{selection['selected']['gain_noembed_fixed_cost']} /m 的对照。", '',
          '所有原有测试集均已经探索过。本报告不将其描述为未触碰确认集。共同预算上限不等于共同实际移动距离。', '',
          '## 结果图', '', '![同轨迹门控](gate_ablations.png)', '',
          '![配对差值](paired_effects.png)', '', '![控制器特征消融](controller_embeddings.png)', '',
          '## 主比较：完整收益控制器的相同轨迹', '',
          '表中是五种子平均准确率（%）；同一行内各读出完全共享路径和距离。', '',
          '| 划分 | cap m | 实耗 m | 最终帧 | 最高置信 | 均值 | 最新可见规则 | 一致性规则 | 二路 | 三路 | 二路无特征 | 三路无特征 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('simple', 'hard'):
        for b in BUDGETS:
            rs = [summary_ix[split, 'gain_full', b, r] for r in results['readouts']]
            md.append(f"| {split} | {b} | {rs[0]['distance_m']['mean']:.2f} | " + ' | '.join(pct(r['accuracy']['mean']) for r in rs) + ' |')
    md += ['', '最新可见规则：新观察可见且未重复则替换。'
           '一致性规则：新旧 top-1 一致则融合，否则取最大概率更高的一方，平局取新。'
           '所有规则和门控共用不可见/重复保留与首次可见替换。', '',
           '## 预先指定的同轨迹配对差值', '',
           '单位为准确率百分点；括号为五种子配对 95% t 区间。区间条件于固定场景和上游模型，未做多重比较校正。', '',
           '| 划分 | cap m | 二路−最新规则 | 二路−一致规则 | 三路−二路 | 三路−最新规则 | 三路−一致规则 | 二路完整−无特征 | 三路完整−无特征 |',
           '|---|---:|---|---|---|---|---|---|---|']
    effect_names = ['binary_vs_latest', 'binary_vs_agreement', 'three_vs_binary', 'three_vs_latest',
                    'three_vs_agreement', 'binary_embeddings', 'three_embeddings']
    for split in ('simple', 'hard'):
        for b in BUDGETS[1:]:
            md.append(f'| {split} | {b} | ' + ' | '.join(effect_text(effect_ix[split, 'gain_full', b, n]['accuracy_delta']) for n in effect_names) + ' |')
    md += ['', '## 控制器的额外特征：最终帧读出', '',
           '这组路径可以不同；显示准确率与实耗，不把准确率差值归因为同路径融合。图中点旁数字为距离上限（m）。', '',
           '| 划分 | cap m | 完整：准确率 / m | 无特征、验证选成本：准确率 / m | 无特征、固定成本：准确率 / m |',
           '|---|---:|---|---|---|']
    for split in ('simple', 'hard'):
        for b in BUDGETS[1:]:
            parts = []
            for c in ('gain_full', 'gain_noembed', 'gain_noembed_fixed_cost'):
                r = summary_ix[split, c, b, 'final']
                parts.append(f"{pct(r['accuracy']['mean'])}% / {r['distance_m']['mean']:.2f}")
            md.append(f'| {split} | {b} | ' + ' | '.join(parts) + ' |')
    md += ['', '## 40 m hard：门控救回与误伤', '',
           '| 读出 | 准确率 % | 救回占全部样本 % | 误伤占全部样本 % |', '|---|---:|---:|---:|']
    for r in results['readouts'][3:]:
        row = summary_ix['hard', 'gain_full', 40, r]
        md.append(f"| {r} | {pct(row['accuracy']['mean'])} | {pct(row['rescue_rate']['mean'])} | {pct(row['harm_rate']['mean'])} |")
    md += ['', '## 核验与范围', '',
           f"独立核验通过，覆盖 {verification.get('artifact_count')} 份轨迹文件、"
           f"{verification.get('episodes'):,} 条轨迹。另从检查点重放全部门控，以及新增控制器的动作决策。"
           '历史控制器路径通过原始文件哈希及路径一致性校验复用。', '',
           '机器可读 metrics.json 保留所有控制器/读出/预算的五种子统计、配对差值、逐场景效应，'
           '包括历史 joint 路径上的辅助验证与无特征控制器路径上的迁移检查。'
           '完整动作数组和模型权重保留在本地 output/active_evidence_ablations。', '',
           '本消融不区分 raw 与 SAM 特征各自的贡献，不评估移除上游图像分类器，'
           '不重训无特征 joint 控制器，也没有增加全新场景。一步前瞻、真实框假设和固定上游模型等限制仍然存在。', '']
    report = '\n'.join(md)
    (docs / 'report.md').write_text(report, encoding='utf8')
    (out / 'report.md').write_text(report, encoding='utf8')
    embeds = ''.join('<img style="max-width:100%" src="data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode() + '">' for p in sorted(docs.glob('*.png')))
    page = '<!doctype html><meta charset="utf-8"><title>Evidence ablations</title><body style="max-width:1200px;margin:30px auto;font-family:system-ui">' + embeds + '<pre style="white-space:pre-wrap">' + html.escape(report) + '</pre></body>'
    (out / 'report.html').write_text(page, encoding='utf8')
    print('Published verified ablation report: ' + str(docs), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'output/active_evidence_ablations')
    parser.add_argument('--docs', type=Path, default=ROOT / 'docs/active_evidence_ablations')
    args = parser.parse_args()
    build(args.out, args.docs)


if __name__ == '__main__':
    main()
