"""Render the fixed-trajectory evidence diagnostic from saved aggregate counts."""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'stationary': '原地', 'sweep': '扫描', 'random': '随机', 'map': 'MAP',
         'memo_map': 'Memo-MAP', 'sam_map': 'SAM-MAP', 'ibe_map': 'IBE-MAP'}
METHOD_NAMES = {'final': '最终帧', 'max_confidence': '最高置信度帧',
                'mean_unique': '概率平均（视点去重）', 'mean_visits': '概率平均（保留重访）'}
POLICIES = tuple(NAMES)


def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    data = json.loads((out / 'summary.json').read_text(encoding='utf8'))
    examples = json.loads((out / 'examples.json').read_text(encoding='utf8'))
    lookup = {(r['split'], r['mode'], r['policy'], r['subset']): r for r in data['summary']}
    ibe = lookup[('hard', 'native', 'ibe_map', 'all')]
    md, body = [], []

    def heading(text, level=2):
        md.append('#' * level + ' ' + text + '\n')
        body.append(f'<h{level}>{html.escape(text)}</h{level}>')

    def paragraph(text):
        md.append(text + '\n')
        body.append(f'<p>{html.escape(text)}</p>')

    def table(headers, rows):
        md.extend(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'])
        md.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
        md.append('')
        body.append('<div class="table"><table><thead><tr>' + ''.join(f'<th>{html.escape(h)}</th>' for h in headers) + '</tr></thead><tbody>')
        body.extend('<tr>' + ''.join(f'<td>{html.escape(str(x))}</td>' for x in row) + '</tr>' for row in rows)
        body.append('</tbody></table></div>')

    def ci(s):
        return f"[{s['ci95_t'][0]:+.3f}, {s['ci95_t'][1]:+.3f}]" if s['ci95_t'] else '—'

    heading('固定轨迹多视角证据诊断', 1)
    paragraph('分析日期：2026-10-23。使用五种子已有轨迹，比较相同已获取图像的不同分类读出。没有重新训练，也没有改变轨迹、停止时间或移动距离。')
    total = sum(x['analyzed'] for x in data['verification']['per_seed'])
    paragraph(f"共扫描 {sum(x['scanned'] for x in data['verification']['per_seed']):,} 条轨迹，分析 simple/native、hard/native 和 hard/固定三次移动，共 {total:,} 条。每种子每策略每组 25 场景 × 605 起点 = 15,125 回合。")
    heading('主要结果')
    final = 100 * ibe['methods']['final']['accuracy']['mean']
    oracle = 100 * ibe['oracle_rate']['mean']
    paragraph(f"IBE-MAP 原生停止：最终帧准确率 {final:.3f}%；最高置信度帧 {100*ibe['methods']['max_confidence']['accuracy']['mean']:.3f}%；去重概率平均 {100*ibe['methods']['mean_unique']['accuracy']['mean']:.3f}%；历史至少一帧认对 {oracle:.3f}%。")
    paragraph(f"已有正确观察但最后答错占全部回合 {100*ibe['lost_correct_rate']['mean']:.3f}%，占最终错误 {100*ibe['lost_fraction_of_final_errors']['mean']:.2f}%。历史中从未有一帧 top-1 正确占 {100*ibe['never_correct_rate']['mean']:.3f}%；这不等于这些回合不可能通过融合或新视角纠正。")
    table(['读出', '准确率 %', '相对最终帧 pp', '配对 95% t 区间 pp', '救回 pp', '误伤 pp'], [
        [METHOD_NAMES[m], f"{100*v['accuracy']['mean']:.3f}", f"{v['delta_vs_final_pp']['mean']:+.3f}",
         ci(v['delta_vs_final_pp']), f"{v['rescued_pp']['mean']:.3f}", f"{v['harmed_pp']['mean']:.3f}"]
        for m, v in ibe['methods'].items()])
    paragraph('救回：最终帧错误、该读出正确；误伤：最终帧正确、该读出错误。二者均以全部回合为分母，净提升 = 救回 − 误伤。')

    # Export a publication-style figure, with explicit units and no hidden smoothing.
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    learned = ['map', 'memo_map', 'sam_map', 'ibe_map']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout='constrained')
    x = np.arange(len(learned))
    colors = ['#6b7280', '#007f86', '#e48b29', '#8461ad']
    for j, method in enumerate(['final', 'max_confidence', 'mean_unique', 'oracle']):
        vals = [100*(lookup[('hard', 'native', p, 'all')]['oracle_rate']['mean'] if method == 'oracle' else
                     lookup[('hard', 'native', p, 'all')]['methods'][method]['accuracy']['mean']) for p in learned]
        axes[0].bar(x + (j-1.5)*.19, vals, .18, color=colors[j], label=['Final view', 'Max confidence', 'Mean (unique)', 'Any-correct-view oracle'][j])
    axes[0].set_xticks(x, [NAMES[p] for p in learned])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel('Accuracy / oracle coverage (%)')
    axes[0].set_title('Hard / native stopping — five-seed means')
    axes[0].legend(loc='lower left', fontsize=8)
    plot_rows = []
    for mode in ('native', 'fixed_moves'):
        r = lookup[('hard', mode, 'ibe_map', 'all')]
        for m in ('max_confidence', 'mean_unique'):
            plot_rows.append((('Native' if mode == 'native' else 'Fixed 3 moves') + '\n' + ('Max conf.' if m == 'max_confidence' else 'Mean'), r['methods'][m]))
    xx = np.arange(len(plot_rows))
    axes[1].bar(xx, [v['rescued_pp']['mean'] for _, v in plot_rows], color='#007f86', label='Rescued errors')
    axes[1].bar(xx, [-v['harmed_pp']['mean'] for _, v in plot_rows], color='#c45757', label='Harmed correct cases')
    net = np.array([v['delta_vs_final_pp']['mean'] for _, v in plot_rows])
    intervals = [v['delta_vs_final_pp']['ci95_t'] for _, v in plot_rows]
    if all(intervals):
        errors = np.array([[net[i]-s[0] for i,s in enumerate(intervals)], [s[1]-net[i] for i,s in enumerate(intervals)]])
        axes[1].errorbar(xx, net, yerr=errors, fmt='o', color='#1f2937', capsize=4, label='Net gain + paired 95% CI')
    else:
        axes[1].plot(xx, net, 'o', color='#1f2937', label='Net gain')
    axes[1].axhline(0, color='#777', linewidth=.8)
    axes[1].set_xticks(xx, [name for name, _ in plot_rows])
    axes[1].set_ylabel('Percentage points of all episodes')
    axes[1].set_title('IBE-MAP: rescue versus harm')
    axes[1].legend(fontsize=8)
    fig.savefig(out / 'fusion_diagnostic.png', dpi=180)
    fig.savefig(out / 'fusion_diagnostic.svg')
    plt.close(fig)
    md.append('![Evidence fusion diagnostic](fusion_diagnostic.png)\n')
    encoded = base64.b64encode((out / 'fusion_diagnostic.png').read_bytes()).decode()
    body.append(f'<img alt="Fixed trajectory fusion diagnostic" src="data:image/png;base64,{encoded}">')

    for split, mode, title in [('hard', 'native', '困难测试：原生停止'),
                               ('simple', 'native', '简单测试：原生停止'),
                               ('hard', 'fixed_moves', '困难测试：固定三次移动（不同米数）')]:
        heading(title)
        values = []
        for policy in POLICIES:
            r = lookup.get((split, mode, policy, 'all'))
            if not r:
                continue
            values.append([NAMES[policy], *[f"{100*r['methods'][m]['accuracy']['mean']:.3f}" for m in ('final', 'max_confidence', 'mean_unique', 'mean_visits')],
                           f"{100*r['oracle_rate']['mean']:.3f}", f"{r['unique_visible_mean']['mean']:.2f}", f"{r['distance_m_mean']['mean']:.2f}"])
        table(['策略', '最终帧 %', '最高置信度 %', '去重平均 %', '保留重访平均 %', '曾认对 %', '可见独立视点数', '距离 m'], values)
    paragraph('不同读出在同一策略的轨迹上距离完全相同。固定三次移动与原生停止的路径和距离不同，不能将两个模式的准确率差解释为公平的同成本收益。')

    heading('IBE-MAP 子集与逐种子结果')
    rows = []
    for subset, title in [('all', '全部回合'), ('multiple_positions', '经过多个位置'),
                          ('multiple_visible_views', '至少两个可见视点'), ('disagreeing_views', '可见视点类别有分歧')]:
        r = lookup.get(('hard', 'native', 'ibe_map', subset))
        if not r:
            continue
        rows.append([title, f"{sum(r['episodes_per_seed'])/len(r['episodes_per_seed']):.1f}",
                     *[f"{100*r['methods'][m]['accuracy']['mean']:.3f}" for m in ('final', 'max_confidence', 'mean_unique')],
                     f"{100*r['oracle_rate']['mean']:.3f}"])
    table(['子集（有重叠）', '平均回合/种子', '最终帧 %', '最高置信度 %', '去重平均 %', '曾认对 %'], rows)
    with (out / 'counts.csv').open(encoding='utf8', newline='') as f:
        counts = list(csv.DictReader(f))
    seed_rows = [r for r in counts if r['split']=='hard' and r['mode']=='native' and r['policy']=='ibe_map' and r['scene']=='all' and r['subset']=='all']
    table(['种子', '回合数', '最终正确', '最高置信度正确', '平均正确', '曾认对', '平均救回', '平均误伤'], [
        [r[k] for k in ('seed', 'n', 'final_correct', 'max_confidence_correct', 'mean_unique_correct', 'oracle', 'mean_unique_rescue', 'mean_unique_harm')]
        for r in seed_rows])
    paragraph(f"IBE-MAP 原生停止中，最终帧置信度 ≥0.90 但分类错误的回合占全部回合 {100*ibe['final_wrong_high_conf_rate']['mean']:.3f}%；最高置信度选帧仍以 ≥0.90 置信度答错的回合占 {100*ibe['max_conf_wrong_high_conf_rate']['mean']:.3f}%。这里只描述高分误判，不等价于完整的校准误差评估。")
    paragraph(f"真实重复访问平均每回合 {ibe['revisits_mean']['mean']:.4f} 次。视点去重与保留重访的差值只反映本批静态轨迹中的完全重复，不衡量相邻不同视点之间的相关性。")

    probe_path = out / 'matched_stop_probe.json'
    if probe_path.exists():
        from diagnostics.evidence_fusion import sha256
        probe = json.loads(probe_path.read_text(encoding='utf8'))
        if not probe['passed'] or probe['base_summary_sha256'] != sha256(out / 'summary.json'):
            raise ValueError('Stop probe does not match this diagnostic')
        heading('配对停止诊断：是否还有值得获取的证据？')
        ps = {r['subset']: r for r in probe['summary'] if r['policy'] == 'ibe_map'}
        immediate = [r for r in probe['rows'] if r['policy']=='ibe_map' and r['subset']=='immediate_stop']
        all_by_seed = {r['seed']: r for r in probe['rows'] if r['policy']=='ibe_map' and r['subset']=='all'}
        fraction = statistics.mean(r['n']/all_by_seed[r['seed']]['n'] for r in immediate)
        error_fraction = statistics.mean((r['n']-r['native_correct']) / (all_by_seed[r['seed']]['n']-all_by_seed[r['seed']]['native_correct']) for r in immediate)
        recovered = statistics.mean(r['final_rescue']/(r['n']-r['native_correct']) for r in immediate)
        paragraph(f"在同种子、同策略、同场景、同起点上配对原生轨迹与固定三次移动轨迹。IBE-MAP 有 {100*fraction:.2f}% 的回合在初始观察后立即 STOP；这些回合贡献了全部最终错误的 {100*error_fraction:.2f}%（逐种子比例的均值）。该子集按已执行动作定义，不使用真实类别挑选。")
        table(['IBE-MAP 子集', '原生最终 %', '固定三步最终 %', '固定三步最高置信度 %', '固定三步曾认对 %', '原生距离 m', '固定距离 m'], [
            [title, *[f"{100*ps[subset]['metrics'][k]['mean']:.3f}" for k in ('native_correct','final_correct','max_confidence_correct','oracle_correct')],
             f"{ps[subset]['metrics']['native_distance']['mean']:.2f}", f"{ps[subset]['metrics']['fixed_distance']['mean']:.2f}"]
            for subset,title in [('all','全部回合'),('immediate_stop','初始立即停止'),('moved_before_stopping','曾移动后停止/耗尽预算')]])
        v = ps['immediate_stop']['metrics']
        paragraph(f"初始立即停止子集中，固定三步后的最终帧纠正了原有错误的 {100*recovered:.2f}%；以该子集全部回合为分母，救回 {100*v['final_rescue']['mean']:.3f}%，误伤 {100*v['final_harm']['mean']:.3f}%。平均增加移动 {v['fixed_distance']['mean']:.2f} m。因此存在继续观察的诊断空间，但不能把无条件多移动当作有效的自适应策略。")
        paragraph('对于原生会移动的回合，强制三步也可能比原生更早结束，因此两个模式的观察集合不一定包含彼此。oracle 的跨模式下降不代表计算错误。')

    heading('下一步决策建议')
    paragraph('这批证据不支持直接将所有历史概率平均，或把最大 softmax 分数当作可靠性。原生学习策略已倾向于结束在更有用的视点；简单融合会重新引入之前的错误预测。具体因果机制仍需进一步验证。')
    paragraph('建议优先研究预算约束下的继续观察价值与停止判断，并把可靠性门控的证据更新作为辅助：新观察与历史冲突时，学习保留、替换或融合。当前固定轨迹的理想选帧空间较小，而增加观察存在明显的收益/成本权衡。')
    paragraph('后续开发应在训练场景内部验证。至少比较原生策略、固定观察预算、基于置信度的停止，以及学习到的继续观察收益；同时保留最终帧、最高置信度与平均融合读出基线。在相同距离预算下报告准确率，单独检查救回与误伤。此次诊断未训练这些新方法。')

    heading('解释样例（事后选取）')
    for category, title in [('mean_rescue', '概率平均救回错误'), ('mean_harm', '概率平均破坏正确结果'),
                             ('max_conf_harm', '最高置信度选中错误帧'), ('lost_correct', '历史认对而最终答错')]:
        e = examples.get(category)
        if not e:
            continue
        t = e['trajectory']
        paragraph(f"{title}：seed={data['seeds'][0]}，场景 {t['scene']}，起点 {t['positions'][0]}，真值 {e['truth']}。最终帧 {e['readouts']['final']}；最高置信度 {e['readouts']['max_confidence']}；概率平均 {e['readouts']['mean_unique']}。")
        end = e['preprocessing']['actual_observations']
        table(['观察', '位置', '单帧预测', '最大置信度', '真类概率（只作解释）'], [
            [i, str(t['positions'][i]), t['predictions'][i], f"{max(t['probabilities'][i]):.3f}",
             f"{t['probabilities'][i][('Hatchback','Pickup','Sedan','Sportscar','SUV').index(e['truth'])]:.3f}"] for i in range(end)])

    heading('统计口径与边界')
    for text in [
        '未知观察使用全零概率，不参与平均或最高置信度选帧；全程未知则输出 Unknown，计为错误。STOP 日志中的末帧重复全部去除。去重平均对每个访问过的可见位置等权；保留重访平均保留真实回访。',
        '最高置信度帧只读取已获取图像的预测分数，不读真值；置信度相同选最早帧。概率平均直接对原分类概率求均值，不作温度标定或阈值搜索。',
        '“曾认对”使用真实类别，表示历史单帧 top-1 至少正确一次。它是理想选帧参考，不能部署，也不是任意融合方法的上限：几张单帧都错的概率仍有可能融合正确。',
        '学习策略报告五训练种子的均值；配对区间来自同种子同轨迹的读出差值。确定性原地/扫描只计一次，随机策略差异来自评估随机性。五种子重复观察同一批场景，不视为五倍独立场景；区间不估计新场景泛化，且未作多重检验校正。',
        '所有方法使用截至原轨迹终点已观察到的图像，因此读出比较没有未来视图泄漏。任何“提前停止”的效果都不能由这些终点读出直接推出，未来必须按每个时刻的因果前缀重新评估。',
        '本次没有读取未访问视点来挑选动作，没有拟合融合器，也没有计算新策略的回报。结果只检验已有轨迹中的识别证据潜力；收益预测与停止机制仍需独立开发、验证。',
        'hard/simple 已用于此次探索，后续方法选择应使用训练场景内部验证，冻结方法后再报告测试，并争取新的未参与开发场景。',
    ]:
        paragraph(text)
    heading('核验与复现')
    checks = data['verification']['per_seed']
    paragraph(f"源文件 SHA-256 与既有独立验证记录匹配；{sum(x['reproduced_metric_groups'] for x in checks)} 组最终帧准确率和距离重现原报告；每场景完整覆盖 605 个唯一共同起点；独立 NumPy 实现复核 {sum(x['independent_numpy_episode_checks'] for x in checks):,} 条分层抽样轨迹。通过标记：{data['verification']['passed']}。")
    paragraph('运行：D:\\anaconda3\\envs\\uav\\python.exe -m diagnostics.evidence_fusion，然后运行同一 Python 的 -m diagnostics.stop_probe 和 -m diagnostics.evidence_report。单元测试：-m unittest diagnostics.test_evidence_fusion。')
    paragraph('summary.json 保存协议、种子统计、输入哈希与核验信息；counts.csv 保存逐种子、逐场景及条件子集的整数计数；examples.json 保存可追溯的示例原始轨迹。')
    (out / 'report.md').write_text('\n'.join(md), encoding='utf8')
    style = 'body{max-width:1150px;margin:40px auto;padding:0 24px;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif;color:#203043;background:#fafcfd}h1,h2{line-height:1.3}h1{color:#006f75}h2{margin-top:36px;border-bottom:1px solid #dce4e8;padding-bottom:8px}p{max-width:1050px}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{padding:9px 10px;border-bottom:1px solid #dde5e9;text-align:left;white-space:nowrap}th{background:#e8f2f3}tr:nth-child(even){background:#f5f8fa}img{width:100%;height:auto;margin:22px 0}'
    (out / 'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>多视角证据诊断</title><style>'+style+'</style><body>'+'\n'.join(body)+'</body></html>', encoding='utf8')
    print(out / 'report.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'output/evidence_fusion')
    render(parser.parse_args().out)
