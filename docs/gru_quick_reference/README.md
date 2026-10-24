# GRU quick experiment

Variant: `ibe_map`; requested budget: 5000 episodes per run.

| Seed | Split | E2E accuracy | Frozen accuracy | E2E return | Frozen return | E2E distance (m) | Frozen distance (m) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | simple | 0.8446 | 0.8645 | 0.2173 | 0.2417 | 18.04 | 17.62 |
| 42 | hard | 0.8754 | 0.8542 | 0.2849 | 0.2691 | 12.58 | 13.55 |
| 42 | hard_fixed_3_moves | 0.7859 | 0.8105 | -0.1229 | -0.1159 | 33.64 | 39.80 |

## Training and parameter audit

Per-run actual episodes, updates, elapsed time, GRU tensor deltas, input hashes, and verification counts are in [metrics.json](metrics.json).

## Limits

- 仅为1–2个seed的短预算探索，不计算跨seed置信区间或显著性，也不构成完整论文复现。
- 训练预算为每组请求5,000回合；探索率日程按短预算压缩。full表示训练场景全集，不表示300k回合。
- SAM为预训练权重，未在UEVAVD上微调。
- 旧MAP 300k结果与本短预算不构成公平对照。
