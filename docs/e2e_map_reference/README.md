# ResNet50 端到端 MAP：单种子参考

本报告只包含 seed 42 的 MAP：当前端到端 MAP 与同 seed、同 BF16 编码器精度的冻结 MAP。两个训练任务各请求 300,000 回合；端到端任务实际完成 300,005 回合，冻结任务实际完成 300,033 回合。所有评估 split 均使用固定的全部起点，并通过独立轨迹核验。

**只有一个训练 seed，因此这里只报告逐组原始值和端到端减冻结的描述性差值；不报告跨 seed SD、置信区间、显著性或稳定提升结论。没有运行 E2E IBE-MAP/SAM+GRU，不能根据 MAP-only 结果判断 SAM+GRU 的变化。**

### Train split

| 组别 | 准确率（%） | 平均折扣回报 | 平均移动距离（m） |
|---|---:|---:|---:|
| 同 seed BF16 冻结 MAP | 97.82 | 0.3995 | 11.12 |
| 端到端 MAP | 99.20 | 0.4189 | 10.10 |
| 端到端 − 冻结（描述性差值） | +1.39 pp | +0.0195 | -1.02 |

### Simple split

| 组别 | 准确率（%） | 平均折扣回报 | 平均移动距离（m） |
|---|---:|---:|---:|
| 同 seed BF16 冻结 MAP | 91.06 | 0.3219 | 13.27 |
| 端到端 MAP | 95.00 | 0.3649 | 12.25 |
| 端到端 − 冻结（描述性差值） | +3.93 pp | +0.0430 | -1.02 |

### Hard split

| 组别 | 准确率（%） | 平均折扣回报 | 平均移动距离（m） |
|---|---:|---:|---:|
| 同 seed BF16 冻结 MAP | 90.02 | 0.3253 | 11.96 |
| 端到端 MAP | 91.99 | 0.3562 | 12.16 |
| 端到端 − 冻结（描述性差值） | +1.97 pp | +0.0309 | +0.20 |

### Hard split：固定三步诊断

| 组别 | 准确率（%） | 平均折扣回报 | 平均移动距离（m） |
|---|---:|---:|---:|
| 同 seed BF16 冻结 MAP | 90.55 | -0.0480 | 39.90 |
| 端到端 MAP | 95.09 | -0.0193 | 42.17 |
| 端到端 − 冻结（描述性差值） | +4.54 pp | +0.0287 | +2.27 |

每个方法各有 105,875 条经独立核验的轨迹，合计 211,750 条。训练 checkpoint、配置和核验文件的 SHA-256、更新数、训练时长及实际回合数见 [`metrics.json`](metrics.json)。完整身份记录见输出目录的 `scope_manifest.json`；旧五种子计划及其源哈希保持为 provenance。

SAM 使用官方预训练权重，未在 UEVAVD 上微调。本文仅比较 MAP，不作 SAM+GRU 结论，也不把 encoder 可训练性变化解释为 SAM 或 GRU 的独立效果。

## English brief

This is a seed-42 MAP-only reference: end-to-end ResNet50 MAP versus the same-seed BF16 frozen MAP control. Each run requested 300,000 episodes and both were independently verified on all evaluation splits. With one seed, the report gives descriptive values and paired differences only; it makes no confidence-interval or significance claim. E2E IBE-MAP/SAM+GRU was not run, so these MAP results cannot answer whether SAM+GRU benefits from end-to-end training.
