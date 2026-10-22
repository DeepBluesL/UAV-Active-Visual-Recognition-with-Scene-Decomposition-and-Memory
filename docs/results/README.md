# 发布的实验摘要

本目录从 `output/ibe_multiseed/summary.json` 导出本地已完成实验的聚合数据，并复制该实验生成的曲线。不是重新训练结果，也不是论文作者报告的指标。

- `metrics.json`：五种子逐策略均值、样本标准差、95% t 区间、逐种子值、配对差值、训练总量及原实验核验计数。
- `curves.json`：各检查点的原始曲线数据，未经平滑。
- `curve_hard_accuracy.png`：hard split 准确率。
- `curve_hard_return_mean.png`：hard split 平均折扣回报。
- `curve_gengap.png`：train 回报减 hard 回报的泛化差距。

`metrics.json` 记录原始摘要及曲线文件的 SHA256，供来源核对。原实验核验记录中的 `passed` 表示原始本地完整产物当时通过核验；发布包不包含模型、CSV 与逐步轨迹，不能仅凭此目录重新执行完整轨迹核验。

完整产物可按根目录 README 的流程生成。正式核验入口为 `python -m ibe_map.verify_multiseed --out output/ibe_multiseed`，它检查各种子的结果、CSV/轨迹哈希、检查点、统计重算及曲线对应关系。
