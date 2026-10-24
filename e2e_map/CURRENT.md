# 当前范围：完成 MAP，再测试训练性能

> 本页保留 2026-10-22 的 MAP 收尾记录。2026-10-24 用户要求继续 GRU，小规模新增实验见 [GRU_QUICK.md](GRU_QUICK.md)：一个 seed，端到端与冻结 SAM+GRU 各 5,000 回合；旧五种子队列仍不恢复。

用户最终选择只完成正在运行的 **seed 42 MAP、300,000 episodes**。其余 9 次端到端任务不再执行，包括尚未开始的 SAM+GRU。已完成的 10 份 BF16 冻结对照保留，只使用同 seed 的冻结 MAP 作当前对照。

原 MAP 训练沿用启动时的配置与代码，不在中途改变数值或重计算设置。收尾程序不包含训练调用，只等待当前进程完成，随后评估端到端与冻结 MAP 的 train/simple/hard 全网格及 hard 固定三步诊断，共 211,750 条轨迹，并运行独立核验。结果写入 `docs/e2e_map_reference/`。这是单种子 MAP 参考，不能作稳定改善或 SAM+GRU 效果结论。

正式训练和评估现已完成：MAP 实际 300,005 回合、10,662 次更新、AMP 跳过 0 次；训练循环累计约 45 分钟。两组共 211,750 条轨迹全部通过核验，详见[实际结果](../docs/e2e_map_reference/README.md)。独立的执行优化与短测见 [PERFORMANCE.md](PERFORMANCE.md)。

```powershell
# 当前训练完成后，仅评估及核验；不会追加训练任务。
D:\anaconda3\envs\uav\python.exe -m e2e_map.finish_current
Get-Content output/e2e_multiseed/status.json
```

随后在独立目录测试训练计算效率：保持有效 batch、TD 目标、奖励和 BatchNorm 语义不变，比较重计算、图像微批与像素传输方式。性能测试与模型识别效果分开记录；更高显存占用本身不代表更高吞吐。正式训练结果不混入优化后的短测试。

短测现已完成，选定关闭激活重算、microbatch=64：单次更新 255.4 → 211.4 ms，吞吐 +20.8%，数值检查通过。随后完成唯一一次 100 更新真实训练检查，AMP 跳过 0 次。所有本次训练、评估和性能测试均已结束，没有追加种子或 SAM+GRU 完整训练。

原 `run_e2e.ps1` 五种子流程与 `e2e_map.pilot` 双策略流程保留为历史实现，均不再用于当前任务。原计划清单保留，最新范围由 `scope_manifest.json` 记录。
