# 单种子初步对照

**历史计划，已停用：** 用户随后取消了 SAM+GRU，仅保留正在运行的 MAP。请使用 [CURRENT.md](CURRENT.md) 中的当前流程；以下双策略命令不应作为当前任务入口。

按用户要求，当前执行计划缩减为 **seed 42 的 MAP + IBE-MAP（SAM+GRU）**。每种方法仍训练 300,000 episodes，与同 seed 的 BF16 冻结版本比较。取消其余 8 次端到端训练，保留已经完成的 10 份冻结对照和当前 MAP 训练进度。

当前 MAP 进程由新调度器等待，不重启、不重复占用 GPU；之后只启动一次 SAM+GRU。两者完成后，评估四组策略的 train/simple/hard 全部起点及 hard 固定三步诊断，共 423,500 条轨迹。使用原独立核验器，并核对训练预算、checkpoint、数据身份和相同实验配置。

结果写入 `docs/e2e_pilot/`，明确标记单种子参考；只报告本次观测到的指标与差值，不报告跨种子标准差、置信区间或显著性。不能据此认定稳定改善，也不能单独归因 SAM 或 GRU。SAM 仍未微调。

```powershell
# 中断后恢复精简任务；已有完整训练与核验结果会复用。
D:\anaconda3\envs\uav\python.exe -m e2e_map.pilot --publish
Get-Content output/e2e_multiseed/status.json
```

`pilot_manifest.json` 记录缩减计划及代码哈希。原五种子 `run_manifest.json` 保留为历史计划，不改写成已完成。后台程序核验通过后自动生成总结并上传现有 GitHub 仓库，成功后写入 `publication.json`。当前模式请使用 `e2e_map.pilot`，不要启动原五种子 `run_e2e.ps1`。
