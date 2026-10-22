# UEVAVD 阶段 1/2：环境对齐与 MAP 基线

已经完成实际训练和评估。MAP 学习移动方向、10–40 m 步长和 STOP，不再使用规则扫描选择动作。训练/验证/测试划分、奖励和边界行为有明确协议与测试。

本次实现复用了现有五类车辆分类器，以冻结 ImageNet ResNet50 提取当前全图特征，再训练无记忆的 MAP 决策头。它是依据论文描述重建的基线；冻结编码器、Q 分支合成和论文未说明的训练细节都有明确记录，**不声称与未公开的作者实现逐项一致**。本阶段没有加入 GRU 或 SAM。

## 直接查看

- 总体报告：`E:\Code\UEVAVD\output\map\report.html`
- 可播放的学习策略轨迹：`E:\Code\UEVAVD\output\map\examples\report.html`
- 对照图：`output/map/comparison.png`
- 完整配置、指标与来源：`output/map/results.json`
- 全部 episode 汇总：`output/map/evaluation.csv`
- 全部逐步轨迹：`output/map/trajectories.jsonl.gz`，可用 Python `gzip.open(..., 'rt')` 逐行读取 JSON。

## 本次实测

模型先在 90 个训练场景上训练，另外 10 个原训练场景用于开发验证；再重新初始化，在原文全部 100 个训练场景上训练 300,009 个 episodes（目标 300,000，64 并行环境导致最后一批额外完成 9 个），共 706,240 次转移、11,004 次梯度更新。全量训练使用最终 checkpoint，不按 hard test 选择模型。训练种子 42。

Hard test 的 25 场景 × 每场景全部 605 起点 = **15,125 个共同起点**，分类器、合法动作、5 次决策上限和奖励相同：

| 策略 | 准确率 | 平均移动 | 平均折扣回报 |
|---|---:|---:|---:|
| 原地观察 | 76.89% | 0.00 m | 0.2689 |
| 规则扫描 | **92.33%** | 30.21 m | 0.2584 |
| 随机移动 | 87.84% | 24.65 m | 0.2275 |
| MAP | 87.97% | **9.90 m** | **0.3216** |

MAP 不是准确率最高的策略。它比规则扫描少移动约 67.2%，准确率低约 4.36 个百分点，在这套含移动和决策成本的奖励下回报最高。MAP 在训练集准确率 98.54%，simple test 92.17%，hard test 87.97%；训练到 hard 的回报差约 0.0820，仍有明显泛化差距。

另做 hard test **固定三次移动后停止**的诊断：扫描/随机/MAP 准确率分别为 82.40% / 77.75% / 90.69%，对应平均距离 106.55 / 68.03 / 40.12 m。此实验控制移动次数，**不等于相同米数预算**；只用于分析移动动作选择与停止规则的影响，不替代主实验。

共保存 408,375 个 episode（训练、simple、hard 的四策略比较，以及 hard 的三策略固定移动诊断）。报告中的置信区间按场景 bootstrap，不能解释成多训练种子的波动范围。

这些结果不能直接与旧 demo 的 89.93% 或论文表格比较：本次覆盖全部 605 起点，最多 5 次决策中允许第五次移动，奖励也统一了。这里没有通过测试结果调整超参数来追求好看的数字。

## 运行

已经配置好的 PowerShell 一键入口：

```powershell
& 'E:\Code\UEVAVD\run_map.ps1'
```

它核对特征缓存、运行环境测试；已有训练 checkpoint 时跳过训练，重新执行完整评估与报告生成。使用 `D:\anaconda3\envs\uav\python.exe`，不需要安装 torchvision。原始数据及旧 demo 的模型、输出均保留。

重新训练：

```powershell
& 'E:\Code\UEVAVD\run_map.ps1' -Retrain
```

分步命令（先进入 `E:\Code\UEVAVD`）：

```powershell
# 核对或生成所有观测缓存
& 'D:\anaconda3\envs\uav\python.exe' -m map_baseline.features

# 90 train / 10 validation，使用原文训练场景内部划分
& 'D:\anaconda3\envs\uav\python.exe' -m map_baseline.experiment train --phase development --episodes 30000

# 全部100训练场景，固定预算，最终checkpoint
& 'D:\anaconda3\envs\uav\python.exe' -m map_baseline.experiment train --phase full --episodes 300000

# 全训练集、simple、hard，包含固定三移动诊断
& 'D:\anaconda3\envs\uav\python.exe' -m map_baseline.experiment evaluate

# 只重跑hard评估，另存目录
& 'D:\anaconda3\envs\uav\python.exe' -m map_baseline.experiment evaluate --splits hard --out output\map_hard_only

# 仅从现有结果重新绘图，不重新评估
& 'D:\anaconda3\envs\uav\python.exe' -c "from map_baseline.report import render_report; render_report('output/map/results.json','output/map')"
```

训练默认目录分别是 `output/map/development`、`output/map/full`。`--device cpu` 可用于功能测试；GPU 实际训练较快是因为完整图像的 ResNet50 编码已缓存，冻结主干，仅训练投影和 Q 网络，不是端到端 ResNet50 训练时间。

本机 ResNet50 全150场景缓存提取约 149 秒；开发阶段训练约 6 秒、全量 Q 网络训练约 53 秒（不含加载缓存等准备工作）。这些数字是本机日志，不保证其他机器的速度。只训练一个随机种子，尚未完成论文的五种子完整统计。

若数据、分类器或预处理改变，缓存准备会检测到变化；旧 checkpoint 与新缓存身份不符时评估会报错，需要重新训练，不能混用后继续报告“同一分类器”。

## 验证与实现入口

一键入口已完整跑通，12 项环境测试通过。另对全部 408,375 条 episode 做了独立交叉核验：CSV、逐步轨迹与 15 组汇总一致；动作边界、实际距离、每步奖励、折扣回报、成功判定与固定三次移动均通过。核验记录及输入文件 SHA-256 保存于 `output/map/verification.json`。HTML 报告已用本机 Chrome 渲染检查。

- `environment.py`：605 点离散网格、25 个动作、场景清单、统一奖励与启发式策略。
- `model.py`：图像 512 + bbox 32 → 状态 512，状态价值/方向优势/幅度优势分支。
- `experiment.py`：Replay、Double DQN、训练、完整评估和轨迹导出。
- `features.py`：ResNet50 权重校验、固定分类器推理、观测缓存与指纹。
- `test_environment.py`：12 项环境单元测试。
- `report.py`：离线结果报告和对照图。为避免本机 PyTorch 与绘图库的 OpenMP 运行库冲突，报告在独立子进程生成，没有关闭运行库安全检查。

完整 reward、success threshold、STOP、第五步时序、边界、冻结主干、训练超参及公平性说明见 [PROTOCOL.md](PROTOCOL.md)。
