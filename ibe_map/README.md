# UEVAVD：SAM 场景分解与 GRU 历史记忆

> **历史记录：本页记录五种子实验之前的单训练种子结果。** 当前最新结果为五随机种子实验，见 [五种子报告与统计说明](MULTISEED.md) 和 `output/ibe_multiseed/report.html`；此页数值仅用于保留前序实验记录。

本目录在原 MAP 基线上实现 Memo-MAP（GRU）、SAM-MAP（仅 SAM 的消融）和 IBE-MAP（SAM+GRU）。比较对象另含原地、扫描、随机和原 MAP。当前主结果是25个 hard 场景 × 605 个共同起点，共15,125回合/策略。全量评估另含 train、simple 与固定移动诊断，共27组指标、726,000条轨迹。

**SAM 使用官方预训练 ViT-B，尚未进行论文中的 UEVAVD 微调：公开数据副本未包含像素级目标mask，作者微调权重也未提供。** ResNet50仍冻结，实验仍为单训练种子。网络结构补齐不意味着原论文数值已完整复现，具体对应与差异见 [PROTOCOL.md](PROTOCOL.md)。

已按用户要求完整盘点 `dataset/multiview_imgs` 与 `dataset/train_605`：共94,392张RGB PNG和151份bbox JSON，无其他文件类型。场景104额外12张未配框图像与对应原图逐字节一致，属于重复副本。场景00和这12张重复图均不进入实验。证据保存于 `output/ibe_map/dataset_inventory.json`；该结论只针对当前发布副本，不代表作者原始采集没有分割GT。

## 历史单种子 hard 测试结果

下表为 `results.json` 中 hard split 实测值。准确率、回报和距离均按15,125回合汇总；距离单位为米。

| 策略 | 准确率 | 平均回报 | 平均距离 |
|---|---:|---:|---:|
| 原地（stationary） | 76.89% | 0.2689 | 0.00 |
| 扫描（sweep） | **92.33%** | 0.2584 | 30.21 |
| 随机（random） | 87.84% | 0.2275 | 24.65 |
| MAP | 87.97% | 0.3216 | 9.90 |
| Memo-MAP | 89.61% | **0.3266** | 11.09 |
| SAM-MAP | 89.92% | 0.3170 | 15.14 |
| IBE-MAP | 89.40% | 0.3195 | 12.59 |

每个策略在25个测试场景的605个视点上使用相同起点，环境分类输出、奖励、动作集合及决策预算统一。hard 结果中的 Memo-MAP 平均回报最高；IBE-MAP 的回报低于原 MAP，因此本次结果没有显示 IBE-MAP 胜过 MAP 的回报。按场景配对 bootstrap（2,000次、95%百分位区间），IBE-MAP−MAP 的平均回报差为 −0.00219，区间 [−0.01319, 0.00927] 跨过0；这是单个训练种子的评估，不能据此宣称稳健优势或跨种子显著性。

开发阶段将100个训练场景内部划分为90个训练、10个验证，先运行约30,000回合；正式阶段重新初始化，使用全部100个训练场景运行约300,000回合，训练种子为42。分类器及 ResNet50 特征编码器冻结，七策略共享相同分类输出、奖励定义、动作空间和最多5次决策预算。SAM 使用官方预训练 ViT-B，**不是在 UEVAVD 上微调的权重**；当前公开副本没有目标 mask，作者微调权重也未提供。ResNet 冻结且仅一个训练种子，以上结果属于本实现和本数据副本的单次实验。

独立核验结果记录于 `output/ibe_map/verification.json`：27组指标、726,000条轨迹、CSV行数和唯一回合键数均一致，`passed: true`。`run.log` 同时记录了31项测试通过，以及对726,000条轨迹的独立核验通过。

`output/ibe_map/ablation.json` 是 hard split 上以场景等权、按相同起点配对的消融汇总，包含各场景指标及2,000次场景 bootstrap 的95%区间。它支持在固定评估场景上的描述性比较，不估计训练种子不确定性，也不能证明跨种子的统计优势。

`results.json` 另记录三步固定移动诊断：强制移动三次后停止时，IBE-MAP准确率91.40%、平均距离39.78米、平均回报−0.0418；强制移动增加的距离与步数成本压低了回报，故此诊断用于观察移动预算效应，不替代原生停止策略的主结果。固定三次移动控制次数，并不控制相同米数。

## 一键运行

```powershell
& 'E:\Code\UEVAVD\run_ibe.ps1'
```

默认Python为 `D:\anaconda3\envs\uav\python.exe`。首次运行下载Meta官方SAM源码与权重，逐图生成SAM分解图和ResNet50特征；后续核验并复用缓存。分解进度按批次保存，网络下载支持断点续传。已有checkpoint时跳过训练，重新评估所有策略并核验轨迹。

```powershell
# 重新训练三种新增策略，保留原MAP作为固定基线
& 'E:\Code\UEVAVD\run_ibe.ps1' -Retrain

# 显存较少时降低SAM图像批次
& 'E:\Code\UEVAVD\run_ibe.ps1' -SamBatchSize 1
```

## 分步执行

先进入 `E:\Code\UEVAVD`。以下命令不需要安装torchvision，SAM模型的官方torch源码保存于项目内。

```powershell
& 'D:\anaconda3\envs\uav\python.exe' ibe_map\bootstrap_sam.py
& 'D:\anaconda3\envs\uav\python.exe' -m ibe_map.features

& 'D:\anaconda3\envs\uav\python.exe' -m ibe_map.experiment train --variant memo_map --phase development --episodes 30000
& 'D:\anaconda3\envs\uav\python.exe' -m ibe_map.experiment train --variant memo_map --phase full --episodes 300000
# 将variant改为sam_map、ibe_map，完成另外两个实验。

& 'D:\anaconda3\envs\uav\python.exe' -m ibe_map.experiment evaluate
& 'D:\anaconda3\envs\uav\python.exe' -m ibe_map.verify_results
```

`evaluate --splits hard --out output/ibe_map_hard_only` 可单独生成hard结果；`--policies map memo_map` 可只比较指定策略。新的输出目录避免覆盖先前结果。`verify_results --out ...` 可核验对应评估目录。

## 输出

- `output/ibe_map/report.html`：七策略总体指标、全部split、限制说明和真实SAM分解预览。
- `output/ibe_map/examples/report.html`：原图上的学习策略轨迹，包含成功和失败样例。
- `output/ibe_map/results.json`：原始指标、缓存/模型身份、训练配置和来源。
- `output/ibe_map/evaluation.csv`、`trajectories.jsonl.gz`：全部回合和逐步观察、动作、概率、奖励。
- `output/ibe_map/verification.json`：独立CPU核验及文件SHA256。
- `output/ibe_map/sam_preview`：固定选取视点的原图/目标mask/分解图，以及实际点提示、灰度和mask数组。
- `output/ibe_map/decomposed/<scene>`：全部实际SAM处理图，目标RGB保留，周边分割对象随机灰度化。
- `output/ibe_map/{memo_map,sam_map,ibe_map}/{development,full}`：训练日志与checkpoint；全量使用最终checkpoint。

GRU训练均匀采样transition，但每条样本携带完整回合前缀。online和target分别重建hidden；padding不产生loss，终止不bootstrap，episode间重置。`test_recurrence.py`覆盖这些性质及缓存类型误标检查。`test_sam.py`检查采样点、目标RGB保留与随机灰度可复现性。`verify_results.py`独立重算全量轨迹的识别、距离、奖励和汇总，不导入训练代码或torch。
