# 训练计算效率优化

本次只完成既有 seed 42 MAP 的 300,000 回合训练。性能修改在独立模块中运行，不修改该正式训练的代码、权重或结果，也不追加完整训练任务。

## 修改内容

原实现将整个 ResNet50 的中间激活丢弃，在反向传播时重新计算，以降低显存占用。`performance.py` 提供三种执行方式：保留原始整网重算 `full`、保存激活而不重算 `none`、仅对 stage 3/4 重算 `late`。图像微批可以单独调整，有效 replay batch 始终为 256。

另一项可选优化将训练图像以 uint8 缓存在 GPU，减少每次更新的 CPU 图像收集和主机到显卡传输。缓存只有原始像素；可训练的在线图像特征仍逐次计算，target 特征仍在每次 target 同步后失效。100 个训练场景的 60,500 张 224×224 RGB 图像约需 8.48 GiB。

这些调整保留 BF16 卷积、FP32 特征与策略头、Adam、Double DQN、奖励、采样、梯度裁剪和固定的 BatchNorm 运行统计量。显存占用高低不是优化目标，实际比较的是在数值检查通过时每秒完成的更新数。

## 测量方法

`benchmark.py` 复制一个完成训练的 MAP resume checkpoint，再让每个候选在独立进程中读取完全相同的模型、target、Adam、GradScaler、replay、目标特征缓存和随机采样索引。每次测量包含 64 个当前状态的动作选择、256 条真实 replay 转移的 TD 前向与反向，以及 Adam 更新；不包含环境状态推进和磁盘保存。因此这些数据不是完整训练耗时或识别准确率实验。

各候选先预热，计时前后同步 CUDA，分别记录已分配和保留的峰值显存，冷 target 缓存更新单列。基线在首尾各运行一次；两次速度差超过 10% 时不自动选出最佳配置。候选运行较短，小幅差异应视为测量噪声。

速度比较之前核对同一批样本的损失、全网络梯度和动作。要求损失差不超过 `1e-6 + 0.005 × |baseline loss|`、梯度相对 L2 差不超过 2%、梯度余弦相似度至少 0.999、动作完全相同，且没有 AMP 跳过或非有限参数。改变卷积微批会改变浮点运算顺序，因此不要求逐比特一致。

通过数值门槛且速度合适的配置，再运行一次独立的 100 次更新真实训练检查。该检查使用原始 300,000 回合探索率日程，并提前停止，不能当作完整训练结果。

## 参考资料

- [PyTorch activation checkpointing](https://pytorch.org/blog/activation-checkpointing-techniques/)：以额外计算换取更低激活显存。
- [Microsoft Task Manager GPU 统计说明](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/)：3D 与 Compute 引擎可能共享计算核心，任务管理器的引擎名称不能单独判断机器学习负载是否正确使用 GPU。

## 本机短测结果

GPU：NVIDIA GeForce RTX 5090 D；PyTorch：2.14.0+cu130。每项预热 5 次、计时 20 次更新，基线首尾各测一次。

| 配置 | ms / update | 相对基线速度 | warm 峰值分配 / 保留（GiB） | 数值检查 |
|---|---:|---:|---:|---|
| baseline_full64 | 256.3 | 1.00× | 3.84 / 4.88 | 通过 |
| none64 | 211.4 | 1.21× | 11.97 / 12.39 | 通过 |
| none128 | 210.2 | 1.21× | 11.97 / 13.44 | 未通过 |
| full64_gpu_pixels | 224.3 | 1.14× | 12.32 / 13.37 | 通过 |
| none128_gpu_pixels | 184.5 | 1.38× | 20.47 / 21.96 | 未通过 |
| baseline_full64_repeat | 254.6 | 1.00× | 3.84 / 4.88 | 通过 |

基线两次均值为 255.4 ms/update，首尾差异 0.7%。本次选择 `none64`：211.4 ms/update，相对基线吞吐 1.21×。这是固定 replay 的短测结果，不代表完整训练必然按同样比例提速。

`none64` 在所测输入上的损失和完整梯度与原配置逐位相同，动作一致率 100%。两个 microbatch=128 候选的梯度相对 L2 差约 4.61%，超过预先设定的 2% 门槛，因此没有采用，也没有放宽门槛。显存字段来自 PyTorch 的分配器统计，不等同于任务管理器包含驱动、桌面和其他程序的整卡显存占用。

随后真实训练检查完成 100 次更新、1,856 回合，训练循环记录耗时 22.73 秒；AMP 跳过 0 次，各 ResNet stage 参数均发生有限变化，BatchNorm 运行统计保持固定。像素缓存建立和程序初始化不计入该训练循环时间。

9 项 CPU 单元测试也已通过，覆盖重算方式的梯度与 BN 语义、像素索引、在线特征更新、完整输入要求及短测限制。可用 `python -m unittest e2e_map.test_performance` 复查。

[完整性能记录](../docs/e2e_map_reference/performance.json)包含各候选的损失、梯度差异、动作一致性、冷 target 测量、源码和输入哈希，以及短训练配置。

## 复现命令

以下命令只运行短测；目录必须与正式结果隔离。

```powershell
python -m e2e_map.benchmark --resume-checkpoint output/e2e_multiseed/seed_42/map/train/resume.pt --out output/map_performance --warmup 5 --steps 20 --candidates baseline_full64 none64 none128 full64_gpu_pixels none128_gpu_pixels
python -m e2e_map.performance --variant map --out output/map_performance_validation --checkpoint-mode none --microbatch 64 --stop-after-updates 100
```
