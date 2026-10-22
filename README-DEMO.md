# UEVAVD 复现与运行记录

**当前阶段：SAM 场景分解 + GRU 历史记忆策略已完成五随机种子实验。** 七策略 hard 结果、五种子配对比较、训练曲线和限制见 [五随机种子报告与统计说明](ibe_map/MULTISEED.md)。最新报告为 `output/ibe_multiseed/report.html`。运行入口为 `& 'E:\Code\UEVAVD\run_multiseed.ps1'`，脚本参数、分阶段运行方式和结果限制见 [MULTISEED.md](ibe_map/MULTISEED.md)。

阶段1/2的 MAP 基线材料仍见 [MAP 运行与结果](map_baseline/README.md) 和 [实验协议及与原文的差异](map_baseline/PROTOCOL.md)，作为前序阶段记录，不代表当前最新阶段。单训练种子结果保留于 [单种子 SAM+GRU 文档](ibe_map/README.md)，作为历史阶段记录。

## 历史规则扫描演示（阶段1之前）

以下内容保留最初的简化 demo，其固定起点采样和步数口径不同于后续 MAP 与 SAM+GRU 实验。**该 demo 未实现 SAM 场景分解、GRU 历史表示或学习策略；不能作为当前 SAM+GRU 实验结果。**

已在本机实际训练并运行，演示论文的核心效果：**无人机改变观察位置，避开遮挡，从而改善车辆识别**。使用本地 UEVAVD 原始图像，没有生成或伪造识别结果。

直接双击 `output/demo/report.html`，点击 **Play** 或拖动滑块；下拉菜单可查看五类车辆的成功案例及一个失败案例。无需启动服务或联网。`output/demo/sequence.gif` 是首个案例动画，`sequence.png` 是连续帧对比。

## 已跑出的结果

25 个 hard-test 场景，每场景 27 个预先固定的起点，共 **675 次实验**；两种移动策略使用同一组起点。

| 方式 | 最终识别准确率 | 平均移动距离 |
|---|---:|---:|
| 原地观察 | 74.96%（506/675） | 0 m |
| 随机方向和步长，置信度达到阈值停止 | 86.07%（581/675） | 23.13 m |
| 简化扫描策略，置信度达到阈值停止 | **89.93%（607/675）** | 32.83 m |

扫描策略有 116 次由错误/无法识别变为正确，也有 15 次由正确变为错误。它比随机策略移动更多，**不能据此声称实现了论文的最短路径优势**。这些是本 demo 的实测结果，不是论文指标的复现。

首个展示案例是场景 27：车辆初始被屋顶完全遮挡；上升和横向移动后，最终识别为 Hatchback，softmax 分数约 0.994。展示案例是在全部评估完成后挑选的，不能用单个成功案例代替总体统计。报告还保留了一个失败案例。

## 一键重跑

PowerShell 中执行：

```powershell
Set-Location E:\Code\UEVAVD
.\run_demo.ps1
```

脚本使用已经验证可用的 `D:\anaconda3\envs\uav\python.exe`，有模型时直接运行演示。已有预测缓存时，本机重跑约 8 秒。

也可直接运行 Python：

```powershell
& 'D:\anaconda3\envs\uav\python.exe' E:\Code\UEVAVD\demo\run_demo.py
```

重新训练分类器并生成演示：

```powershell
.\run_demo.ps1 -Retrain
```

单场景快速运行（另存结果，不覆盖默认报告）：

```powershell
& 'D:\anaconda3\envs\uav\python.exe' demo\run_demo.py --scenes 27 --checkpoint output\demo\classifier.pt --out output\scene27
```

`--device cpu` 可切换到 CPU。`--refresh-cache` 会重新执行分类推理。模型首次训练需要下载约 45 MB 的官方 ImageNet ResNet18 权重；本机已经缓存，后续可以离线运行。不需要 torchvision、Unreal Engine 或 AirSim。

## 与论文的对应关系

| 环节 | 本 demo 的实现 |
|---|---|
| 多视角环境 | 使用真实 UEVAVD 图像与 `bboxes.json`，每个场景 605 个视点 |
| 车辆分类 | ImageNet 预训练 ResNet18，在作者提供的 `train_605` 裁剪图上微调 |
| 已知目标位置 | 使用数据集真值框裁剪分类；绿框是 **GT bbox，不是模型检测框** |
| 动作 | 沿 x/y/z 单轴移动，每次 10–40 m，最多 5 次观察（最多 4 次移动） |
| 运动策略 | 先尝试升高，再横向扫描；当前分类置信度达到 0.90 时停止 |
| SAM 场景分解 | 未实现 |
| GRU 历史表示、Dueling DQN 策略学习 | 未实现；使用几何启发式策略 |
| 论文完整训练及多随机种子评估 | 未执行 |

作者的 [GitHub 仓库](https://github.com/Leo000ooo/UEVAVD_dataset) 在本次检查时只有 README 和数据下载入口，没有公开 IBE-MAP 训练代码、策略权重或微调后的 SAM。因此此处是**自行实现的轻量效果演示，不是作者完整 IBE-MAP 算法复现**。论文方法与实验设定见 [arXiv v1](https://arxiv.org/html/2411.04348v1)。

运动策略不读取真实车辆类别，也不查看尚未到达的视点分数。为加快实验，分类分数会预先缓存，但每一步策略只使用当前观察的分数。真实类别仅用于结束后的正确率统计和挑选展示案例。

## 数据与训练细节

- `train_605`：5 类，每类 605 张，共 3025 张。固定 seed=42，在每类内部划分 80% 训练/20% 验证，即 2420/605 张。
- 分类器：标准 ResNet18 的本地 PyTorch 实现，输入 128×128，ImageNet 归一化。官方初始权重的 SHA256 前缀 `f37072fd` 已检查。
- 训练：8 epochs，AdamW，学习率 0.0003，batch size 96，带亮度/对比度/色彩增强；按验证准确率、再按 NLL 选择模型。本机 RTX 5090 D 训练约 13 秒。
- 内部验证准确率为 100%，但相邻视角来自同一个无遮挡场景；这只是训练正常的检查，**不是独立场景的测试性能**。
- 演示评估场景：26–30、56–60、86–90、116–120、146–150。每场景从 x/y∈{-40,0,40}、z∈{-60,-80,-100} 的 27 个组合开始，seed=42。
- 文件坐标采用 AirSim NED：z 为负数，实际高度为 -z；上升对应 z 变得更负。
- `bboxes.json` 为 `[0,0,0,0]` 时目标完全不可见，输出 Unknown，不调用分类器。其他图像按给定可见目标框裁剪。
- softmax 分数不是校准后的正确概率，遮挡情况下也可能高分误判。
- 每场景只读取标注中列出的 605 张图，忽略少量没有标注的额外 PNG。原始数据和现有 Python 环境没有被修改。

## 文件

- `demo/classifier.py`：分类器、训练和批量推理。
- `demo/run_demo.py`：数据解析、运动策略、固定起点评估、结果导出。
- `demo/render_results.py`：PNG、GIF 和带概率条、XY 路径及播放控件的离线 HTML。
- `output/demo/classifier.pt`：已训练模型。
- `output/demo/training.json`：完整训练参数、训练/验证文件列表和每轮记录。
- `output/demo/results.json`：配置、总体指标、展示轨迹和模型哈希。
- `output/demo/evaluation.csv`：675 次扫描与 675 次随机实验的逐条记录。
- `output/demo/cache/`：官方初始权重和各场景分类缓存。预测缓存与模型哈希、数据路径、标注、图像大小及修改时间绑定。

已验证：模型保存后重新加载、五类样图推理、全部 25 场景评估、一键脚本缓存重跑、报告 JavaScript 语法以及 Chrome 无窗口渲染。
