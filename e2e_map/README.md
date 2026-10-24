# End-to-end ResNet50 policy training

**Current execution:** the seed-42 MAP reference and performance test are complete. On 2026-10-24, the user requested a small GRU continuation: one seed of E2E SAM+GRU and a matched frozen control, 5,000 episodes each. See [GRU_QUICK.md](GRU_QUICK.md) and the previous [MAP record](CURRENT.md). The five-seed design below is retained as the original full experiment plan, not a claim of completed results.

This stage compares **MAP and IBE-MAP (SAM+GRU)** with a trainable ResNet50 against newly matched BF16 frozen-backbone five-seed controls. The existing FP16 frozen results remain a separate historical reference. It is an independent experiment based on the [UEVAVD paper](https://arxiv.org/html/2411.04348v1), which identifies an ImageNet-pretrained ResNet50 but does not specify its freezing or BatchNorm policy. End-to-end training therefore fills a documented implementation gap; it is not proof of an exact author implementation.

本阶段固定种子 `42,43,44,45,46`，每种方法请求 **300,000 episodes**，端到端十次训练，另补十次同精度的冻结主干训练。最终结果仅在全部二十次完整训练、全网格评估及独立轨迹核验通过后生成。已有冻结版结果、原图和 SAM 图像缓存均保留。

## Controlled comparison

| Item | Frozen control and E2E treatment |
|---|---|
| Train / simple / hard | Same 100 / 25 / 25 scenes, all 605 starts per evaluation scene |
| Initialization | Same official ImageNet ResNet50 V1 weights; same-seed policy-head initialization |
| Input | Same original RGB for MAP; same existing SAM-decomposed RGB for IBE |
| Image preprocessing | PIL bilinear 224×224, ImageNet normalization, BF16 image encoder, FP32 feature boundary (both matched arms) |
| Recognition/rewards | Identical frozen ResNet18 probabilities on original bbox crops |
| Policy / replay | Same Q heads; IBE uses g544/GRU544/state512; complete causal prefix replay |
| Optimization | Adam 1e-4, batch256, replay100k, 64 environments, warmup2048, one update per vector step, target sync500 |
| Objective / schedule | Same Double DQN, SmoothL1, global gradient clip10, epsilon1→.05 over first70% |
| Episode / movement | Same Tmax5, 25 actions, gamma.9, terminal±.5, step−.1, distance−.02/unit |
| Selection | Final300k checkpoint; all training finishes before held-out evaluation |

**Changed variable:** all ResNet50 backbone convolution parameters and BatchNorm affine parameters now receive the TD gradient together with the policy heads/GRU. The unused ImageNet classifier layer is removed. BatchNorm running mean, variance and counters remain fixed at pretrained values. This avoids mixing other replay samples or future sequence frames into the current observation through batch statistics.

卷积采用 CUDA BF16 autocast，权重、优化器状态、输出特征和策略头为 FP32，GradScaler 初始 scale128。AMP 非有限梯度导致的跳过次数单独记录。全网络梯度裁剪包含主干，因此其对策略头的相对影响也可能变化。微批与 channels-last 卷积的浮点顺序可能不同，不声称跨实现逐比特一致。

FP16 先导训练在第 242 次 TD forward 出现溢出：固定同一权重和输入，以 FP32 诊断得到 layer4 最大激活约 84,082，超过 FP16 最大有限值 65,504，导致 Inf 经 LayerNorm 变为 NaN；参数和 target Q 本身仍有限。BF16 与 FP32 前向均正常。失败尝试不计入正式结果。为避免把精度变化误认为端到端效果，重新提取 BF16 冻结特征并训练五种子 MAP/IBE 作为主对照。新缓存的 probs/boxes/names/visible/positions 与原缓存逐项一致，只有特征及其精度描述变化。

## Training and recovery

```powershell
# Complete experiment; existing incomplete jobs resume with their optimizer/replay/RNG.
.\run_e2e.ps1

# Explicitly publish verified results and bounded source files to existing origin/main.
.\run_e2e.ps1 -Publish

# Read progress without modifying a running process.
Get-Content output/e2e_multiseed/status.json
Get-Content output/e2e_multiseed/seed_42/map/training.log -Tail 3
```

Linux/other Python environments can run `python -m e2e_map.pipeline --stage all`. The PowerShell wrapper accepts `-Python` for an alternate environment. `--stage plan` validates experiment identity without training; `train`, `evaluate`, `report`, and explicit `publish --publish` allow restarting a completed phase.

Each E2E run saves `resume.pt` at logging boundaries with online/target weights, Adam state, GradScaler, replay ring and pointer, partial environment episodes, random-generator states, and valid target feature cache. Completed jobs are skipped only after identity and budget checks. The short frozen-control jobs use the existing trainer, archive an incomplete job, and restart it from scratch. A process lock prevents two pipelines using the same output directory. Snapshots occur at 0/30k/…/300k; resume does not overwrite completed historical snapshots.

Replay stores observation indices, not frozen online features. The current observation and every historical frame influencing its GRU state are re-encoded with current weights on each update, with gradients retained. No stale GRU state crosses optimizer updates or episodes. Padding and terminal bootstraps do not contribute loss. Activation checkpointing and image microbatches preserve this computation while bounding GPU memory.

Target image features can be reused only while the target encoder is unchanged; every target synchronization invalidates this cache. They are persisted across interruption to preserve numerical replay. At final evaluation only, the fixed final encoder is run once over each image and its resulting features are reused for exhaustive trajectories.

The RGB cache stores only resized uint8 pixels, not trainable features, in memory-mapped files. Source PNG content hashes are recorded at construction; source names, sizes, timestamps and pixel-cache SHA256 are checked on reuse. Frozen cache metadata and raw/SAM progress signatures are cross-checked.

## Analysis and artifacts

The primary hard-set comparisons are paired by training seed:

1. E2E BF16 MAP − matched frozen BF16 MAP and E2E BF16 IBE − matched frozen BF16 IBE.
2. `(IBE − MAP) E2E BF16 − (IBE − MAP) frozen BF16`: whether the combined SAM+GRU advantage changes with encoder training.
3. E2E BF16 versus the historical FP16 frozen results is supplementary: both trainability and numerical precision change in that comparison.

Report accuracy, discounted return, movement distance and train-minus-hard return gap together. Accuracy alone cannot establish improved efficiency. Means, sample SD and 95% Student-t intervals use five seed values, with no multiple-comparison adjustment. These intervals condition on fixed scenes, classifier and SAM preprocessing. The second comparison evaluates the combined SAM+GRU treatment; it does not isolate either component.

Each new run evaluates train60,500 + simple15,125 + hard15,125 + hard-fixed-three-move15,125 = **105,875 trajectories**. Ten E2E runs yield **1,058,750 independently checked new trajectories**; the ten matched frozen controls add another **1,058,750**, for **2,117,500 new verified trajectories** in this stage. The published frozen stationary/sweep/random/Memo/SAM ablations remain useful context; only MAP and IBE are retrained in this stage, in both the matched BF16 frozen and E2E conditions.

- `output/e2e_multiseed/status.json`: actual stage, current run, or failure details.
- `control/`: BF16 feature caches, matched frozen training/evaluation and summary.
- `seed_*/{map,ibe_map}/train/`: training diagnostics, snapshots and resume state.
- `seed_*/{map,ibe_map}/evaluation/`: complete metrics, CSV, compressed trajectories and independent verification.
- `comparison_summary.json`, `comparison_report.html`: produced only after all matched runs pass.
- `docs/e2e_results/`: compact portable results exported only after completion.
- `publication.json`: remote commit identity, written only after successful push and remote-head verification.

Publishing is opt-in and uses the configured existing public repository. It excludes dataset, model weights, pixel caches, full trajectories and local credentials. It refuses a dirty staging area, a changed remote/branch, or unrelated local/remote divergence. A recorded commit created by this same pipeline may be pushed again after a network failure if the remote still points to its direct parent. It never force-pushes. Existing Git credentials are used noninteractively. During execution, a temporary Windows system-use request prevents automatic sleep; it allows the display to turn off and does not block explicit user sleep. This request is released on completion or process exit. A push/network failure leaves local results available and records a failed status for retry.

## Remaining differences from the paper

The released files provide RGB images and boxes, but no pixel segmentation labels or author-finetuned SAM weights. This stage still uses the existing official pretrained SAM ViT-B decomposition **without UEVAVD SAM fine-tuning**. Recognition and known target bbox remain fixed; no online tracker, flight dynamics or collision model is introduced. Replay, Q-branch composition, AMP and BatchNorm choices remain locally specified where the paper is silent.
