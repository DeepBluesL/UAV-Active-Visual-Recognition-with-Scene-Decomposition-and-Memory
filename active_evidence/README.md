# Budgeted observation value and causal evidence gating

This stage adds two independently implemented components to the UEVAVD project:

1. **One-step observation value:** predict the probability of correct recognition after each candidate movement and after stopping. Move only when predicted improvement exceeds movement cost and the action fits the remaining distance budget.
2. **Discrete evidence gating:** estimate the reliability of replacing memory with the new observation, retaining memory, or averaging the two. The selected evidence becomes the final recognition output. A second controller predicts gains after this gate update, closing the perception/action loop.

The gate is compared with final-view, maximum-confidence and unique-view mean readouts on **exactly the same acquired trajectories**. The joint controller follows its own paths and is compared under the same hard distance caps, with actual distance reported separately.

These are project contributions relative to the preceding reproduction; no claim of globally novel algorithms or state-of-the-art performance is made. The method uses dense supervised outcomes from training-scene neighboring views, rather than the original reinforcement-learning objective. Differences against historical IBE therefore reflect the complete new method, not an isolated architecture substitution.

## Completed experiment

The [published report](../docs/active_evidence/report.md) and [machine-readable summary](../docs/active_evidence/metrics.json) contain the verified five-seed results. Models and large trajectory arrays remain in local `output/active_evidence/`.

Development uses 90 training scenes and selects a movement-cost coefficient on 10 other training scenes, averaged over five seeds. The method and coefficients are frozen before final retraining on all 100 training scenes and evaluation on simple/hard. These benchmark splits were inspected in the preceding exploratory diagnostic; they are not new untouched confirmation data.

The predeclared [protocol](PROTOCOL.md) fixes seeds 42–46, 20 training epochs, MLP sizes, cost grid, and distance caps **0, 10, 20, 40, 60 m**. Every final comparison covers all 605 starts per scene, with at most five movements. A common cap does not imply equal actual distance. The old IBE checkpoint is used only as a separately labeled final reference, not to generate development supervision or tune the new method.

## Completed ablations

The follow-up [report](../docs/active_evidence_ablations/report.md) and [findings](../docs/active_evidence_ablations/FINDINGS.md) narrow the evidence-gating claim to **learned selective retention/replacement**. The ablations reuse the 90-scene development / 10-scene validation / 100-scene final split and seeds 42–46. Full-feature controls exactly reproduced the original gate and gain checkpoints. The no-extra-embedding models still receive the frozen classifier probabilities, bbox and position; this is not a vision-free system.

At the hard 40 m cap, on the same trajectories, final-view accuracy is 90.70%, the latest-view rule is 90.73%, the learned binary gate is 91.80%, and the three-way gate is 91.84%. The three-way minus binary difference is +0.037 percentage points (95% CI [−0.016, +0.090]), so these results do not establish that fusion is necessary. The no-extra-embedding gain controller reaches 92.32% at 13.69 m on hard versus 90.70% at 14.51 m for full features, while simple accuracy falls from 92.76% to 91.70%; the results do not support a general claim that embeddings help.

Run the ablation workflow with `.\run_active_evidence_ablations.ps1`. It writes local models, paired rollouts and summaries under `output/active_evidence_ablations/`; detailed method and artifact notes are in the report and findings links above.

## Model details

Frozen raw/SAM ResNet50 features are standardized per image and projected through a fixed, unfitted Gaussian matrix to 64 dimensions each. Bbox and current position produce a 135-dimensional observation. The gain model receives this observation and the current classifier probabilities; the joint version also receives the causal memory. Small 128/64 MLPs output 25 correctness logits or three gate logits.

Training labels for action outcomes use the known neighboring training views. At inference, the action predictor sees only the current observation and acquired history. It never reads the actual feature/probabilities of a candidate unvisited viewpoint. Labels, scene IDs and filenames are excluded from model inputs.

Gate training predicts each candidate's top-1 correctness with binary cross-entropy. This avoids the degeneracy of choosing the smallest true-class NLL among old/new/arithmetic-mean candidates, where averaging cannot strictly beat both endpoints. Invisible observations and repeated locations retain existing evidence. All-invisible histories remain Unknown. STOP does not append a duplicate observation.

## Run

Use the existing dependencies in `requirements-ibe.txt`. First prepare the original and SAM observation caches and historical five-seed IBE checkpoints through `run_multiseed.ps1`. The downloaded dataset and model files are not included in Git.

```powershell
.\run_active_evidence.ps1
```

Portable commands from the repository root:

```bash
python -m unittest active_evidence.test_model
python -m active_evidence.experiment --stage development
python -m active_evidence.experiment --stage final
python -m active_evidence.verify
python -m active_evidence.report
```

Both experiment and verification support `--out`; the experiment also supports `--device`. The PowerShell wrapper accepts `-Python`. Training checkpoints are reused only when configuration, code identity, cache identity and saved checkpoint hashes match. If interrupted mid-training, that small training task restarts; this is not optimizer-state resumption. A final rerun re-evaluates the final checkpoints.

Reports require complete results and independent verification. Result artifacts include:

- `selection.json`: internal-validation coefficients, all candidate scores and initial-state gain calibration bins.
- `development/`, `final/`: per-seed model files and complete BCE training histories.
- `rollouts/*.npz`: common starts, causal observation indices, actions, distance, all readouts and per-step gate choices.
- `results.json`: per-scene and aggregate results, selection and rollout hashes.
- `verification.json`: independent replay of motion, probability readouts and summary statistics.
- `report.html`, `report.md`, plots: local presentation; bounded counterparts in `docs/active_evidence/` are suitable for GitHub.

## Interpretation

Five-seed intervals condition on these fixed scenes and upstream models; paired gate-minus-final intervals compare identical paths. Deterministic stationary/sweep paths do not make their learned gate outputs deterministic across gate-training seeds. Random-controller variation describes evaluation RNG; IBE-reference variation describes its historical policy training, and joint/gain variation describes the new models.

The controller estimates **one-step** benefit and cannot anticipate a necessary initially unhelpful detour. Ground-truth target boxes, pretrained SAM without UEVAVD fine-tuning, a frozen classifier, discrete motion and fixed upstream features remain limitations. Future improvement should be developed on training-scene validation and assessed with new scenes, matched actual-cost analysis and visual-input ablations.
