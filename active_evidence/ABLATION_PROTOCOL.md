# Predeclared gate and embedding ablations

Fixed before running these ablations on 2026-10-23. This extends, rather than
changes, [the original protocol](PROTOCOL.md). The existing simple/hard sets
have already been inspected: all results are exploratory benchmark re-evaluation.

## Shared controls

- Keep DEVELOPMENT 90 / VALIDATION 10 scenes, final TRAIN 100, seeds 42–46,
  frozen classifier and raw/SAM caches, projection seed 1729, 20 epochs,
  AdamW and MLP widths, five movements and caps 0/10/20/40/60 m unchanged.
- Preserve the original experiment/model source and outputs. Authenticate old
  selection, training manifests, checkpoint and trajectory hashes before reuse.
- Reproduce the original full-feature gate and gain training, then verify their
  weights against the original checkpoints. New gate variants share the exact
  original training examples, starting weights and minibatch order. New gain
  variants likewise share examples, starting weights and minibatch order.
- Gate training examples retain the original mixed causal-memory distribution,
  including arithmetic mixtures even for the binary gate. This controls the
  supervision distribution; it is not on-policy training for each variant.

## A. Evidence updates on identical acquired paths

Replay original gain-only paths as primary, original joint-controller paths as
secondary, and new no-embedding gain paths as a transfer check. These readouts do
not change the saved actions, stopping times, observations or distances:

1. Final frame, historical maximum-confidence frame and unique-view mean.
2. `rule_latest`: replace with each new visible unique observation.
3. `rule_agree`: if old/new top-1 agree, half-half fuse; otherwise use the higher
   maximum probability, ties choosing new. No fitted threshold.
4. `gate2_full`: separately train retain/replace correctness heads.
5. `gate3_full`: original retain/replace/fuse correctness heads.
6. `gate2_noembed` and `gate3_noembed`: matched variants without embeddings.

All gates/rules share invisible/repeated-view retention and first-visible
replacement. Unknown remains -1. The binary gate keeps the same hidden widths
and initial weights; the third output receives no loss and is never selectable.
Report rescue/harm and operation counts, not just accuracy. Primary contrasts:
gate2_full minus rule_latest/rule_agree; gate3_full minus gate2_full; gate3_full
minus each rule; full minus noembed within each learned gate.

## B. Extra visual embedding ablation

Zero both 64-dimensional raw and 64-dimensional SAM projected feature blocks at
every model invocation, during training and inference. Keep the architecture,
probabilities, bbox and position unchanged. Thus this removes **extra image
embeddings**, not all visual information: the classifier probabilities are still
image-derived. It does not distinguish raw features from SAM features.

- Gate-only effect: full/noembed gates on identical paths, without retraining
  their producing controller.
- Controller effect: retrain the gain-only predictor without embeddings; compare
  its final-frame output with the full-feature gain-only controller. Select its
  cost from the original grid on the same internal validation utility, averaged
  over five seeds and four nonzero caps; then retrain on all TRAIN scenes.
- Additionally evaluate the noembed gate on noembed gain paths (both new modules
  lack embeddings), and the full gate on those paths to separate readout effects.
- Reuse the full controller's already frozen validation coefficient. Report the
  noembed controller with its own validation-selected coefficient and a secondary
  run with the full controller's coefficient, to distinguish tuning from inputs.
- Do not retrain the joint controller in this study: its fixed paths are a
  secondary readout test, not a noembed joint-controller ablation.

## Reporting and audit

Evaluate all 605 starts in each of the 25 simple and 25 hard scenes. Primary
tables show every nonzero cap and both splits; 40 m hard is a readable anchor,
not a newly selected best outcome. Report actual distance alongside the cap.
Pair contrasts by seed, split, start and path where applicable. Use five-seed
paired t intervals (df=4), conditional on fixed scenes/upstream models, without
multiple-comparison correction. Retain per-scene summaries and artifacts.
Independently reconstruct path legality, cost, evidence updates, labels,
predictions and aggregates. Check neural decisions separately by model replay.
Publish source, protocol, bounded metrics and plots; caches/weights/full rollout
arrays remain local. Report negative or inconclusive results without modifying
architecture, rules, epochs or cost grid after test inspection.
