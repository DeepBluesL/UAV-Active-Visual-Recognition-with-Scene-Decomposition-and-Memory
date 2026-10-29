"""CPU evaluation regressions using the real environment and sequence rollout."""
import copy
import csv
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from map_baseline.environment import Dataset, GRID, Protocol, STOP
from e2e_map import evaluate as evaluation


class DirectionalHead(nn.Module):
    def __init__(self, recurrent):
        super().__init__()
        self.recurrent = recurrent

    def forward(self, features, boxes):
        q = features.new_full((*features.shape[:-1], 25), -1)
        q[..., 0] = 1
        q[..., STOP] = (features[..., 0] >= 0).float() * 2
        return q

    def forward_sequence(self, features, boxes):
        return self(features, boxes), None


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.features = np.zeros((1210, 2048), dtype=np.float32)
        self.features[:, 0] = np.tile(GRID[:, 0], 2)
        self.raw = Dataset.__new__(Dataset)
        self.raw.scenes, self.raw.scene_to_slot = [1, 26], {1: 0, 26: 1}
        self.raw.names = np.asarray([f'{i}.png' for i in range(1210)])
        self.raw.boxes = np.ones((1210, 4), dtype=np.float32)
        self.raw.visible = np.ones(1210, dtype=bool)
        self.raw.labels = np.zeros(1210, dtype=np.int64)
        self.raw.correct = self.features[:, 0] >= 0
        self.raw.predicted = (~self.raw.correct).astype(np.int64)
        self.raw.probs = np.eye(5, dtype=np.float32)[self.raw.predicted]
        self.raw.true_probability = self.raw.probs[:, 0]
        self.raw.confidence = self.raw.probs.max(axis=1)
        self.raw.fingerprint = 'raw'
        self.raw.observation_signature = {'preprocess': {}, 'classifier_sha256': 'classifier',
                                         'pretrained_sha256': 'encoder', 'dataset_path': str(self.root)}
        self.sam = copy.copy(self.raw)
        self.sam.fingerprint = 'sam'
        self.sam.observation_signature = {**self.raw.observation_signature,
                                         'preprocess': {'scene_decomposition': {'sam': 'fixed'}}}

    def run_variant(self, variant):
        data = self.sam if variant in ('sam_map', 'ibe_map') else self.raw
        model = nn.Module()
        model.head = DirectionalHead(variant in ('memo_map', 'ibe_map'))
        config = {'variant': variant, 'seed': 42, 'requested_episodes': 10, 'phase': 'full',
                  'protocol': Protocol().as_dict(), 'observation_signature': data.observation_signature}
        checkpoint = self.root / 'last.pt'
        torch.save({'config': config, 'completed_episodes': 10, 'state_dict': model.state_dict()}, checkpoint)
        (self.root / 'training.json').write_text(json.dumps({'config': config}), encoding='utf-8')
        args = SimpleNamespace(checkpoint=checkpoint, out=self.root / variant, splits=['train', 'hard'],
            raw_cache='raw', sam_cache='sam', data=self.root, decomposed=self.root, image_cache=self.root,
            weights=None, microbatch=64, device='cpu')
        original = vars(args).copy()
        with patch.object(evaluation, 'TRAIN', [1]), patch.object(evaluation, 'HARD', [26]), \
                patch.object(evaluation, 'Dataset', side_effect=lambda cache, scenes: getattr(self, cache)), \
                patch.object(evaluation, 'RGBImageStore', return_value=SimpleNamespace(fingerprint='rgb')), \
                patch.object(evaluation, 'E2EModel', return_value=model), \
                patch.object(evaluation.E2ERunner, 'image_features',
                             side_effect=lambda model, ids: torch.from_numpy(self.features[ids])), \
                patch('demo.render_results.render'), patch('builtins.print'):
            result = evaluation.evaluate(args)
        return result, args, original

    def test_variants_preserve_rollout_outputs_and_example_metrics(self):
        for variant in ('map', 'memo_map', 'sam_map', 'ibe_map'):
            with self.subTest(variant=variant):
                result, args, _ = self.run_variant(variant)
                metrics = {row['split']: row for row in result['metrics']}
                self.assertEqual(set(metrics), {'train', 'hard', 'hard_fixed_3_moves'})
                self.assertTrue(all(row['policy'] == variant and row['episodes'] == 605 for row in metrics.values()))
                self.assertEqual(metrics['hard']['generalization_gap_return'], 0)
                self.assertEqual(result['protocol']['evaluation_seed'], 42)
                with (args.out / 'evaluation.csv').open(encoding='utf-8', newline='') as source:
                    rows = list(csv.DictReader(source))
                self.assertEqual(len(rows), 1815)
                self.assertEqual({row['stop_reason'] for row in rows if row['mode'] == 'native'},
                                 {'policy_stop', 'max_decisions_after_move'})
                self.assertEqual({row['stop_reason'] for row in rows if row['mode'] == 'fixed_moves'},
                                 {'fixed_movement_budget'})
                with gzip.open(args.out / 'trajectories.jsonl.gz', 'rt', encoding='utf-8') as source:
                    traces = [json.loads(line) for line in source]
                self.assertEqual(len(traces), len(rows))
                self.assertTrue(all(len(row['positions']) == row['decisions'] + 1 for row in traces))
                self.assertTrue(all(row['actions'][-1] == STOP and row['decisions'] == 4
                                    for row in traces if row['mode'] == 'fixed_moves'))
                examples = json.loads((args.out / 'examples.json').read_text(encoding='utf-8'))
                self.assertEqual(examples['metrics'], metrics['hard'])
                self.assertEqual({row['policy'] for row in examples['episodes']}, {variant})

    def test_evaluation_does_not_mutate_arguments(self):
        _, args, original = self.run_variant('map')
        self.assertEqual(vars(args), original)


if __name__ == '__main__':
    unittest.main()
