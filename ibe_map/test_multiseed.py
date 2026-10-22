"""Regression checks for seed units, pairings, and deterministic references."""
import math
import unittest

from ibe_map.multiseed import LEARNED, POLICIES, seed_stats, snapshot_budgets, summarize_runs


class MultiseedStatisticsTests(unittest.TestCase):
    def test_five_seed_student_t_uses_sample_sd(self):
        result = seed_stats([1, 2, 3, 4, 5])
        self.assertEqual(result['n'], 5)
        self.assertEqual(result['ddof'], 1)
        self.assertAlmostEqual(result['std'], math.sqrt(2.5))
        self.assertAlmostEqual(result['ci95_t'][1], 3 + 2.776445105 / math.sqrt(2))

    def test_deterministic_duplicates_are_not_independent_seeds(self):
        result = seed_stats([.5] * 5, deterministic=True)
        self.assertEqual(result['n'], 1)
        self.assertEqual(result['values'], [.5])
        self.assertIsNone(result['ci95_t'])
        with self.assertRaises(ValueError):
            seed_stats([.5, .51], deterministic=True)

    def test_single_seed_has_no_invented_interval(self):
        result = seed_stats([.5])
        self.assertIsNone(result['std'])
        self.assertIsNone(result['ci95_t'])

    def test_snapshot_grid_includes_initial_and_nonmultiple_final(self):
        self.assertEqual(snapshot_budgets(61, 30), [0, 30, 60, 61])

    def test_pair_differences_follow_run_identity(self):
        runs = []
        for seed in (46, 42, 45, 43, 44):
            rows = []
            for policy in POLICIES:
                value = .5 if policy in ('stationary', 'sweep') else seed / 100
                if policy == 'memo_map':
                    value += .01
                rows.append({'split': 'hard', 'policy': policy, 'episodes': 15125,
                    'accuracy': value, 'return_mean': value, 'distance_m': value})
            runs.append({'seed': seed, 'metrics': rows})
        summary, pairs = summarize_runs(runs)
        memo_map = next(p for p in pairs if p['left_policy'] == 'memo_map' and p['right_policy'] == 'map')
        self.assertAlmostEqual(memo_map['metrics']['accuracy']['mean'], .01)
        self.assertLess(memo_map['metrics']['accuracy']['std'], 1e-15)
        self.assertEqual(next(r for r in summary if r['policy'] == 'random')['randomness'], 'evaluation_seed')
        for policy in LEARNED:
            self.assertEqual(next(r for r in summary if r['policy'] == policy)['metrics']['accuracy']['n'], 5)
        with self.assertRaises(ValueError):
            summarize_runs(runs + [runs[0]])


if __name__ == '__main__':
    unittest.main()
