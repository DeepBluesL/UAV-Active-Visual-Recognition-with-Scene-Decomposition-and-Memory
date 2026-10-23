import unittest

from diagnostics.evidence_fusion import label_for_scene, predict_readouts, score


class EvidenceTests(unittest.TestCase):
    def test_stop_duplicate_does_not_bias_mean(self):
        p = [[.8, .2, 0, 0, 0], [.3, .7, 0, 0, 0], [.3, .7, 0, 0, 0]]
        r, info = predict_readouts([[0, 0, 0], [1, 0, 0], [1, 0, 0]], p, [0, 24])
        self.assertEqual(r['mean_visits'], 0)
        self.assertEqual(info['actual_observations'], 2)
        self.assertEqual(info['stop_duplicate_removed'], 1)

    def test_revisit_deduplication_changes_weighting(self):
        a, b = [.8, .2, 0, 0, 0], [.1, .9, 0, 0, 0]
        r, info = predict_readouts([[0, 0, 0], [1, 0, 0], [0, 0, 0]], [a, b, a], [0, 1])
        self.assertEqual(r['mean_unique'], 1)
        self.assertEqual(r['mean_visits'], 0)
        self.assertEqual(info['revisits'], 1)

    def test_unknown_does_not_vote_for_first_class(self):
        r, _ = predict_readouts([[0, 0, 0], [0, 0, 0]], [[0]*5, [0]*5], [24])
        self.assertTrue(all(x == -1 for x in r.values()))

    def test_invisible_terminal_can_recover_history(self):
        r, _ = predict_readouts([[0, 0, 0], [1, 0, 0]], [[0, 1, 0, 0, 0], [0]*5], [0])
        self.assertEqual(r['final'], -1)
        self.assertEqual(r['max_confidence'], 1)
        self.assertEqual(r['mean_unique'], 1)

    def test_fusion_can_beat_any_correct_view_reference(self):
        p = [[.40, .45, .15, 0, 0], [.40, .15, .45, 0, 0]]
        pred, info = predict_readouts([[0, 0, 0], [1, 0, 0]], p, [0])
        t = dict(scene=1, probabilities=p, predictions=['Pickup', 'Sedan'],
                 initial_correct=False, final_correct=False, distance_m=10)
        c = score(t, pred, info)
        self.assertEqual(c['oracle'], 0)
        self.assertEqual(c['mean_unique_correct'], 1)
        self.assertEqual(c['mean_unique_correct_without_correct_view'], 1)

    def test_earliest_confidence_tie(self):
        p = [[.9, .1, 0, 0, 0], [.1, .9, 0, 0, 0]]
        r, _ = predict_readouts([[0, 0, 0], [1, 0, 0]], p, [0])
        self.assertEqual(r['max_confidence'], 0)

    def test_scene_class_order(self):
        self.assertEqual(label_for_scene(61), 3)  # Sportscar precedes Sedan in scene blocks.
        self.assertEqual(label_for_scene(91), 2)

    def test_bad_stop_and_changed_static_observation_rejected(self):
        with self.assertRaises(ValueError):
            predict_readouts([[0, 0, 0], [1, 0, 0]], [[1, 0, 0, 0, 0]]*2, [24])
        with self.assertRaises(ValueError):
            predict_readouts([[0, 0, 0], [0, 0, 0]], [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], [0])


if __name__ == '__main__':
    unittest.main()
