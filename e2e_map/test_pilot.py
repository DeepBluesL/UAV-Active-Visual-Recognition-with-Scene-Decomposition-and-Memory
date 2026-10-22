import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import patch
from . import pilot


class PilotTests(unittest.TestCase):
    def test_single_seed_plan_and_descriptive_differences(self):
        self.assertEqual(pilot.SEED, 42)
        self.assertEqual(pilot.POLICIES, ('map', 'ibe_map'))
        keys = ('accuracy', 'return_mean', 'distance_m', 'generalization_gap_return')
        values = {'frozen_map': 1., 'e2e_map': 2., 'frozen_ibe_map': 3., 'e2e_ibe_map': 6.}
        groups = {name: {'hard': dict.fromkeys(keys, value)} for name, value in values.items()}
        result = pilot.differences(groups)
        self.assertEqual(result['e2e_minus_frozen']['map']['accuracy'], 1.)
        self.assertEqual(result['e2e_minus_frozen']['ibe_map']['accuracy'], 3.)
        self.assertEqual(result['difference_in_differences']['accuracy'], 2.)
        self.assertNotIn('ci95_t', str(result))

    def test_missing_arm_rejected_and_publication_is_separate(self):
        with self.assertRaises(KeyError):
            pilot.differences({})
        self.assertTrue(all(path.startswith('docs/e2e_pilot/') for path in pilot.RESULTS))
        self.assertFalse(any('*' in path or path.startswith(('output/', 'dataset/'))
                             for path in pilot.SOURCES + pilot.RESULTS))

    def test_controller_schedules_only_two_seed42_trainings(self):
        with tempfile.TemporaryDirectory(dir=pilot.p.ROOT / 'tmp') as folder:
            args = SimpleNamespace(out=Path(folder), frozen=Path(folder), follow_pid=None,
                                   microbatch=64, device='cuda', publish=False)
            with patch.object(pilot, 'identity'), patch.object(pilot, 'wait_existing') as waiting, \
                 patch.object(pilot, 'evaluate_control'), patch.object(pilot, 'summarize'), \
                 patch.object(pilot, 'status'), patch.object(pilot.p, 'single_process', return_value=nullcontext()), \
                 patch.object(pilot.p, 'keep_awake', return_value=nullcontext()), \
                 patch.object(pilot.p, 'complete_train', side_effect=[False, True, False, True]), \
                 patch.object(pilot.p, 'verify_current', return_value=True), patch.object(pilot.p, 'run') as run:
                pilot.execute(args)
            waiting.assert_called_once_with(args)
            self.assertEqual(run.call_count, 2)
            commands = [call.args[1] for call in run.call_args_list]
            self.assertEqual([cmd[cmd.index('--variant') + 1] for cmd in commands], ['map', 'ibe_map'])
            self.assertTrue(all(cmd[cmd.index('--seed') + 1] == 42 for cmd in commands))


if __name__ == '__main__':
    unittest.main()
