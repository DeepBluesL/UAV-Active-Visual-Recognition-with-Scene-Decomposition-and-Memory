"""Protect final verification and the bounded publication contract."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import pipeline as p


class PipelineTests(unittest.TestCase):
    def setUp(self):
        directory = p.ROOT / 'tmp/e2e_pipeline_tests'
        directory.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=directory)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make_artifacts(self):
        base = self.root / 'seed_42/map'
        (base / 'train').mkdir(parents=True)
        (base / 'evaluation').mkdir()
        (base / 'train/last.pt').write_bytes(b'test checkpoint')
        p.write_json(base / 'train/training.json', {'config': {'variant': 'map'}})
        p.write_json(base / 'evaluation/results.json', {'protocol': {
            'checkpoint_sha256': {'map': p.sha256(base / 'train/last.pt')}}})
        for name in ('evaluation.csv', 'trajectories.jsonl.gz'):
            (base / 'evaluation' / name).write_bytes(b'test artifact')
        p.write_json(base / 'evaluation/verification.json', {'passed': True,
            'counts': {k: 105875 for k in ('trajectories', 'csv_rows', 'unique_episode_keys')},
            'input_sha256': {str((base / 'evaluation' / name).resolve()): p.sha256(base / 'evaluation' / name)
                for name in ('results.json', 'evaluation.csv', 'trajectories.jsonl.gz')}})
        return base

    def test_relative_paths_and_tampered_result_are_checked(self):
        base = self.make_artifacts()
        # Child processes receive absolute --out; lookup also resolves paths.
        relative = base.relative_to(Path.cwd())
        self.assertTrue(p.verify_current(relative))
        (base / 'evaluation/evaluation.csv').write_bytes(b'changed')
        self.assertFalse(p.verify_current(relative))

    def test_old_checkpoint_and_bad_counts_rejected(self):
        base = self.make_artifacts()
        (base / 'train/last.pt').write_bytes(b'another model')
        self.assertFalse(p.verify_current(base))
        (base / 'train/last.pt').write_bytes(b'test checkpoint')
        path = base / 'evaluation/verification.json'
        v = p.read_json(path); v['counts']['unique_episode_keys'] -= 1
        p.write_json(path, v)
        self.assertFalse(p.verify_current(base))

    def test_publication_sources_are_static_and_content_locked(self):
        self.assertNotIn('*', ''.join(p.PUBLISH_SOURCES + p.PUBLISH_RESULTS))
        self.assertTrue(all(not path.startswith(('output/', 'dataset/'))
                            for path in p.PUBLISH_SOURCES + p.PUBLISH_RESULTS))
        (self.root / 'a.py').write_text('original')
        (self.root / 'ibe_map').mkdir()
        (self.root / 'ibe_map/verify_results.py').write_text('verifier')
        from types import SimpleNamespace
        args = SimpleNamespace(out=self.root)
        with patch.object(p, 'ROOT', self.root), patch.object(p, 'PUBLISH_SOURCES', ('a.py',)), \
             patch.object(p, 'git', return_value='base'):
            p.publication_identity(args)
            (self.root / 'a.py').write_text('unreviewed modification')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                p.publication_identity(args)

    def test_publication_retry_only_allows_its_own_unpushed_commit(self):
        from types import SimpleNamespace
        args = SimpleNamespace(out=self.root)
        self.assertEqual(p.publication_base(args, 'base', 'base'), 'base')
        with self.assertRaises(ValueError):
            p.publication_base(args, 'unrelated', 'base')
        p.write_json(self.root / 'publication_pending.json', {
            'commit': 'owned', 'base_commit': 'base', 'repository': p.REPOSITORY,
            'files': list(p.PUBLISH_SOURCES + p.PUBLISH_RESULTS)})
        with patch.object(p, 'git', return_value='base'):
            self.assertEqual(p.publication_base(args, 'owned', 'base'), 'base')
            with self.assertRaises(ValueError):
                p.publication_base(args, 'owned', 'remote-advanced')
        with patch.object(p, 'git', return_value='unexpected-parent'):
            with self.assertRaises(ValueError):
                p.publication_base(args, 'owned', 'base')


if __name__ == '__main__':
    unittest.main()
