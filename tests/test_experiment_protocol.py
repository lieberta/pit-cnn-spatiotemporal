"""Disjoint splits, train-only normalization, metadata integrity and stable seeds."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict

import torch

from data import prepare_experiment_split
from training.train_utils import seed_everything, model_state_hash


class ExperimentProtocolTests(unittest.TestCase):
    def fixture(self, root, count=32):
        root = Path(root)
        (root/'info.json').write_text('{}')
        files = []
        for n in range(count):
            folder = root/f'experiment_{n:03d}'
            folder.mkdir()
            (folder/'metadata.json').write_text(json.dumps(dict(min_temp=n, max_temp=100+n, num_fires=n%10+1)))
            files.extend((str(folder), 'store', step) for step in (1,2,3))
        return SimpleNamespace(base_path=str(root), files=files, data_cache=OrderedDict())

    def test_disjoint_complete_reused_and_train_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self.fixture(tmp)
            subsets, manifest = prepare_experiment_split(d)
            self.assertEqual([len(manifest[k]) for k in ('train','validation','test')],[22,4,6])
            selected = [set(s.indices) for s in subsets.values()]
            self.assertFalse(selected[0]&selected[1] or selected[0]&selected[2] or selected[1]&selected[2])
            self.assertEqual(set.union(*selected),set(range(96)))
            expected = [int(n.split('_')[-1]) for n in manifest['train']]
            self.assertEqual(d.min_temp,min(expected))
            self.assertEqual(d.max_temp,100+max(expected))
            torch.manual_seed(9999)
            _, again = prepare_experiment_split(d)
            self.assertEqual(manifest,again)
            with self.assertRaisesRegex(ValueError,'seed/source-count'):
                prepare_experiment_split(d,seed=1)
            # Modifying held-out metadata must not silently refit an old split.
            heldout=Path(tmp)/manifest['test'][0]/'metadata.json'
            heldout.write_text(json.dumps(dict(min_temp=-999,max_temp=999,num_fires=2)))
            with self.assertRaisesRegex(ValueError,'metadata changed'):
                prepare_experiment_split(d)
            # An explicitly new manifest still fits only training temperatures.
            _, new = prepare_experiment_split(d,Path(tmp)/'new_split.json')
            self.assertEqual(new['normalization'],manifest['normalization'])

    def test_source_count_extrapolation_is_never_used_for_training_or_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=self.fixture(tmp,30)
            _,m=prepare_experiment_split(d,training_source_counts=[3,4,5,6,7])
            for name in m['train']+m['validation']:
                self.assertIn(int(name.split('_')[-1])%10+1,[3,4,5,6,7])
            self.assertTrue(any(int(n.split('_')[-1])%10+1 in [1,2,8,9,10] for n in m['test']))

    def test_same_seed_initializes_identically(self):
        seed_everything(42)
        a=torch.nn.Linear(3,2)
        torch.rand(100)
        seed_everything(42)
        b=torch.nn.Linear(3,2)
        self.assertEqual(model_state_hash(a),model_state_hash(b))


if __name__=='__main__':
    unittest.main()
