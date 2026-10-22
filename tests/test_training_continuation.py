"""A mid-epoch checkpoint must reproduce uninterrupted stochastic training."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import Dataset, Subset

from training.train_pitcnn_dynamic import BaseModel_dynamic


class ToyData(Dataset):
    min_temp, max_temp, ambient_temp = 0., 1., 0.
    dt, prediction_horizon, alpha = .1, 1., .001
    dx = dy = dz = rho = cp = 1.

    def __len__(self): return 5

    def __getitem__(self, i):
        initial = torch.full((1, 4, 4, 4), .1 * i)
        zero = torch.zeros_like(initial)
        return (torch.cat([initial, zero]), torch.tensor([2]), initial,
                initial, zero, zero), initial + .2


class ToyModel(BaseModel_dynamic):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.3))
        self.dropout = torch.nn.Dropout(.2)

    def forward(self, x, tau):
        return x[:, :1] + self.weight * self.dropout(torch.ones_like(x[:, :1])) * tau.view(-1, 1, 1, 1, 1)

    def save_loss_plot(self, *args): pass
    def save_proc_time(self, *args): return 0.


def fake_split(dataset, *args):
    indices = dict(train=[0, 1, 2], validation=[3], test=[4])
    return ({k: Subset(dataset, v) for k, v in indices.items()},
            dict(indices, metadata_sha256='toy', normalization=dict(min_temp=0, max_temp=1, temp_range=1)))


class ContinuationTests(unittest.TestCase):
    def test_mid_epoch_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch('training.train_pitcnn_dynamic.prepare_experiment_split', side_effect=fake_split), \
             patch('torch.cuda.is_available', return_value=False):
            def train(name, resume=None):
                directory = Path(tmp, name)
                directory.mkdir(exist_ok=True)
                config = directory / 'config.json'
                if not config.exists(): config.write_text('{}')
                model = ToyModel()
                model.train_model([dict(epochs=2, lp_weight=.01, mse_weight=1.)],
                                  ToyData(), 2, 2, .001, name, tmp, seed=42,
                                  physics_residual_mode='model_past', loader_workers=0,
                                  resume_checkpoint_path=resume)
                return model
            reference = train('reference')
            with patch('training.train_pitcnn_dynamic.TrainingBudget.expired', side_effect=[False, True]):
                with self.assertRaises(SystemExit) as exit_info:
                    train('resumed')
            self.assertEqual(exit_info.exception.code, 75)
            checkpoint_path = str(Path(tmp, 'resumed', 'resumed.pth'))
            partial = torch.load(checkpoint_path, map_location='cpu')
            self.assertFalse(partial['epoch_complete'])
            self.assertEqual(partial['progress']['next_batch'], 1)
            self.assertEqual(partial['progress']['train_samples'], 2)
            resumed = train('resumed', checkpoint_path)
            torch.testing.assert_close(reference.weight, resumed.weight, rtol=0, atol=0)
            def rows(name):
                with Path(tmp, name, 'loss_components.csv').open() as stream:
                    return list(csv.DictReader(stream))
            expected, actual = rows('reference'), rows('resumed')
            self.assertEqual(len(actual), 2)
            for left, right in zip(expected, actual):
                for key in left:
                    if key != 'run_id': self.assertEqual(left[key], right[key])
            self.assertTrue(json.loads(Path(tmp, 'resumed', 'config.json').read_text())['training_complete'])
