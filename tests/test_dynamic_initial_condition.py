"""Initial-state anchoring and IC diagnostics across dynamic loss modes."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

from training.loss import CombinedLoss_dynamic
from training.train_pitcnn_dynamic import BaseModel_dynamic, _anchor_initial_state


class InitialConditionTests(unittest.TestCase):
    def test_mixed_steps_anchor_only_temperature_and_preserve_later_gradients(self):
        inputs = torch.zeros(3, 2, 4, 4, 4)
        inputs[:, 0] = 0.25
        inputs[:, 1] = 99.0
        prediction = torch.full((3, 1, 4, 4, 4), 0.75, requires_grad=True)
        result = _anchor_initial_state(inputs, prediction, torch.tensor([[1], [2], [1]]))
        torch.testing.assert_close(result[[0, 2]], inputs[[0, 2], :1])
        torch.testing.assert_close(result[1], prediction[1])
        result.sum().backward()
        torch.testing.assert_close(prediction.grad[[0, 2]], torch.zeros_like(prediction.grad[[0, 2]]))
        torch.testing.assert_close(prediction.grad[1], torch.ones_like(prediction.grad[1]))

    def test_real_trainer_checks_unanchored_zero_time_in_every_mode(self):
        # A deliberately wrong t=0 prediction must stay visible in diagnostics,
        # even when the physics residual's previous state is anchored to x0.
        class ToyDataset(Dataset):
            min_temp, max_temp = 0.0, 1.0
            dt, prediction_horizon = 0.1, 120.0
            rho = cp = 1.0
            alpha = 1e-5
            dx = dy = dz = 0.1
            ambient_temp = 0.25

            def __len__(self):
                return 5

            def __getitem__(self, index):
                temperature = torch.full((1, 4, 4, 4), 0.25)
                source = torch.zeros_like(temperature)
                model_input = torch.cat([temperature, torch.full_like(temperature, 99.0)])
                return (model_input, torch.tensor([1]), temperature, temperature,
                        source, source), temperature

        class ToyModel(BaseModel_dynamic):
            def __init__(self):
                super().__init__()
                self.offset = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, inputs, tau):
                return inputs[:, :1] + self.offset + tau.view(-1, 1, 1, 1, 1)

            def save_loss_plot(self, *args):
                pass

            def save_proc_time(self, *args):
                return 0.0

        def local_loader(*args, **kwargs):
            kwargs.update(num_workers=0, pin_memory=False, persistent_workers=False)
            return DataLoader(*args, **kwargs)

        def fake_split(dataset, *args):
            from torch.utils.data import Subset
            groups = {'train': [0, 1, 2], 'validation': [3], 'test': [4]}
            return ({k: Subset(dataset, v) for k,v in groups.items()},
                    dict(groups, metadata_sha256='toy', normalization={'min_temp':0,'max_temp':1,'temp_range':1}))

        original_components = CombinedLoss_dynamic.compute_components
        for mode, lp, mw in [('model_past', 1.0, 1.0), ('soft_physics', 1.0, 1.0),
                             ('gt_past', 1.0, 1.0), ('gt_future', 1.0, 1.0),
                             ('gt_future', 0.0, 1.0), ('gt_past', 1.0, 0.0)]:
            with self.subTest(mode=mode, lp=lp, mw=mw), tempfile.TemporaryDirectory() as tmp:
                calls = []

                def checked_components(criterion, **kwargs):
                    previous = kwargs['output_past']
                    if mode in {'model_past', 'soft_physics'}:
                        torch.testing.assert_close(previous, kwargs['input_t0'][:, :1])
                    else:
                        self.assertIsNone(previous)
                    calls.append(torch.is_grad_enabled())
                    return original_components(criterion, **kwargs)

                with patch('training.train_pitcnn_dynamic.prepare_experiment_split', side_effect=fake_split), \
                     patch('training.train_pitcnn_dynamic.DataLoader', side_effect=local_loader), \
                     patch('torch.cuda.is_available', return_value=False), \
                     patch.object(CombinedLoss_dynamic, 'compute_components', checked_components):
                    ToyModel().train_model(
                        loss_weight_schedule=[dict(epochs=1, lp_weight=lp, mse_weight=mw)],
                        dataset=ToyDataset(), num_epochs=1, batch_size=2, learning_rate=0.0,
                        model_name='toy', save_path=tmp, physics_residual_mode=mode, seed=42,
                    )
                checkpoint = torch.load(Path(tmp) / 'toy' / 'best.pth', map_location='cpu')
                self.assertEqual(checkpoint['seed'], 42)
                self.assertEqual(checkpoint['epoch'], 0)
                self.assertEqual(checkpoint['protocol']['split']['metadata_sha256'], 'toy')
                self.assertGreater(checkpoint['validation_mse'], 0)
                self.assertIn(True, calls)  # actual training loop
                self.assertIn(False, calls)  # actual validation loop
                with (Path(tmp) / 'toy' / 'initial_condition_metrics.csv').open() as f:
                    rows = list(csv.DictReader(f))
                with (Path(tmp) / 'toy' / 'loss_components.csv').open() as f:
                    components = next(csv.DictReader(f))
                self.assertGreater(float(components['val_mse_loss']), 0)
                self.assertGreater(float(components['val_physics_loss']), 0)
                self.assertEqual(len(rows), 1)
                self.assertAlmostEqual(float(rows[0]['val_ic_mse']), 0.25, places=6)
                self.assertAlmostEqual(float(rows[0]['val_ic_max_abs_error']), 0.5, places=6)


if __name__ == '__main__':
    unittest.main()
