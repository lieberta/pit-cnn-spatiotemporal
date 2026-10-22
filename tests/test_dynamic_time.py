"""Regression checks for dataset indices and physical-time derivatives."""
import unittest

import numpy as np
import torch

from data import HeatEquationMultiDataset_dynamic
from training.loss import CombinedLoss_dynamic
from training.train_pitcnn_dynamic import _normalized_model_times


class DynamicTimeTests(unittest.TestCase):
    def test_adjacent_steps_use_normalized_times(self):
        for dtype in (torch.float32, torch.float64):
            for dt, horizon in ((0.1, 120.0), (0.01, 10.0), (0.001, 10.0)):
                with self.subTest(dtype=dtype, dt=dt, horizon=horizon):
                    last = round(horizon / dt)
                    steps = torch.tensor([[1], [last // 2], [last]], dtype=torch.long)
                    tau, past = _normalized_model_times(steps, dt, horizon, dtype)
                    self.assertEqual(past[0].item(), 0.0)
                    self.assertTrue(torch.all(past >= 0))
                    torch.testing.assert_close(tau[-1], torch.ones(1, dtype=dtype))
                    torch.testing.assert_close(
                        (tau - past) * horizon,
                        torch.full_like(tau, dt),
                        rtol=2e-3 if dtype == torch.float32 else 1e-10,
                        atol=1e-8 if dtype == torch.float32 else 1e-12,
                    )

    def test_loss_derivative_is_per_second_not_per_normalized_time(self):
        steps = torch.tensor([[1], [120], [1200]], dtype=torch.long)
        for horizon in (120.0, 240.0):
            with self.subTest(horizon=horizon):
                dt = 0.1
                tau, past = _normalized_model_times(steps, dt, horizon, torch.float64)
                # Analytic model u(t)=3t: changing H must not change du/dt=3.
                output = 3.0 * horizon * tau
                output_past = 3.0 * horizon * past
                loss = CombinedLoss_dynamic(
                    lp_weight=1.0, mse_weight=1.0, device='cpu',
                    alpha=1e-5, min_temp=0.0, max_temp=100.0,
                    dx=0.1, dy=0.1, dz=0.1, dt=dt, ambient_temp=20.0,
                    source_region_weight=1.0, physics_residual_mode='model_past',
                )
                torch.testing.assert_close(
                    loss.temporal_derivative(output, output_past),
                    torch.full_like(output, 3.0), rtol=1e-10, atol=1e-10,
                )

    def test_dataset_index_matches_ground_truth_neighbors(self):
        dataset = object.__new__(HeatEquationMultiDataset_dynamic)
        dataset.files = [('experiment', 'store', n) for n in (1, 3)]
        temperature = np.stack([np.full((2, 2, 2), n) for n in range(5)])
        payload = dict(temperature=temperature, T0=temperature[0],
                       source_mask=np.zeros((2, 2, 2)), source_field=np.zeros((2, 2, 2)))
        dataset._get_or_load_experiment_payload = lambda folder, store: payload
        for i, n in enumerate((1, 3)):
            inputs, target = dataset[i]
            _, step, past, future, _, _ = inputs
            self.assertEqual(step.dtype, torch.long)
            self.assertEqual(tuple(step.shape), (1,))
            self.assertEqual(step.item(), n)
            self.assertTrue(torch.all(past == n - 1))
            self.assertTrue(torch.all(target == n))
            self.assertTrue(torch.all(future == n + 1))


if __name__ == '__main__':
    unittest.main()
