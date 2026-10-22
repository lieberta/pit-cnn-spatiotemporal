"""Independent zero-weight physics measurement and gradient calibration."""
import unittest

import torch

from training.loss import CombinedLoss_dynamic
from training.train_utils import loss_gradient_scales


class LossMeasurementTests(unittest.TestCase):
    def test_zero_weight_can_measure_physics_without_changing_objective(self):
        criterion = CombinedLoss_dynamic(
            lp_weight=0, mse_weight=1, device='cpu', alpha=0.001,
            min_temp=0, max_temp=10, dx=0.1, dy=0.2, dz=0.3, dt=0.1,
            ambient_temp=0, source_region_weight=1, physics_residual_mode='model_past')
        past = torch.zeros(1, 1, 4, 4, 4)
        output = torch.ones_like(past, requires_grad=True)
        total, data, physics = criterion.compute_components(
            past, output, past, past, past, past, past, past, measure_physics=True)
        self.assertAlmostEqual(physics.item(), 100)
        torch.testing.assert_close(total, data)
        total.backward()
        torch.testing.assert_close(output.grad, 2 * torch.ones_like(output) / output.numel())

    def test_gradient_probe_preserves_existing_gradients(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(2)
        model.weight.grad = torch.full_like(model.weight, 7)
        output = model(torch.tensor([[3.0]]))
        scales = loss_gradient_scales(model, (output-1).square().mean(),
                                      (output+1).square().mean(), 0.1)
        self.assertAlmostEqual(scales['data_gradient_norm'], 30)
        self.assertAlmostEqual(scales['objective_physics_gradient_norm'], 42)
        self.assertAlmostEqual(scales['weighted_physics_to_data_gradient'], 0.14)
        torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, 7))


if __name__ == '__main__':
    unittest.main()
