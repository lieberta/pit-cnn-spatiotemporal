"""Physical units, sample weighting and separate condition errors."""
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from evaluation.evaluate_physics import evaluate_experiments


class EvaluationTests(unittest.TestCase):
    def test_physical_units_and_batch_size_invariance(self):
        class Data(Dataset):
            files = [('experiment_a', 'store', n) for n in (1,2,3)]
            dt, temp_range, min_temp, ambient_temp = 0.1, 10.0, 0.0, 0.0
            alpha, dx, dy, dz = 0.001, 0.1, 0.1, 0.1
            def __len__(self): return 3
            def __getitem__(self, i):
                n = i + 1
                u0 = torch.zeros((1,4,4,4), dtype=torch.float64)
                source = torch.ones_like(u0)*0.25
                target = u0 + n*self.dt*0.25
                return (torch.cat([u0, source]), torch.tensor([n]),
                        target-0.025,target+0.025,source*0,source),target
        class Model(torch.nn.Module):
            def forward(self, x, tau):
                return x[:, :1] + 0.25*120*tau.view(-1,1,1,1,1)
        with patch('evaluation.evaluate_physics.load_experiment_metadata',return_value={'num_fires':3}):
            a,times=evaluate_experiments(Model(),Data(),['experiment_a'],120,batch_size=2,dtype=torch.float64)
            b,_=evaluate_experiments(Model(),Data(),['experiment_a'],120,batch_size=3,dtype=torch.float64)
        for key in ('mae_c','rmse_c','pde_rmse_k_per_s','ic_rmse_c'):
            self.assertLess(a[0][key],1e-12)
            self.assertAlmostEqual(a[0][key],b[0][key],places=12)
        # A spatially uniform heated field violates the fixed ambient boundary.
        self.assertAlmostEqual(a[0]['boundary_rmse_c'],((0.25**2+0.5**2+0.75**2)/3)**0.5)
        self.assertAlmostEqual(a[0]['boundary_max_abs_c'],0.75)
        self.assertEqual(len(times),3)
        self.assertAlmostEqual(times[-1]['time_seconds'],0.3)


if __name__=='__main__':
    unittest.main()
