"""Check the shared residual against actual stored Euler trajectories."""
import json
import unittest
from pathlib import Path

import numpy as np
import torch
import zarr

from training.loss import Laplacian3DLayer, physics_residual


class StoredSimulationResidualTests(unittest.TestCase):
    def test_float64_steel_ground_truth_and_wrong_parameters(self):
        base=Path(__file__).resolve().parents[1]/'data/steel_train_medium'
        if not (base/'info.json').exists():
            self.skipTest('Local steel simulation dataset is not installed.')
        info=json.loads((base/'info.json').read_text())
        folder=sorted(base.glob('experiment_*'))[0]
        root=zarr.open_group(str(next(folder.glob('*.zarr'))),mode='r')
        scale=info['temp_range']
        def temperature(n):
            return torch.from_numpy((np.asarray(root['temperature'][n],dtype=np.float64)-info['min_temp'])/scale)[None,None]
        source=torch.from_numpy(np.asarray(root['source_field'],dtype=np.float64)/(info['rho']*info['cp']*scale))[None,None]
        # Build in float64: casting a float32 stencil afterwards retains its rounding error.
        previous_dtype=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            lap=Laplacian3DLayer(info['dx'],info['dy'],info['dz'],
                                 (info['ambient_temp']-info['min_temp'])/scale).double()
        finally:
            torch.set_default_dtype(previous_dtype)
        for n in (1,100,600,1200):
            now,past=temperature(n),temperature(n-1)
            r=physics_residual(now,past,info['dt'],info['alpha'],lap,source)
            self.assertLess(r.square().mean().item(),1e-24)
        self.assertGreater(physics_residual(now,past,info['dt'],info['alpha']*2,lap,source).square().mean().item(),1e-12)
        try:
            torch.set_default_dtype(torch.float64)
            wrong_lap = Laplacian3DLayer(info['dx']*2, info['dy']*2, info['dz']*2,
                                        (info['ambient_temp']-info['min_temp'])/scale)
        finally:
            torch.set_default_dtype(previous_dtype)
        self.assertGreater(physics_residual(now,past,info['dt'],info['alpha'],wrong_lap,source).square().mean().item(),1e-12)
        self.assertGreater(physics_residual(now,past,info['dt']*2,info['alpha'],lap,source).square().mean().item(),1e-12)
        self.assertGreater(physics_residual(now,past,info['dt'],info['alpha'],lap,torch.zeros_like(source)).square().mean().item(),1e-8)


if __name__=='__main__':
    unittest.main()
