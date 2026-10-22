import tempfile
import unittest
from unittest.mock import patch
from collections import OrderedDict

import numpy as np
import torch

from data import HeatEquationMultiDataset_dynamic


class MmapLoaderTests(unittest.TestCase):
    def test_staging_preserves_samples_and_avoids_repeated_zarr_reads(self):
        data = HeatEquationMultiDataset_dynamic.__new__(HeatEquationMultiDataset_dynamic)
        data.files = [('train_exp', 'train_store', 1), ('test_exp', 'test_store', 1)]
        data.data_cache, data.mmap_cache, data.cache_size = OrderedDict(), {}, 1
        data.min_temp, data.max_temp = 0., 100.
        payload = dict(temperature=np.arange(192, dtype=np.float64).reshape(3,4,4,4)/200,
                       T0=np.ones((4,4,4))*.2, source_mask=np.ones((4,4,4)),
                       source_field=np.ones((4,4,4))*.05)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(data, '_load_experiment_payload', return_value=payload) as load, \
             patch('data.load_experiment_metadata', return_value={'dt': .1}):
            expected = data[0]
            data.prepare_mmap_cache(tmp, {'train_exp'})
            self.assertEqual(set(data.mmap_cache), {'train_store'})
            self.assertEqual(load.call_count, 2)
            actual = data[0]
            data.data_cache.clear()
            data[0]
            self.assertEqual(load.call_count, 2)
            for left, right in zip(expected[0], actual[0]):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            torch.testing.assert_close(expected[1], actual[1], rtol=0, atol=0)
            self.assertIsInstance(data.data_cache['train_store']['temperature'], np.memmap)
