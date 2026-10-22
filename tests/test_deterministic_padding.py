import unittest

import torch
from torch.nn import functional as F

from training.deterministic_padding import _ReflectionPad3d


class ReflectionTests(unittest.TestCase):
    def test_values_and_gradients_match_native_reflection(self):
        for padding in ((1, 1, 1, 1, 1, 1), (2, 1, 0, 2, 1, 0), (0, 0, 0, 0, 0, 0)):
            x = torch.randn(2, 3, 4, 5, 6, dtype=torch.float64, requires_grad=True)
            expected = F.pad(x, padding, mode='reflect')
            actual = _ReflectionPad3d.apply(x, padding)
            weights = torch.randn_like(actual)
            torch.testing.assert_close(actual, expected)
            native_grad = torch.autograd.grad(expected, x, weights)[0]
            custom_grad = torch.autograd.grad(actual, x, weights)[0]
            torch.testing.assert_close(custom_grad, native_grad)
