"""Reflection padding with deterministic slice reductions in its backward pass."""
import torch
from torch import nn
from torch.nn import functional as F


class _ReflectionPad3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, padding):
        ctx.padding = tuple(padding)
        return F.pad(value, ctx.padding, mode='reflect')

    @staticmethod
    def backward(ctx, gradient):
        # Undo each axis separately. Reflection repeats input values, so add
        # their boundary contributions with ordered, elementwise operations
        # rather than the CUDA atomic scatter used by the native backward.
        for axis in range(-3, 0):
            offset = 2 * (-axis - 1)
            left, right = ctx.padding[offset:offset+2]
            size = gradient.shape[axis] - left - right
            result = gradient.narrow(axis, left, size).clone()
            if left:
                result.narrow(axis, 1, left).add_(gradient.narrow(axis, 0, left).flip((axis,)))
            if right:
                result.narrow(axis, size-right-1, right).add_(
                    gradient.narrow(axis, left+size, right).flip((axis,)))
            gradient = result
        return gradient, None


class ReflectionConv3d(nn.Conv3d):
    """Conv3d with unchanged parameters and deterministic reflection gradients."""
    def _conv_forward(self, value, weight, bias):
        if self.padding_mode != 'reflect':
            return super()._conv_forward(value, weight, bias)
        padded = _ReflectionPad3d.apply(value, self._reversed_padding_repeated_twice)
        return F.conv3d(padded, weight, bias, self.stride, 0, self.dilation, self.groups)
