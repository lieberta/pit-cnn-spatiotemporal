import torch
import torch.nn as nn
import torch.nn.functional as F



class Laplacian3DLayer(nn.Module):
    def __init__(self, device):
        super(Laplacian3DLayer, self).__init__()
        self.laplace_kernel_3d = torch.tensor(
            [[[[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, -6, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]],
            dtype=torch.float32,
            requires_grad=False,
        ).to(device)

    def forward(self, x):
        kernel = self.laplace_kernel_3d
        if kernel.dtype != x.dtype or kernel.device != x.device:
            kernel = kernel.to(dtype=x.dtype, device=x.device)
        return F.conv3d(x, kernel, padding=1, groups=x.shape[1])


class HeatEquationLoss(nn.Module):
    def __init__(
        self,
        device,
        alpha=0.0257,
        delta_t=3.0,
        source_intensity=100000.0,
        source_threshold=1000.0,
        min_temp=20.0,
        max_temp=27373.34765625,
    ):
        super(HeatEquationLoss, self).__init__()
        self.alpha = alpha
        self.delta_t = delta_t
        self.laplacian_layer = Laplacian3DLayer(device)
        temp_range = max_temp - min_temp
        self.source_intensity = source_intensity / temp_range
        self.source_threshold = (source_threshold - min_temp) / temp_range

    def temporal_derivative(self, u_next, u_current):
        return (u_next - u_current) / self.delta_t

    def create_source_term(self, input_tensor):
        mask = input_tensor > self.source_threshold
        source_term = torch.zeros_like(input_tensor)
        source_term[mask] = self.source_intensity
        return source_term

    def forward(self, model_output, current_input):
        temporal_derivative = self.temporal_derivative(model_output, current_input)
        laplacian = self.laplacian_layer(model_output)
        source_term = self.create_source_term(current_input)
        return torch.mean((temporal_derivative - self.alpha * laplacian - source_term) ** 2)


class CombinedLoss(nn.Module):
    def __init__(self, a, predicted_time, device):
        super(CombinedLoss, self).__init__()
        self.predicted_time = predicted_time
        self.mse_loss = nn.MSELoss().to(device)
        self.physics_loss = HeatEquationLoss(delta_t=predicted_time, device=device).to(device)
        self.a = a

    def forward(self, current_input, output, target):
        return self.mse_loss(output, target) + self.a * self.physics_loss(current_input, output)


class CombinedLoss_dynamic(nn.Module):
    def __init__(
        self,
        a,
        device,
        alpha=0.0257,
        source_intensity=100000.0,
        source_threshold=500.0,
        min_temp=20.0,
        max_temp=27373.34765625,
    ):
        super(CombinedLoss_dynamic, self).__init__()
        self.alpha = alpha
        self.laplacian_layer = Laplacian3DLayer(device)
        temp_range = max_temp - min_temp
        if temp_range <= 0:
            raise ValueError(f"Invalid normalization range: min_temp={min_temp}, max_temp={max_temp}")
        self.source_intensity = source_intensity / temp_range
        self.fire_threshold = (source_threshold - min_temp) / temp_range
        self.mse_loss = nn.MSELoss().to(device)
        self.a = a

    def temporal_derivative(self, output, output_past, t, t_past):
        # expand the tensor with the time value (for a whole batch) to match output dimensions
        t = t.view(t.size(0), t.size(1), 1, 1, 1).expand_as(output)
        t_past = t_past.view(t_past.size(0),t.size(1),1,1,1).expand_as(output_past)
        return (output - output_past) / (t-t_past)

    def create_source_term(self, input_tensor):
        mask = input_tensor > self.fire_threshold
        source_term = torch.zeros_like(input_tensor)
        source_term[mask] = self.source_intensity
        return source_term

    def compute_components(self, input, output, output_past, t, t_past, target):
        mse = self.mse_loss(output, target)
        if self.a == 0:
            physics = torch.zeros((), dtype=output.dtype, device=output.device)
        else:
            temporal_derivative = self.temporal_derivative(output, output_past, t, t_past)
            laplacian = self.laplacian_layer(output)
            source_term = self.create_source_term(input)
            physics = torch.mean((temporal_derivative - self.alpha * laplacian - source_term) ** 2)
        total = mse + self.a * physics
        return total, mse, physics

    def forward(self, input, output, output_past, t, t_past, target):
        total, _, _ = self.compute_components(input, output, output_past, t, t_past, target)
        return total
