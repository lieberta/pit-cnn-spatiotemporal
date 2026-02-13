import torch
import torch.nn as nn
import torch.nn.functional as F


class Laplacian3DLayer(nn.Module):
    def __init__(self, device):
        super(Laplacian3DLayer, self).__init__()
        self.laplace_kernel_3d = torch.tensor(
            [[[[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, -6, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]],
            dtype=torch.float64,
            requires_grad=False,
        ).to(device)

    def forward(self, x):
        return F.conv3d(x, self.laplace_kernel_3d, padding=1, groups=x.shape[1])


class HeatEquationLoss(nn.Module):
    def __init__(self, device, alpha=0.0257, delta_t=3.0, source_intensity=100000.0):
        super(HeatEquationLoss, self).__init__()
        self.alpha = alpha
        self.delta_t = delta_t
        self.laplacian_layer = Laplacian3DLayer(device)
        self.source_intensity = source_intensity

    def temporal_derivative(self, u_next, u_current):
        return (u_next - u_current) / self.delta_t

    def create_source_term(self, input_tensor):
        mask = input_tensor > 1000
        source_term = torch.zeros_like(input_tensor)
        source_term[mask] = self.source_intensity
        return source_term

    def forward(self, model_output, current_input):
        temporal_derivative = self.temporal_derivative(model_output, current_input)
        laplacian = self.laplacian_layer(current_input)
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
    def __init__(self, a, device, alpha=0.0257, source_intensity=100000.0):
        super(CombinedLoss_dynamic, self).__init__()
        self.alpha = alpha
        self.laplacian_layer = Laplacian3DLayer(device)
        self.source_intensity = source_intensity
        self.mse_loss = nn.MSELoss().to(device)
        self.a = a

    def temporal_derivative(self, input, t, output):
        t = t.view(t.size(0), t.size(1), 1, 1, 1).expand_as(output)
        return (output - input) / t

    def create_source_term(self, input_tensor):
        mask = input_tensor > 1000
        source_term = torch.zeros_like(input_tensor)
        source_term[mask] = self.source_intensity
        return source_term

    def forward(self, input, t, output, target):
        temporal_derivative = self.temporal_derivative(input, t, output)
        laplacian = self.laplacian_layer(input)
        source_term = self.create_source_term(input)
        p_loss = torch.mean((temporal_derivative - self.alpha * laplacian - source_term) ** 2)
        return self.mse_loss(output, target) + self.a * p_loss
