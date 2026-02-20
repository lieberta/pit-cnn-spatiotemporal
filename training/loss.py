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
        # Normalization-aware source term:
        # Data is normalized as u_norm = (u - min_temp) / (max_temp - min_temp),
        # so the source term must be scaled by the same range R.
        #
        # For data/testset/normalization_values.json:
        # max_temp = 27373.34765625
        # min_temp = 20.0
        # R = max_temp - min_temp = 27353.34765625
        #
        # raw source_intensity = 100000.0
        # source_intensity_norm = 100000.0 / 27353.34765625 = 3.65585
        #
        # old fire threshold (raw): 1000.0
        # fire_threshold_norm = (1000.0 - 20.0) / 27353.34765625 = 0.035828
        #
        # This is a normalization correction because the loss is computed in normalized space.
        self.source_intensity = source_intensity / 27353.34765625
        self.fire_threshold = (1000.0 - 20.0) / 27353.34765625
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

    def forward(self, input, output, output_past, t, t_past, target):

            if self.a != 0:
                # Calculate the temporal derivative
                temporal_derivative = self.temporal_derivative(output, output_past, t, t_past)

                # Calculate the Laplacian
                laplacian = self.laplacian_layer(input)

                # Create the source term
                source_term = self.create_source_term(input)


                # Compute the heat equation loss (physics loss)
                p_loss = torch.mean((temporal_derivative - self.alpha * laplacian - source_term) ** 2)

                # Return the weighted combination of mse and physics_loss
                return self.mse_loss(output, target) + self.a * p_loss

            # if a=0 then the system shouldn't calculate physics loss
            else:
                # If self.a is 0, only return the MSE loss
                return self.mse_loss(output, target)