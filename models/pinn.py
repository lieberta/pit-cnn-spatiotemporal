import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    Simple PINN backbone for the 3D heat equation.
    Input features are expected as [x, y, z, t] with shape (..., 4).
    Output is temperature with shape (..., 1).
    """

    def __init__(self, in_features=4, hidden_features=128, hidden_layers=6, out_features=1):
        super(PINN, self).__init__()
        layers = [nn.Linear(in_features, hidden_features), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_features, hidden_features), nn.Tanh()])
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.net(inputs)

    @staticmethod
    def _grad(outputs, inputs):
        return torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
        )[0]

    def heat_residual(self, coords, alpha=0.0257, source=None):
        """
        Computes residual r = u_t - alpha * (u_xx + u_yy + u_zz) - source.
        coords: tensor with last dimension [x, y, z, t], requires_grad=True.
        source: optional tensor broadcastable to u.
        """
        u = self.forward(coords)
        du = self._grad(u, coords)

        u_x = du[..., 0:1]
        u_y = du[..., 1:2]
        u_z = du[..., 2:3]
        u_t = du[..., 3:4]

        u_xx = self._grad(u_x, coords)[..., 0:1]
        u_yy = self._grad(u_y, coords)[..., 1:2]
        u_zz = self._grad(u_z, coords)[..., 2:3]

        laplace_u = u_xx + u_yy + u_zz
        if source is None:
            source = 0.0

        return u_t - alpha * laplace_u - source
