import torch
import torch.nn as nn
import torch.nn.functional as F


class Laplacian3DLayer(nn.Module):
    """
    Fixed 3D Laplacian layer that mirrors the simulator discretization.

    The simulator uses explicit constant padding with the ambient temperature to
    enforce Dirichlet boundary conditions. The loss must do the same to avoid a
    train/simulator mismatch in the PDE residual.
    """

    def __init__(self, dx, dy, dz, ambient_temp_norm):
        super().__init__()
        self.ambient_temp_norm = float(ambient_temp_norm)

        kernel = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float32)
        kernel[0, 0, 1, 1, 1] = -2.0 * ((1.0 / dx ** 2) + (1.0 / dy ** 2) + (1.0 / dz ** 2))
        kernel[0, 0, 1, 1, 0] = kernel[0, 0, 1, 1, 2] = 1.0 / dz ** 2
        kernel[0, 0, 1, 0, 1] = kernel[0, 0, 1, 2, 1] = 1.0 / dy ** 2
        kernel[0, 0, 0, 1, 1] = kernel[0, 0, 2, 1, 1] = 1.0 / dx ** 2
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        """
        Apply the Laplacian to a tensor with shape (B, C, Nx, Ny, Nz).

        Constant ambient padding keeps the residual consistent with the simulator.
        """

        kernel = self.kernel
        if kernel.dtype != x.dtype or kernel.device != x.device:
            kernel = kernel.to(dtype=x.dtype, device=x.device)

        x_pad = F.pad(x, (1, 1, 1, 1, 1, 1), mode="constant", value=self.ambient_temp_norm)
        return F.conv3d(x_pad, kernel, padding=0, groups=1)


class HeatEquationLoss(nn.Module):
    """
    Static physics loss for one explicit Euler step.

    This path is kept for the static training code. All physical parameters must
    be provided explicitly so the loss never relies on hidden defaults.
    """

    def __init__(
        self,
        device,
        alpha=0.0257,
        delta_t=3.0,
        dx=6.3 / 63.0,
        dy=3.1 / 31.0,
        dz=1.5 / 15.0,
        ambient_temp_norm=0.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.delta_t = float(delta_t)
        self.laplacian_layer = Laplacian3DLayer(
            dx=dx,
            dy=dy,
            dz=dz,
            ambient_temp_norm=ambient_temp_norm,
        ).to(device)

    def temporal_derivative(self, u_next, u_current):
        """Compute the finite-difference temporal derivative for one Euler step."""

        return (u_next - u_current) / self.delta_t

    def forward(self, model_output, current_input, source_term):
        """Evaluate the PDE residual on the interior cells only."""

        temporal_derivative = self.temporal_derivative(model_output, current_input)
        laplacian = self.laplacian_layer(current_input)
        inner = (slice(None), slice(None), slice(1, -1), slice(1, -1), slice(1, -1))
        residual_inner = temporal_derivative[inner] - self.alpha * laplacian[inner] - source_term[inner]
        return torch.mean(residual_inner ** 2)


class CombinedLoss(nn.Module):
    """
    Static combined loss wrapper.

    The static path is not the main target of the current refactor, but it is kept
    compatible with the explicit-parameter philosophy used in the dynamic loss.
    """

    def __init__(
        self,
        a,
        predicted_time,
        device,
        alpha=0.0257,
        dx=6.3 / 63.0,
        dy=3.1 / 31.0,
        dz=1.5 / 15.0,
        ambient_temp_norm=0.0,
    ):
        super().__init__()
        self.predicted_time = float(predicted_time)
        self.mse_loss = nn.MSELoss().to(device)
        # The static training path still relies on the legacy constructor shape.
        # These defaults keep that path operational while the dynamic pipeline
        # uses explicit metadata-driven parameters throughout.
        self.physics_loss = HeatEquationLoss(
            device=device,
            alpha=alpha,
            delta_t=predicted_time,
            dx=dx,
            dy=dy,
            dz=dz,
            ambient_temp_norm=ambient_temp_norm,
        ).to(device)
        self.a = float(a)

    def forward(self, current_input, output, target, source_term):
        """Combine supervised MSE loss with the static physics residual."""

        return self.mse_loss(output, target) + self.a * self.physics_loss(output, current_input, source_term)


class CombinedLoss_dynamic(nn.Module):
    """
    Dynamic combined loss for time-conditioned training.

    The physics residual is evaluated as:
        dT_norm/dt - alpha * laplace(T_norm) - source_term_norm

    where source_term_norm is passed in explicitly from the dataloader. The loss
    no longer reconstructs heat sources from T0 because T0 and source placement
    are now stored as separate dataset fields.

    The residual mode controls which time-discretization variant is used:
    - model_past:
        (output_n - output_{n-1}) / dt - alpha * laplace(output_{n-1}) - source
    - gt_past:
        (output_n - target_{n-1}) / dt - alpha * laplace(target_{n-1}) - source
    - gt_future:
        (target_{n+1} - output_n) / dt - alpha * laplace(output_n) - source
    - soft_physics:
        (output_n - output_{n-1}) / dt - alpha * laplace(input_t0)

    soft_physics keeps the temporal derivative model-based, but anchors the
    diffusion term to the initial condition U0 instead of to a predicted or
    ground-truth neighboring state. The source term is omitted in this
    residual variant.
    """

    def __init__(
        self,
        lp_weight,
        mse_weight,
        device,
        alpha,
        min_temp,
        max_temp,
        dx,
        dy,
        dz,
        dt,
        ambient_temp,
        source_region_weight,
        physics_residual_mode,
    ):
        super().__init__()

        self.alpha = float(alpha)
        self.min_temp = float(min_temp)
        self.max_temp = float(max_temp)
        self.dt = float(dt)
        self.source_region_weight = float(source_region_weight)
        self.physics_residual_mode = str(physics_residual_mode)

        valid_modes = {"model_past", "gt_past", "gt_future", "soft_physics"}
        if self.physics_residual_mode not in valid_modes:
            raise ValueError(
                f"Invalid physics_residual_mode='{self.physics_residual_mode}'. "
                f"Expected one of {sorted(valid_modes)}."
            )

        temp_range = self.max_temp - self.min_temp
        if temp_range <= 0.0:
            raise ValueError(
                f"Invalid normalization range for loss setup: min_temp={self.min_temp}, max_temp={self.max_temp}"
            )

        # In normalized temperature space, the ambient value must also be normalized
        # so that padding matches the simulator's ambient Dirichlet boundary.
        ambient_temp_norm = (float(ambient_temp) - self.min_temp) / temp_range
        self.laplacian_layer = Laplacian3DLayer(
            dx=dx,
            dy=dy,
            dz=dz,
            ambient_temp_norm=ambient_temp_norm,
        ).to(device)

        self.mse_loss = nn.MSELoss().to(device)
        self.lp_weight = float(lp_weight)
        self.mse_weight = float(mse_weight)

    def temporal_derivative(self, output, output_past):
        """Use the dataset dt directly for the temporal finite difference."""

        return (output - output_past) / self.dt

    def _build_physics_terms(self, input_t0, output, output_past, target_past, target_future):
        """
        Build the temporal derivative and Laplacian terms for the selected mode.

        The supported residual variants are implemented centrally here so the
        rest of the loss stays mode-agnostic and easy to compare across runs.
        """

        if self.physics_residual_mode == "model_past":
            if output_past is None:
                raise ValueError("output_past must be provided when physics_residual_mode='model_past'.")
            temporal_derivative = self.temporal_derivative(output, output_past)
            laplacian_term = self.laplacian_layer(output_past)
            return temporal_derivative, laplacian_term

        if self.physics_residual_mode == "gt_past":
            temporal_derivative = self.temporal_derivative(output, target_past)
            laplacian_term = self.laplacian_layer(target_past)
            return temporal_derivative, laplacian_term

        if self.physics_residual_mode == "soft_physics":
            if output_past is None:
                raise ValueError("output_past must be provided when physics_residual_mode='soft_physics'.")
            # Keep the temporal term model-based, but diffuse from the initial
            # condition U0 to obtain a softer PDE anchor.
            temporal_derivative = self.temporal_derivative(output, output_past)
            laplacian_term = self.laplacian_layer(input_t0)
            return temporal_derivative, laplacian_term

        # In the gt_future variant, the model output takes the role of the
        # "current" state in the Laplacian while the future ground-truth frame
        # provides the forward temporal-difference anchor.
        temporal_derivative = (target_future - output) / self.dt
        laplacian_term = self.laplacian_layer(output)
        return temporal_derivative, laplacian_term

    def compute_components(
        self,
        input_t0,
        output,
        output_past,
        target_past,
        target,
        target_future,
        source_mask,
        source_field,
    ):
        """
        Compute the total, MSE, and physics loss components.

        input_t0 is the normalized initial temperature field U0. The
        soft_physics mode uses it directly in the Laplacian term.
        """

        if self.mse_weight == 0.0:
            mse = torch.zeros((), dtype=output.dtype, device=output.device)
        else:
            mse = self.mse_loss(output, target)

        if self.lp_weight == 0.0:
            physics = torch.zeros((), dtype=output.dtype, device=output.device)
        else:
            temporal_derivative, laplacian = self._build_physics_terms(
                input_t0=input_t0,
                output=output,
                output_past=output_past,
                target_past=target_past,
                target_future=target_future,
            )

            # Enforce the PDE only on interior cells because the outer faces are
            # fixed by Dirichlet boundary conditions in both simulation and loss.
            inner = (slice(None), slice(None), slice(1, -1), slice(1, -1), slice(1, -1))
            residual_inner = temporal_derivative[inner] - self.alpha * laplacian[inner]
            if self.physics_residual_mode != "soft_physics":
                residual_inner = residual_inner - source_field[inner]

            # The source mask is not required for the PDE itself, but it can be
            # used to up-weight residuals inside source cells if desired.
            if self.source_region_weight != 1.0:
                weight_inner = torch.ones_like(source_mask[inner], dtype=output.dtype, device=output.device)
                weight_inner = weight_inner + (self.source_region_weight - 1.0) * source_mask[inner]
                physics = torch.mean(weight_inner * (residual_inner ** 2))
            else:
                physics = torch.mean(residual_inner ** 2)

        total = self.mse_weight * mse + self.lp_weight * physics
        return total, mse, physics

    def forward(
        self,
        input_t0,
        output,
        output_past,
        target_past,
        target,
        target_future,
        source_mask,
        source_field,
    ):
        """Expose the same combined loss through the module forward method."""

        total, _, _ = self.compute_components(
            input_t0=input_t0,
            output=output,
            output_past=output_past,
            target_past=target_past,
            target=target,
            target_future=target_future,
            source_mask=source_mask,
            source_field=source_field,
        )
        return total
