

class CombinedLoss_dynamic_precise_t_deviation(nn.Module):
    # physics enhanced loss for the dynamic method pecnn_dynamic
    def __init__(self, a, device, alpha = 0.0257, source_intensity=100000.0):
        super(CombinedLoss_dynamic, self).__init__()
        self.alpha = alpha
        self.laplacian_layer = Laplacian3DLayer(device)
        self.source_intensity = source_intensity

        #self.predicted_time = predicted_time predicted time muss aus dem current_input gezogen werden

        self.mse_loss = nn.MSELoss().to(device)
        #self.physics_loss = HeatEquationLoss(delta_t=predicted_time).to(
            #device)  # Assuming this is a custom class you've defined
        self.a = a

    def temporal_derivative(self, output, output_past, t, t_past):
        # expand the tensor with the time value (for a whole batch) to match output dimensions
        t = t.view(t.size(0), t.size(1), 1, 1, 1).expand_as(output)
        return (output - output_past) / (t-t_past)

    def create_source_term(self, input_tensor):
        # Create a mask of where input_tensor is greater than 1000
        mask = input_tensor > 1000

        # Initialize the source_term tensor with zeros of the same shape as input_tensor
        source_term = torch.zeros_like(input_tensor)

        # Apply source_intensity at positions where mask is True
        source_term[mask] = self.source_intensity

        return source_term

    def forward(self, input, output, output_past, t, t_past, target):

        # Calculate the temporal derivative
        temporal_derivative = self.temporal_derivative(output, output_past, t, t_past)

        # Calculate the Laplacian
        laplacian = self.laplacian_layer(input)

        # Create the source term
        source_term = self.create_source_term(input)

        # Compute the heat equation loss
        p_loss = torch.mean((temporal_derivative - self.alpha * laplacian - source_term) ** 2)

        # return the weighted combination of mse and physics_loss
        return self.mse_loss(output, target) + self.a * p_loss