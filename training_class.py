import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os # to check if a plot already exists
import json

######################################
# physical contrains to make sure, that the temporal deviation and the spatial deviation has been
# calculated properly:

# This layer calculates the spatial deviation
class Laplacian3DLayer(nn.Module):
    def __init__(self):
        super(Laplacian3DLayer, self).__init__()
        # Define a 3D Laplacian kernel focusing on direct neighbors
        self.laplace_kernel_3d = torch.tensor([[[[[0, 0, 0],
                                                  [0, 1, 0],
                                                  [0, 0, 0]],
                                                 [[0, 1, 0],
                                                  [1, -6, 1],
                                                  [0, 1, 0]],
                                                 [[0, 0, 0],
                                                  [0, 1, 0],
                                                  [0, 0, 0]]]]],
                                               dtype=torch.float64, requires_grad=False).to('cuda')

    def forward(self, x):
        # Assuming x is of shape [batch, channel, depth, height, width]
        laplacian_3d = F.conv3d(x, self.laplace_kernel_3d, padding=1, groups=x.shape[1])
        return laplacian_3d

class HeatEquationLoss(nn.Module):
    def __init__(self, alpha = 0.0257, delta_t = 3., source_intensity=100000.0):
        super(HeatEquationLoss, self).__init__()
        self.alpha = alpha
        self.delta_t = delta_t
        self.laplacian_layer = Laplacian3DLayer()
        self.source_intensity = source_intensity

    def temporal_derivative(self, u_next, u_current):
        return (u_next - u_current) / self.delta_t

    def create_source_term(self, input_tensor):
        # Create a mask of where input_tensor is greater than 1000
        mask = input_tensor > 1000

        # Initialize the source_term tensor with zeros of the same shape as input_tensor
        source_term = torch.zeros_like(input_tensor)

        # Apply source_intensity at positions where mask is True
        source_term[mask] = self.source_intensity

        return source_term



    def forward(self, model_output, current_input):
        # Calculate the temporal derivative
        temporal_derivative = self.temporal_derivative(model_output, current_input)

        # Calculate the Laplacian
        laplacian = self.laplacian_layer(current_input)

        # Create the source term
        source_term = self.create_source_term(current_input)

        # Compute the heat equation loss
        loss = torch.mean((temporal_derivative - self.alpha * laplacian - source_term) ** 2)
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, a, predicted_time, device):
        super(CombinedLoss, self).__init__()
        self.predicted_time = predicted_time
        self.mse_loss = nn.MSELoss().to(device)
        self.physics_loss = HeatEquationLoss(delta_t= predicted_time).to(device)  # Assuming this is a custom class you've defined
        self.a = a
    def forward(self, current_input, output, target):
        return self.mse_loss(output, target) + self.a*self.physics_loss(current_input, output)

# Loss function
class Binarycross_physics(nn.Module):
    def __init__(self, a, predicted_time,device):
        super(Binarycross_physics).__init__()
        self.predicted_time = predicted_time
        self.physics_loss = HeatEquationLoss(delta_t= predicted_time).to(device)  # Assuming this is a custom class you've defined
        self.a = a
    def forward(self, current_input, output, target):
        # Calculate the Binary Cross Entropy loss across each dimension
        BCE = F.binary_cross_entropy(output, target, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE+KLD + self.a*self.physics_loss(current_input, output)


class BaseModel(nn.Module):
    def __init__(self, loss_fn):
        super(BaseModel, self).__init__()
        self.loss_fn = loss_fn

    def train_model(self, dataset, num_epochs, batch_size, learning_rate, model_name,save_path):
        # save time:
        tic = time.perf_counter()  # Start time

        # switch to graphic card
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        print("Device = " + device)
        self.to(device)
        self.double()

        #initialize lists to store losses
        train_losses = []
        val_losses = []

        # Datasets:
        shuffle = True
        pin_memory = True
        num_workers = 1

        train_set, val_set = torch.utils.data.random_split(dataset, [math.ceil(len(dataset) * 0.8),
                                                                        math.floor(len(dataset) * 0.2)])
        train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=pin_memory)
        val_loader = DataLoader(dataset=val_set, shuffle=shuffle, batch_size=batch_size,
                                num_workers=num_workers, pin_memory=pin_memory)


        criterion = self.loss_fn.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)


        for epoch in range(num_epochs):
            train_loss = 0.0
            val_loss = 0.0

            # Training loop
            self.train()
            loop = tqdm(train_loader, total=len(train_loader), leave=True)
            for i, (input, target) in enumerate(loop):
                # Forward pass, loss calculation, backward pass, optimization, etc.
                input = input.to(device)
                target = target.to(device)
                output = self(input.double())

                if isinstance(self.loss_fn, CombinedLoss):
                    loss = criterion(input, output, target) # here we have to change things for normal losses (delete input)
                else:
                    loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(trainloss=train_loss/(i+1))


            # calculate average train loss for this epoch
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # write train loss value in the bar
            #loop.set_postfix(trainloss=avg_train_loss)


            # Validation loop
            self.eval()

            # Create a tqdm progress bar for the validation loop

            with torch.no_grad():
                for ind, (input, target) in enumerate(val_loader):
                    input = input.to(device)
                    target = target.to(device)
                    output = self(input.double())
                    if isinstance(self.loss_fn, CombinedLoss):
                        loss = criterion(input, output, target)
                    else:
                        loss = criterion(output,target)

                    val_loss+=loss.item()

            # Calculate the average val losses and add to list
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Save the model after each epoch
            self.save_model(epoch, model_name,save_path)



        self.save_loss_plot(model_name, num_epochs, train_losses, val_losses,save_path)
        self.save_proc_time(model_name, tic,save_path)
        self.save_losses_data(model_name, num_epochs, train_losses, val_losses,save_path)



    def save_model(self, epoch, model_name,save_path):
        model_dir = os.path.join(save_path, model_name)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"epoch_{epoch}.pth")
        torch.save(self.state_dict(), model_path)
        model_path = os.path.join(model_dir, f"{model_name}.pth")
        torch.save(self.state_dict(), model_path) # save the endmodel to its name

    def save_loss_plot(self, model_name, num_epochs, train_losses, val_losses, save_path):
        # Create a list of epochs for the x-axis
        epochs = range(1, num_epochs + 1)

        # Create the losses plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses, label='Train Loss', color='blue') #  HTML color names are possible
        plt.plot(epochs, val_losses, label='Validation Loss', color='red')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        #plt.grid(True)

        # Save the losses plot in the same folder as the model
        model_dir = os.path.join(save_path, model_name)
        losses_plot_filename = os.path.join(model_dir, f"losses_plot_{model_name}.png")
        plt.savefig(losses_plot_filename)

        plt.show()

    def save_proc_time(self, model_name, start_time,save_path):
        # Calculate the training process time
        end_time = time.perf_counter()
        proc_time = end_time - start_time

        # Save the training process time to a text file
        model_dir = os.path.join(save_path, model_name)
        proc_time_filename = os.path.join(model_dir, f"proc_time_{model_name}.txt")

        # Format the time and save it to the file
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(proc_time))
        with open(proc_time_filename, "w") as file:
            file.write(f"Training process duration: {formatted_time}")

    def save_losses_data(self, model_name, epochs, train_losses, val_losses,save_path):
        # Create a dictionary to store losses data
        losses_data = {
            'epochs': epochs,
            'train_losses': train_losses,
            'val_losses': val_losses
        }


        # Save the losses data as a NumPy .npz file
        model_dir = os.path.join(save_path, model_name)
        losses_data_filename = os.path.join(model_dir, f"losses_data_{model_name}.npz")
        np.savez(losses_data_filename, **losses_data)

        # save the losses in .txt
        filename = os.path.join(model_dir,'losses_data.txt')

        # Save the dictionary to a text file in JSON format
        with open(filename, 'w') as file:
            json.dump(losses_data, file)



if __name__ == '__main__':
    model = BaseModel()
