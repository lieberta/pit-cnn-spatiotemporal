import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary    # for summarizing the model
import math
from training_class import BaseModel

class EncoderBlock_cross(nn.Module):
    # one 3D convolutional block with batchnorm, relu activation and two convolution
    # one convolution extends the output channels and one keeps the channels
    def __init__(self, in_c, out_c):
        super(EncoderBlock_cross, self).__init__()

        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, padding_mode='reflect')
        self.gn1 = nn.GroupNorm(num_groups=int(out_c/8), num_channels= out_c) #nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, padding_mode='reflect')
        self.gn2 = nn.GroupNorm(num_groups=int(out_c/8), num_channels=out_c)
        self.avgpool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.gelu(self.gn1(self.conv1(x))) # gaussian linear unit for non-linearity
        y = F.gelu(self.gn2(self.conv2(x)))
        x = self.avgpool(y)

        return x, y # y is the output for the crossconnection, x is the output dim reduced output
class DecoderBlock_transposedcnn_cross(nn.Module):
    # one 3D convolutional block with batchnorm, relu activation and two convolution
    # one convolution extends the output channels and one keeps the channels
    def __init__(self, in_c, cross_c,out_c,dropout = 0.0):
        super(DecoderBlock_transposedcnn_cross, self).__init__()

        self.up = nn.ConvTranspose3d(in_c, in_c, kernel_size= 2,stride=2, output_padding = 0)
        self.conv1 = nn.Conv3d(in_c+cross_c, out_c, kernel_size=3, padding=1) # here additional input_channels for cross connections can be added
        self.gn1 = nn.GroupNorm(num_groups=int(out_c/8), num_channels= out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=int(out_c/8), num_channels= out_c)

    def forward(self,x,x_cross):
        x = self.up(x)
        x = torch.cat((x,x_cross), dim = -4)
        x = F.gelu(self.gn1(self.conv1(x)))
        x = F.gelu(self.gn2(self.conv2(x)))
        return x
class PICNN_static(BaseModel):
    # this is a new version of CNN3D1D with cross connections, but only 2 blocks deep
    # maxpooling instead of step size and additional conv layers in each block
    # cross means it has crossconnections between layers
    def __init__(self, loss_fn,channels=8):
        super(PICNN_static, self).__init__(loss_fn=loss_fn)
        # initialize the latent space dimensions:

        # Encoder
        self.encoder1 = EncoderBlock_cross(in_c = 1, out_c = channels)
        self.encoder2 = EncoderBlock_cross(in_c = channels, out_c = channels*2)
        self.encoder3 = EncoderBlock_cross(in_c= channels*2, out_c=channels*4)

        # Middle:
        self.conv1 = nn.Conv3d(channels*4, channels*8, kernel_size=3, padding=1, padding_mode='reflect')
        self.gn1 = nn.GroupNorm(num_groups=int(channels), num_channels= channels*8) #nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(channels*8, channels*8, kernel_size=3, padding=1, padding_mode='reflect')
        self.gn2 = nn.GroupNorm(num_groups=int(channels), num_channels=channels*8)

        # Decoder
        self.dblock1=DecoderBlock_transposedcnn_cross(channels*8, channels*4,channels*4)    # additional 16 channels for the crossconnection
        self.dblock2=DecoderBlock_transposedcnn_cross(channels*4,channels*2,channels*2)    # additional 8 channels for the crossconnection
        self.dblock3=DecoderBlock_transposedcnn_cross(channels*2,channels,channels)

        # Output
        self.con_end = nn.Conv3d(channels,1,kernel_size=1, padding =0, padding_mode='reflect')



    def forward(self, x):
        x_input = x
        # Spatial Encoding:
        x, x_cross1 = self.encoder1(x)  # x_cross1 shape: [batch, 4, 32,64,26]
        x, x_cross2 = self.encoder2(x)  # x_cross2 shape [batch, 8, 32,64,26]
        x, x_cross3 = self.encoder3(x)  # x_cross3 shape [batch, 16, 32,64,26]

        # Middle forward:
        x = F.gelu(self.gn2(self.conv2(F.gelu(self.gn1(self.conv1(x))))))

        # Decoder:
        x = self.dblock1(x,x_cross3)
        x = self.dblock2(x,x_cross2)
        x = self.dblock3(x,x_cross1)

        # Output:
        x= self.con_end(x)



        # impose dirichlet bc as a padding

        # Set the front and back slices
        x[:, :, 0, :, :] = x_input[:, :, 0, :, :]
        x[:, :, -1, :, :] = x_input[:, :, -1, :, :]

        # Set the top and bottom slices
        x[:, :, :, 0, :] = x_input[:, :, :, 0, :]
        x[:, :, :, -1, :] = x_input[:, :, :, -1, :]

        # Set the left and right slices
        x[:, :, :, :, 0] = x_input[:, :, :, :, 0]
        x[:, :, :, :, -1] = x_input[:, :, :, :, -1]
        return x

class PICNN_VAR(BaseModel):
    # this is a new version of CNN3D1D with cross connections, but only 2 blocks deep
    # maxpooling instead of step size and additional conv layers in each block
    # cross means it has crossconnections between layers
    def __init__(self, loss_fn):
        super(PICNN_static, self).__init__(loss_fn=loss_fn)
        # initialize the latent space dimensions:

        # Encoder
        self.encoder1 = EncoderBlock_cross(in_c = 1, out_c = 8)
        self.encoder2 = EncoderBlock_cross(in_c = 8, out_c = 16)
        self.encoder31 = EncoderBlock_cross(in_c=16, out_c=32)
        self.encoder32 = EncoderBlock_cross(in_c=16, out_c=32)

        # Middle mu:
        self.conv1mu = nn.Conv3d(32, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.gn1mu = nn.GroupNorm(num_groups=int(64/8), num_channels= 64) #nn.BatchNorm3d(out_c)
        self.conv2mu = nn.Conv3d(64, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.gn2mu = nn.GroupNorm(num_groups=int(64/8), num_channels=64)

        # Middle logvar:
        self.conv1logvar = nn.Conv3d(32, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.gn1logvar = nn.GroupNorm(num_groups=int(64/8), num_channels= 64) #nn.BatchNorm3d(out_c)
        self.conv2logvar = nn.Conv3d(64, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.gn2logvar = nn.GroupNorm(num_groups=int(64/8), num_channels=64)

        # Decoder
        self.dblock1=DecoderBlock_transposedcnn_cross(64, 32,32)    # additional 16 channels for the crossconnection
        self.dblock2=DecoderBlock_transposedcnn_cross(32,16,16)    # additional 8 channels for the crossconnection
        self.dblock3=DecoderBlock_transposedcnn_cross(16,8,8)

        # Output
        self.con_end = nn.Conv3d(8,1,kernel_size=1, padding =0, padding_mode='reflect')

    def middleblock(self,x):
        x = self.conv1mu(self.gn1mu(self.conv2mu(self.gn2mu(x))))
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def forward(self, x):
        x_input = x
        # Spatial Encoding:
        x, x_cross1 = self.encoder1(x)  # x_cross1 shape: [batch, 4, 32,64,26]
        x, x_cross2 = self.encoder2(x)  # x_cross2 shape [batch, 8, 32,64,26]
        x, x_cross3 = self.encoder3(x)  # x_cross3 shape [batch, 16, 32,64,26]



        # Middle Variational:
        mu = self.conv1mu(self.gn1mu(self.conv2mu(self.gn2mu(x))))
        logvar = self.conv1logvar(self.gn1logvar(self.conv2logvar(self.gn2logvar(x))))

        x = self.reparameterize(mu,logvar)

        # Decoder:
        x = self.dblock1(x,x_cross3)
        x = self.dblock2(x,x_cross2)
        x = self.dblock3(x,x_cross1)

        # Output:
        x = self.con_end(x)



        # impose dirichlet bc as a padding

        # Set the front and back slices
        x[:, :, 0, :, :] = x_input[:, :, 0, :, :]
        x[:, :, -1, :, :] = x_input[:, :, -1, :, :]

        # Set the top and bottom slices
        x[:, :, :, 0, :] = x_input[:, :, :, 0, :]
        x[:, :, :, -1, :] = x_input[:, :, :, -1, :]

        # Set the left and right slices
        x[:, :, :, :, 0] = x_input[:, :, :, :, 0]
        x[:, :, :, :, -1] = x_input[:, :, :, :, -1]
        return x, mu, logvar



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PICNN_static(loss_fn = torch.nn.MSELoss()).to(device)
    #summary(model, (1, 1, 32, 64, 16))
    x = torch.arange(8*1*32*64*16).reshape(8, 1, 64, 32, 16).to(device)
    y = model(x.float())
    print(y.shape)




