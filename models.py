import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary    # for summarizing the model
import math
from training_class import BaseModel, BaseModel_dynamic

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

class PECNN_dynamic(BaseModel_dynamic):
    # This is a Encoder-Decoder U-Net structured physics enhanced conv network with outputpadding for
    # boundary conditions in the data and cross connections between encoder blocks and decoder blocks
    # latent room in middle part enhanced with one layer of timestep

    # 3 Encoder Blocks with: 3dconv / groupnormalization / 3dconv / groupnorm / avgpool
    # Middle: 3dconv / groupnorm / 3dconv / groupnorm
    # 3 Decoder Blocks with: tconv / groupnorm / tconv / groupnorm
    # groupnumber in group normalization increases with number of outputchannels
    def __init__(self, loss_fn, c = 8,):
        super(PECNN_dynamic, self).__init__(loss_fn=loss_fn)

        # conv<i><j> with i represents blocknumber and j represents the index inside of the block
        # one avgpool initilization for all blocks (no trainable parameter)

        # Encoderblock 1
        self.conv11 = nn.Conv3d(1, c, kernel_size=3, padding=1, padding_mode='reflect')
        self.gn11 = nn.GroupNorm(num_groups=int(c / 8), num_channels=c)
        self.conv12 = nn.Conv3d(c, c, kernel_size=3, padding=1, padding_mode='reflect')
        self.gn12 = nn.GroupNorm(num_groups=int(c / 8), num_channels=c)
        self.avgpool = nn.AvgPool3d(kernel_size=2, stride=2)

        # Encoderblock 2
        self.conv21 = nn.Conv3d(c, c*2, kernel_size=3, padding=1, padding_mode='reflect')
        self.gn21 = nn.GroupNorm(num_groups=int(c*2 / 8), num_channels=c*2)
        self.conv22 = nn.Conv3d(c*2, c*2, kernel_size=3, padding=1, padding_mode='reflect')
        self.gn22 = nn.GroupNorm(num_groups=int(c*2 / 8), num_channels=c*2)
        # avgpool in forward

        # Encoderblock 3
        self.conv31 = nn.Conv3d(c*2, c*4, kernel_size=3, padding=1, padding_mode='reflect')
        self.gn31 = nn.GroupNorm(num_groups=int(c*4 / 8), num_channels=c*4)
        self.conv32 = nn.Conv3d(c*4, c*4, kernel_size=3, padding=1, padding_mode='reflect')
        self.gn32 = nn.GroupNorm(num_groups=int(c*4 / 8), num_channels=c*4)
        # avgpool in forward

        # Middle Block
        self.conv_m1 = nn.Conv3d(c*4+1, c*8, kernel_size=3, padding=1, padding_mode='reflect')  # here +1 inputlayer filled with the current timestep
        self.gn_m1 = nn.GroupNorm(num_groups=int(c), num_channels= c*8)                           # groups = int(c) because it would be c*8/8
        self.conv_m2 = nn.Conv3d(c*8, c*8, kernel_size=3, padding=1, padding_mode='reflect')
        self.gn_m2 = nn.GroupNorm(num_groups=int(c), num_channels=c*8)

        # Decoderblock 1
        self.up1 = nn.ConvTranspose3d(c*8, c*8, kernel_size= 2,stride=2, output_padding = 0)
        self.conv41 = nn.Conv3d(c*8 + c*4, c*4, kernel_size=3, padding=1)                       # here additional input_channels for cross connections can be added
        self.gn41 = nn.GroupNorm(num_groups=int(c*4/8), num_channels= c*4)
        self.conv42 = nn.Conv3d(c*4, c*4, kernel_size=3, padding=1)
        self.gn42 = nn.GroupNorm(num_groups=int(c*4/8), num_channels= c*4)

        # Decoderblock 2
        self.up2 = nn.ConvTranspose3d(c*4, c*4, kernel_size= 2,stride=2, output_padding = 0)
        self.conv51 = nn.Conv3d(c*4+c*2, c*2, kernel_size=3, padding=1)                         # here additional input_channels for cross connections can be added
        self.gn51 = nn.GroupNorm(num_groups=int(c*2/8), num_channels= c*2)
        self.conv52 = nn.Conv3d(c*2, c*2, kernel_size=3, padding=1)
        self.gn52 = nn.GroupNorm(num_groups=int(c*2/8), num_channels= c*2)

        # Decoderblock 3
        self.up3 = nn.ConvTranspose3d(c*2, c*2, kernel_size= 2,stride=2, output_padding = 0)
        self.conv61 = nn.Conv3d(c*2+c, c, kernel_size=3, padding=1)                             # here additional input_channels for cross connections can be added
        self.gn61 = nn.GroupNorm(num_groups=int(c/8), num_channels= c)
        self.conv62 = nn.Conv3d(c, c, kernel_size=3, padding=1)
        self.gn62 = nn.GroupNorm(num_groups=int(c/8), num_channels= c)

        # Output
        self.conv_end = nn.Conv3d(c,1,kernel_size=1, padding =0, padding_mode='reflect')

    def forward(self,x_input, time):

        x = x_input                                     # we need initial x_input for boundary conditions later

        # Encoder Block 1
        x = F.gelu(self.gn11(self.conv11(x)))           # gaussian linear unit (gelu) for non-linearity
        x_cross1 = F.gelu(self.gn12(self.conv12(x)))    # save x_cross<i> for cross connection
        x = self.avgpool(x_cross1)

        # Encoder Block 2
        x = F.gelu(self.gn21(self.conv21(x)))
        x_cross2 = F.gelu(self.gn22(self.conv22(x)))
        x = self.avgpool(x_cross2)

        # Encoder Block 3
        x = F.gelu(self.gn31(self.conv31(x)))
        x_cross3 = F.gelu(self.gn32(self.conv32(x)))
        x = self.avgpool(x_cross3)

        # Middle forward:
        # create additional channel with values filled with time_value
        batch, channels, x_dim, y_dim, z_dim = x.shape

        # old method with error, because time is a tensor of size [batch,1] and not [1], delete if the
        # time_channel = torch.full((batch, 1, x_dim, y_dim, z_dim), time.item(), device=x.device, dtype=x.dtype)

        # Ensure time is reshaped to the correct shape
        time = time.view(batch, 1, 1, 1, 1)
        time_channel = time.expand(batch, 1, x_dim, y_dim, z_dim)

        # concat time_channel
        x = torch.cat((x, time_channel), dim=-4)
        x = F.gelu(self.gn_m1(self.conv_m1(x)))
        x = F.gelu(self.gn_m2(self.conv_m2(x)))

        # Decoder Block 1
        x = self.up1(x)
        x = torch.cat((x,x_cross3), dim = -4)
        x = F.gelu(self.gn41(self.conv41(x)))
        x = F.gelu(self.gn42(self.conv42(x)))

        # Decoder Block 2
        x = self.up2(x)
        x = torch.cat((x,x_cross2), dim = -4)
        x = F.gelu(self.gn51(self.conv51(x)))
        x = F.gelu(self.gn52(self.conv52(x)))

        # Decoder Block 3
        x = self.up3(x)
        x = torch.cat((x,x_cross1), dim = -4)
        x = F.gelu(self.gn61(self.conv61(x)))
        x = F.gelu(self.gn62(self.conv62(x)))

        # Output:
        x= self.conv_end(x)

        # Boundary Condition Padding:
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


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PECNN_dynamic(loss_fn = torch.nn.MSELoss()).to(device)
    #summary(model, (1, 1, 32, 64, 16))
    x = torch.arange(8*1*32*64*16).reshape(8, 1, 64, 32, 16).to(device)
    t = .1
    y = model(x.float(),t)
    print(y.shape)




