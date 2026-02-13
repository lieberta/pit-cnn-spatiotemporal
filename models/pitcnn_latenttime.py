import torch
import torch.nn as nn
import torch.nn.functional as F

from training.train_pitcnn_dynamic import BaseModel_dynamic


class PITCNN_dynamic(BaseModel_dynamic):
    def __init__(self, c=8):
        super(PITCNN_dynamic, self).__init__()
        self.c = c

        self.conv11 = nn.Conv3d(1, c, kernel_size=3, padding=1, padding_mode="reflect")
        self.gn11 = nn.GroupNorm(num_groups=int(c / 8), num_channels=c)
        self.conv12 = nn.Conv3d(c, c, kernel_size=3, padding=1, padding_mode="reflect")
        self.gn12 = nn.GroupNorm(num_groups=int(c / 8), num_channels=c)
        self.avgpool = nn.AvgPool3d(kernel_size=2, stride=2)

        self.conv21 = nn.Conv3d(c, c * 2, kernel_size=3, padding=1, padding_mode="reflect")
        self.gn21 = nn.GroupNorm(num_groups=int(c * 2 / 8), num_channels=c * 2)
        self.conv22 = nn.Conv3d(c * 2, c * 2, kernel_size=3, padding=1, padding_mode="reflect")
        self.gn22 = nn.GroupNorm(num_groups=int(c * 2 / 8), num_channels=c * 2)

        self.conv31 = nn.Conv3d(c * 2, c * 4, kernel_size=3, padding=1, padding_mode="reflect")
        self.gn31 = nn.GroupNorm(num_groups=int(c * 4 / 8), num_channels=c * 4)
        self.conv32 = nn.Conv3d(c * 4, c * 4, kernel_size=3, padding=1, padding_mode="reflect")
        self.gn32 = nn.GroupNorm(num_groups=int(c * 4 / 8), num_channels=c * 4)

        self.conv_m1 = nn.Conv3d(c * 8, c * 8, kernel_size=3, padding=1, padding_mode="reflect")
        self.gn_m1 = nn.GroupNorm(num_groups=int(c), num_channels=c * 8)
        self.conv_m2 = nn.Conv3d(c * 8, c * 8, kernel_size=3, padding=1, padding_mode="reflect")
        self.gn_m2 = nn.GroupNorm(num_groups=int(c), num_channels=c * 8)

        self.up1 = nn.ConvTranspose3d(c * 8, c * 8, kernel_size=2, stride=2, output_padding=0)
        self.conv41 = nn.Conv3d(c * 8 + c * 4, c * 4, kernel_size=3, padding=1)
        self.gn41 = nn.GroupNorm(num_groups=int(c * 4 / 8), num_channels=c * 4)
        self.conv42 = nn.Conv3d(c * 4, c * 4, kernel_size=3, padding=1)
        self.gn42 = nn.GroupNorm(num_groups=int(c * 4 / 8), num_channels=c * 4)

        self.up2 = nn.ConvTranspose3d(c * 4, c * 4, kernel_size=2, stride=2, output_padding=0)
        self.conv51 = nn.Conv3d(c * 4 + c * 2, c * 2, kernel_size=3, padding=1)
        self.gn51 = nn.GroupNorm(num_groups=int(c * 2 / 8), num_channels=c * 2)
        self.conv52 = nn.Conv3d(c * 2, c * 2, kernel_size=3, padding=1)
        self.gn52 = nn.GroupNorm(num_groups=int(c * 2 / 8), num_channels=c * 2)

        self.up3 = nn.ConvTranspose3d(c * 2, c * 2, kernel_size=2, stride=2, output_padding=0)
        self.conv61 = nn.Conv3d(c * 2 + c, c, kernel_size=3, padding=1)
        self.gn61 = nn.GroupNorm(num_groups=int(c / 8), num_channels=c)
        self.conv62 = nn.Conv3d(c, c, kernel_size=3, padding=1)
        self.gn62 = nn.GroupNorm(num_groups=int(c / 8), num_channels=c)

        self.conv_end = nn.Conv3d(c, 1, kernel_size=1, padding=0, padding_mode="reflect")

    def forward(self, x_input, time):
        x = x_input

        x = F.gelu(self.gn11(self.conv11(x)))
        x_cross1 = F.gelu(self.gn12(self.conv12(x)))
        x = self.avgpool(x_cross1)

        x = F.gelu(self.gn21(self.conv21(x)))
        x_cross2 = F.gelu(self.gn22(self.conv22(x)))
        x = self.avgpool(x_cross2)

        x = F.gelu(self.gn31(self.conv31(x)))
        x_cross3 = F.gelu(self.gn32(self.conv32(x)))
        x = self.avgpool(x_cross3)

        batch, _, x_dim, y_dim, z_dim = x.shape
        time = time.view(batch, 1, 1, 1, 1)
        time_channel = time.expand(batch, self.c * 4, x_dim, y_dim, z_dim)

        x = torch.cat((x, time_channel), dim=-4)
        x = F.gelu(self.gn_m1(self.conv_m1(x)))
        x = F.gelu(self.gn_m2(self.conv_m2(x)))

        x = self.up1(x)
        x = torch.cat((x, x_cross3), dim=-4)
        x = F.gelu(self.gn41(self.conv41(x)))
        x = F.gelu(self.gn42(self.conv42(x)))

        x = self.up2(x)
        x = torch.cat((x, x_cross2), dim=-4)
        x = F.gelu(self.gn51(self.conv51(x)))
        x = F.gelu(self.gn52(self.conv52(x)))

        x = self.up3(x)
        x = torch.cat((x, x_cross1), dim=-4)
        x = F.gelu(self.gn61(self.conv61(x)))
        x = F.gelu(self.gn62(self.conv62(x)))

        x = self.conv_end(x)

        x[:, :, 0, :, :] = x_input[:, :, 0, :, :]
        x[:, :, -1, :, :] = x_input[:, :, -1, :, :]
        x[:, :, :, 0, :] = x_input[:, :, :, 0, :]
        x[:, :, :, -1, :] = x_input[:, :, :, -1, :]
        x[:, :, :, :, 0] = x_input[0, 0, -1, -1, -1]
        x[:, :, :, :, -1] = x_input[:, :, :, :, -1]
        return x


class PITCNN_dynamic_latenttime1(PITCNN_dynamic):
    def __init__(self, c=8):
        super().__init__(c=c)
        self.conv_m1 = nn.Conv3d(self.c * 4 + 1, self.c * 8, kernel_size=3, padding=1, padding_mode="reflect")

    def forward(self, x_input, time):
        x = x_input

        x = F.gelu(self.gn11(self.conv11(x)))
        x_cross1 = F.gelu(self.gn12(self.conv12(x)))
        x = self.avgpool(x_cross1)

        x = F.gelu(self.gn21(self.conv21(x)))
        x_cross2 = F.gelu(self.gn22(self.conv22(x)))
        x = self.avgpool(x_cross2)

        x = F.gelu(self.gn31(self.conv31(x)))
        x_cross3 = F.gelu(self.gn32(self.conv32(x)))
        x = self.avgpool(x_cross3)

        batch, _, x_dim, y_dim, z_dim = x.shape
        time = time.view(batch, 1, 1, 1, 1)
        time_channel = time.expand(batch, 1, x_dim, y_dim, z_dim)

        x = torch.cat((x, time_channel), dim=-4)
        x = F.gelu(self.gn_m1(self.conv_m1(x)))
        x = F.gelu(self.gn_m2(self.conv_m2(x)))

        x = self.up1(x)
        x = torch.cat((x, x_cross3), dim=-4)
        x = F.gelu(self.gn41(self.conv41(x)))
        x = F.gelu(self.gn42(self.conv42(x)))

        x = self.up2(x)
        x = torch.cat((x, x_cross2), dim=-4)
        x = F.gelu(self.gn51(self.conv51(x)))
        x = F.gelu(self.gn52(self.conv52(x)))

        x = self.up3(x)
        x = torch.cat((x, x_cross1), dim=-4)
        x = F.gelu(self.gn61(self.conv61(x)))
        x = F.gelu(self.gn62(self.conv62(x)))

        x = self.conv_end(x)

        x[:, :, 0, :, :] = x_input[:, :, 0, :, :]
        x[:, :, -1, :, :] = x_input[:, :, -1, :, :]
        x[:, :, :, 0, :] = x_input[:, :, :, 0, :]
        x[:, :, :, -1, :] = x_input[:, :, :, -1, :]
        x[:, :, :, :, 0] = x_input[0, 0, -1, -1, -1]
        x[:, :, :, :, -1] = x_input[:, :, :, :, -1]
        return x


class PITCNN_dynamic_batchnorm(PITCNN_dynamic):
    def __init__(self, c=8):
        super(PITCNN_dynamic_batchnorm, self).__init__(c=c)

        self.gn11 = nn.BatchNorm3d(num_features=c)
        self.gn12 = nn.BatchNorm3d(num_features=c)
        self.gn21 = nn.BatchNorm3d(num_features=c * 2)
        self.gn22 = nn.BatchNorm3d(num_features=c * 2)
        self.gn31 = nn.BatchNorm3d(num_features=c * 4)
        self.gn32 = nn.BatchNorm3d(num_features=c * 4)
        self.gn_m1 = nn.BatchNorm3d(num_features=c * 8)
        self.gn_m2 = nn.BatchNorm3d(num_features=c * 8)
        self.gn41 = nn.BatchNorm3d(num_features=c * 4)
        self.gn42 = nn.BatchNorm3d(num_features=c * 4)
        self.gn51 = nn.BatchNorm3d(num_features=c * 2)
        self.gn52 = nn.BatchNorm3d(num_features=c * 2)
        self.gn61 = nn.BatchNorm3d(num_features=c)
        self.gn62 = nn.BatchNorm3d(num_features=c)
        