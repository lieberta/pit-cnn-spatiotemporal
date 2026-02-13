import torch
import torch.nn as nn
import torch.nn.functional as F

from training.train_picnn_static import BaseModel


class EncoderBlock_cross(nn.Module):
    def __init__(self, in_c, out_c):
        super(EncoderBlock_cross, self).__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, padding_mode="reflect")
        self.gn1 = nn.GroupNorm(num_groups=int(out_c / 8), num_channels=out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, padding_mode="reflect")
        self.gn2 = nn.GroupNorm(num_groups=int(out_c / 8), num_channels=out_c)
        self.avgpool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.gelu(self.gn1(self.conv1(x)))
        y = F.gelu(self.gn2(self.conv2(x)))
        x = self.avgpool(y)
        return x, y


class DecoderBlock_transposedcnn_cross(nn.Module):
    def __init__(self, in_c, cross_c, out_c):
        super(DecoderBlock_transposedcnn_cross, self).__init__()
        self.up = nn.ConvTranspose3d(in_c, in_c, kernel_size=2, stride=2, output_padding=0)
        self.conv1 = nn.Conv3d(in_c + cross_c, out_c, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=int(out_c / 8), num_channels=out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=int(out_c / 8), num_channels=out_c)

    def forward(self, x, x_cross):
        x = self.up(x)
        x = torch.cat((x, x_cross), dim=-4)
        x = F.gelu(self.gn1(self.conv1(x)))
        x = F.gelu(self.gn2(self.conv2(x)))
        return x


class PICNN_static(BaseModel):
    def __init__(self, loss_fn, channels=8):
        super(PICNN_static, self).__init__(loss_fn=loss_fn)

        self.encoder1 = EncoderBlock_cross(in_c=1, out_c=channels)
        self.encoder2 = EncoderBlock_cross(in_c=channels, out_c=channels * 2)
        self.encoder3 = EncoderBlock_cross(in_c=channels * 2, out_c=channels * 4)

        self.conv1 = nn.Conv3d(channels * 4, channels * 8, kernel_size=3, padding=1, padding_mode="reflect")
        self.gn1 = nn.GroupNorm(num_groups=int(channels), num_channels=channels * 8)
        self.conv2 = nn.Conv3d(channels * 8, channels * 8, kernel_size=3, padding=1, padding_mode="reflect")
        self.gn2 = nn.GroupNorm(num_groups=int(channels), num_channels=channels * 8)

        self.dblock1 = DecoderBlock_transposedcnn_cross(channels * 8, channels * 4, channels * 4)
        self.dblock2 = DecoderBlock_transposedcnn_cross(channels * 4, channels * 2, channels * 2)
        self.dblock3 = DecoderBlock_transposedcnn_cross(channels * 2, channels, channels)

        self.con_end = nn.Conv3d(channels, 1, kernel_size=1, padding=0, padding_mode="reflect")

    def forward(self, x):
        x_input = x
        x, x_cross1 = self.encoder1(x)
        x, x_cross2 = self.encoder2(x)
        x, x_cross3 = self.encoder3(x)

        x = F.gelu(self.gn2(self.conv2(F.gelu(self.gn1(self.conv1(x))))))

        x = self.dblock1(x, x_cross3)
        x = self.dblock2(x, x_cross2)
        x = self.dblock3(x, x_cross1)

        x = self.con_end(x)

        x[:, :, 0, :, :] = x_input[:, :, 0, :, :]
        x[:, :, -1, :, :] = x_input[:, :, -1, :, :]
        x[:, :, :, 0, :] = x_input[:, :, :, 0, :]
        x[:, :, :, -1, :] = x_input[:, :, :, -1, :]
        x[:, :, :, :, 0] = x_input[:, :, :, :, 0]
        x[:, :, :, :, -1] = x_input[:, :, :, :, -1]
        return x
