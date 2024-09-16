import torch
from ConvBlock import ConvBlock
import torchinfo
class Discriminator(torch.nn.Module):
    def __init__(self, input_shape):
        super(Discriminator,self).__init__()
        in_channels, in_height, in_width = input_shape
        patch_height, patch_width = int(in_height // 2 ** 4), int(in_width // 2 ** 4)
        self.conv_1 = torch.nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=64, stride=1, padding=1)
        self.activation = torch.nn.LeakyReLU(0.2)
        discriminator_block = []
        activation_type = "lrelu"
        disc_out_channels = [64, 128, 256, 512]
        for i, out_channels in enumerate(disc_out_channels):
            if i == 0:
                discriminator_block.append(
                    ConvBlock(in_channels=in_channels,
                              out_channels=disc_out_channels[i],
                              activation_type=activation_type,
                              stride=2, padding=1, kernel_size=3)
                )
            elif i == 1:
                discriminator_block.append(
                    ConvBlock(in_channels=disc_out_channels[0],
                              out_channels=disc_out_channels[i], kernel_size=3,
                              stride=1,
                              activation_type=activation_type,
                              padding=1)
                )
            else:
                discriminator_block.append(
                    ConvBlock(in_channels=disc_out_channels[i - 1],
                              out_channels=disc_out_channels[i], kernel_size=3,
                              stride=1,
                              activation_type=activation_type,
                              padding=1)
                )
        self.dics = torch.nn.Sequential(*discriminator_block)
    def forward(self, x):
        x = self.dics(x)
        return x
