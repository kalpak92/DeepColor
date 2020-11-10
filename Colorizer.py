import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchsummary import summary
from Constants import Constants
from Regressor import Regressor


class Colorizer(nn.Module):
    def __init__(self, in_channel=3, hidden_channel=3, out_channel=2,
                 activation_function=Constants.SIGMOID):
        super(Colorizer, self).__init__()
        self.activation_function = activation_function
        self.feature_maps = Regressor(in_channel=1, hidden_channel=3, out_dims=2,
                                      train_mode="colorizer")

        self.conv_transpose1 = nn.ConvTranspose2d(in_channels=512, out_channels=256,
                                                  kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=256)

        self.conv_transpose2 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                                  kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=256)

        self.conv_transpose3 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                                  kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)

        self.conv_transpose4 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                                  kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=64)

        self.conv_transpose5 = nn.ConvTranspose2d(in_channels=64, out_channels=32,
                                                  kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=32)

        self.conv_transpose6 = nn.ConvTranspose2d(in_channels=32, out_channels=out_channel,
                                                  kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.feature_maps(x)
        x = F.leaky_relu(self.bn1(self.conv_transpose1(x)))
        x = F.leaky_relu(self.bn2(self.conv_transpose2(x)))
        x = F.leaky_relu(self.bn3(self.conv_transpose3(x)))
        x = F.leaky_relu(self.bn4(self.conv_transpose4(x)))
        x = F.leaky_relu(self.bn5(self.conv_transpose5(x)))

        if self.activation_function == Constants.SIGMOID:
            return torch.sigmoid(self.conv_transpose6(x))
        elif self.activation_function == Constants.TANH:
            return torch.tanh(self.conv_transpose6(x))
        elif self.activation_function == Constants.RELU:
            return torch.relu(self.conv_transpose6(x))
