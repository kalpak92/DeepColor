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
        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=50, out_channels=32,
                               kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=16,
                               kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=16, out_channels=8,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=8, out_channels=8,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=8, out_channels=4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=4, out_channels=out_channel,
                               kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        if self.activation_function == Constants.SIGMOID:
            return torch.sigmoid(self.up_sample(self.feature_maps(x)))
        elif self.activation_function == Constants.TANH:
            return torch.tanh(self.up_sample(self.feature_maps(x)))
        elif self.activation_function == Constants.RELU:
            return torch.relu(self.up_sample(self.feature_maps(x)))
