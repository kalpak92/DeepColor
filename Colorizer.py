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
            nn.ConvTranspose2d(in_channels=75, out_channels=50,
                               kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(75),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=50, out_channels=25,
                               kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=25, out_channels=25,
                               kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=25, out_channels=12,
                               kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=12, out_channels=6,
                               kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=6, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        if self.activation_function == Constants.SIGMOID:
            return torch.sigmoid(self.up_sample(self.feature_maps(x)))
        elif self.activation_function == Constants.TANH:
            return torch.tanh(self.up_sample(self.feature_maps(x)))
        elif self.activation_function == Constants.RELU:
            return torch.relu(self.up_sample(self.feature_maps(x)))
