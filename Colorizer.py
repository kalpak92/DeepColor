import torch
import torch.nn as nn

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
            nn.ConvTranspose2d(in_channels=in_channel, out_channels=hidden_channel,
                               kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=hidden_channel, out_channels=hidden_channel,
                               kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=hidden_channel, out_channels=hidden_channel,
                               kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=hidden_channel, out_channels=hidden_channel,
                               kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=hidden_channel, out_channels=hidden_channel,
                               kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=hidden_channel, out_channels=out_channel,
                               kernel_size=2, stride=2)
        )

    def forward(self, x):
        if self.activation_function == Constants.SIGMOID:
            return torch.sigmoid(self.up_sample(self.feature_maps(x)))
        elif self.activation_function == Constants.TANH:
            return torch.tanh(self.up_sample(self.feature_maps(x)))
        elif self.activation_function == Constants.RELU:
            return torch.relu(self.up_sample(self.feature_maps(x)))
