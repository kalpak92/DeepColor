import torch.nn as nn

# from torchsummary import summary
from Regressor import Regressor


class Colorizer(nn.Module):
    def __init__(self, in_channel=3, hidden_channel=3, out_channel=2, is_RELU=True):
        super(Colorizer, self).__init__()
        self.feature_maps = Regressor(in_channel=1, hidden_channel=3, out_dims=2,
                                      train_mode="colorizer")
        if is_RELU:
            self.up_sample = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channel, out_channels=hidden_channel,
                                   kernel_size=2, stride=2),
                nn.BatchNorm2d(hidden_channel),
                nn.ReLU(True),

                nn.ConvTranspose2d(in_channels=hidden_channel, out_channels=hidden_channel,
                                   kernel_size=2, stride=2),
                nn.BatchNorm2d(hidden_channel),
                nn.ReLU(True),

                nn.ConvTranspose2d(in_channels=hidden_channel, out_channels=hidden_channel,
                                   kernel_size=2, stride=2),
                nn.BatchNorm2d(hidden_channel),
                nn.ReLU(True),

                nn.ConvTranspose2d(in_channels=hidden_channel, out_channels=hidden_channel,
                                   kernel_size=2, stride=2),
                nn.BatchNorm2d(hidden_channel),
                nn.ReLU(True),

                nn.ConvTranspose2d(in_channels=hidden_channel, out_channels=hidden_channel,
                                   kernel_size=2, stride=2),
                nn.BatchNorm2d(hidden_channel),
                nn.ReLU(True),

                nn.ConvTranspose2d(in_channels=hidden_channel, out_channels=out_channel,
                                   kernel_size=2, stride=2),
                nn.ReLU(),
            )

    def forward(self, x):
        return self.up_sample(self.feature_maps(x))
