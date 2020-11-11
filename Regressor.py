import torch
import torch.nn as nn
import torch.nn.functional as F


class Regressor(nn.Module):
    def __init__(self, in_channel=1, hidden_channel=3, out_dims=2,
                 train_mode="regressor"):
        super(Regressor, self).__init__()
        self.train_mode = train_mode

        self.feature_maps = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d(in_channels=in_channel, out_channels=32,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )

        if self.train_mode == "regressor":
            self.lin = nn.Linear(in_features=512 * 2 * 2, out_features=out_dims)

    def forward(self, x):
        feature_maps = self.feature_maps(x)
        if self.train_mode == "regressor":
            y_hat = torch.sigmoid(self.lin(feature_maps.reshape(-1, 512 * 2 * 2)))
            return y_hat

        else:
            return feature_maps

# model = Regressor()
#
# summary(model, (1, 128, 128))
