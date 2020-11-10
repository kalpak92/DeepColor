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
            nn.Conv2d(in_channels=in_channel, out_channels=10,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=10, out_channels=25,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=25, out_channels=50,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=50, out_channels=75,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(75),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=75, out_channels=100,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=100, out_channels=125,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(125),
            nn.LeakyReLU(),
        )

        if self.train_mode == "regressor":
            self.lin = nn.Linear(in_features=125 * 2 * 2, out_features=out_dims)

    def forward(self, x):
        feature_maps = self.feature_maps(x)
        if self.train_mode == "regressor":
            y_hat = torch.sigmoid(self.lin(feature_maps.reshape(-1, 125 * 2 * 2)))
            return y_hat

        else:
            return feature_maps

# model = Regressor()
#
# summary(model, (1, 128, 128))
