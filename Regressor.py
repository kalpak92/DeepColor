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
            nn.Conv2d(in_channels=in_channel, out_channels=4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=4, out_channels=8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=8, out_channels=8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=8, out_channels=16,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=32, out_channels=50,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(),
        )

        # if self.train_mode == "regressor":
        #     self.lin = nn.Linear(in_features=50 * 2 * 2, out_features=out_dims)

    def forward(self, x):
        feature_maps = self.feature_maps(x)
        if self.train_mode == "regressor":
            # y_hat = torch.sigmoid(self.lin(feature_maps.reshape(-1, 50 * 2 * 2)))
            return feature_maps

        else:
            return feature_maps

# model = Regressor()
#
# summary(model, (1, 128, 128))
