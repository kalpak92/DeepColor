import torch
import torch.nn as nn
# from torchsummary import summary


class Regressor(nn.Module):
    def __init__(self, in_channel=1, hidden_channel=3, out_dims=2,
                 train_mode="regressor"):
        super(Regressor, self).__init__()
        self.train_mode = train_mode

        self.feature_maps = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d(in_channels=in_channel, out_channels=hidden_channel,
                      kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel,
                      kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel,
                      kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel,
                      kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel,
                      kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel,
                      kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(),
        )

        if self.train_mode == "regressor":
            self.lin = nn.Linear(in_features=3 * 2 * 2, out_features=out_dims)

    def forward(self, input):
        feature_maps = self.feature_maps(input)

        if self.train_mode == "regressor":
            y_hat = torch.sigmoid(self.lin(feature_maps.reshape(-1, 3 * 2 * 2)))
            return y_hat

        else:
            return feature_maps

# model = Regressor()
#
# summary(model, (1, 128, 128))
