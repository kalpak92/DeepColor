import torch
import torch.nn as nn
import torch.nn.functional as F


class Regressor(nn.Module):
    def __init__(self, in_channel=1, hidden_channel=3, out_dims=2,
                 train_mode="regressor"):
        super(Regressor, self).__init__()
        self.train_mode = train_mode
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=256)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256,
                               kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=256)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=512)

        if self.train_mode == "regressor":
            self.lin = nn.Linear(in_features=512 * 2 * 2, out_features=out_dims)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.leaky_relu(self.bn5(self.conv5(x)))

        feature_maps = F.leaky_relu(self.bn6(self.conv6(x)))

        if self.train_mode == "regressor":
            y_hat = torch.sigmoid(self.lin(feature_maps.reshape(-1, 512 * 2 * 2)))
            return y_hat

        else:
            return feature_maps

# model = Regressor()
#
# summary(model, (1, 128, 128))
