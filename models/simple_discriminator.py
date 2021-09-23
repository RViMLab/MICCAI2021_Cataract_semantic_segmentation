import torch
import torch.nn as nn


class SimpleDiscriminator(nn.Module):
    r"""
    Args:
        c channel (int) 
        h height (int) 
        w width (int) 
    """
    def __init__(self, c: int, h: int, w: int, d: int=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=d, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=d)
        self.conv2 = nn.Conv2d(in_channels=d, out_channels=2*d, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(num_features=2*d)
        self.conv3 = nn.Conv2d(in_channels=2*d, out_channels=4*d, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(num_features=4*d)
        self.flatten = nn.Flatten()

        n = self._numel(c, h, w)
        self.fc1 = nn.Linear(n, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # Convolutional layers
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)  # flatten

        # Linear layers
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

    def _numel(self, c: int, h: int, w: int):
        x = torch.ones((1, c, h, w))
        x = self.conv3(self.conv2(self.conv1(x)))
        return x.numel()
