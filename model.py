import torch
import torch.nn as nn
from global_config import *


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=9, stride=1, padding=9//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=1, stride=1, padding=0//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=3,
                      kernel_size=5, stride=1, padding=5//2),
            nn.ReLU(),
        )
        self.loss_function = nn.MSELoss()

    def evalvalidate(self, input):
        return self(input)

    def forward(self, input):
        return torch.clamp(self.model(input), min=1e-12, max=1-(1e-12))


class ModelVDSR(nn.Module):
    def __init__(self):
        super(ModelVDSR, self).__init__()
        self.in_layer = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        res_layers = []
        for i in range(10):
            res_layers.append(nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True))
            )
        self.residual_layer = nn.Sequential(*res_layers)
        self.out_layer = nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.loss_function = nn.MSELoss()

    def evalvalidate(self, input):
        return torch.cat((self(input[0:1]), input[1:3]))

    def forward(self, input):
        residual = input
        out = self.relu(self.in_layer(input))
        out = self.residual_layer(out)
        out = self.out_layer(out)
        out = torch.add(out, residual)
        return out
