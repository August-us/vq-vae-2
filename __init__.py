import torch
from torch import nn


class EvoNorms0_2d(nn.Module):
    __constants__ = ['num_features', 'eps', 'nonlinearity']

    def __init__(self, num_features, eps=1e-5, nonlinearity=True):
        super(EvoNorms0_2d, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.nonlinearity = nonlinearity

        self.weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        if self.nonlinearity:
            self.v = nn.Parameter(torch.Tensor(1, num_features, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, 1)
        nn.init.constant_(self.bias, 0)
        if self.nonlinearity:
            nn.init.constant_(self.v, 1)

    def group_std(self, x, groups=32):
        N, C, H, W = x.shape
        x = torch.reshape(x, (N, groups, C//groups, H, W))
        std = torch.std(x, (2, 3, 4), keepdim=True).expand_as(x)
        return torch.reshape(std + self.eps, (N, C, H, W))

    def forward(self, x):
        if self.nonlinearity:
            num = x*torch.sigmoid(self.v * x)
            return num / self.group_std(x) * self.weight + self.bias
        else:
            return x * self.weight + self.bias


EvoNorms0_2d(3)