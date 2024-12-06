import torch
from torch import nn

class GaussianRegressionLoss(nn.Module):
    def __init__(self):
        super(GaussianRegressionLoss, self).__init__()

    def forward(self, pred, target):
        assert pred.shape[1] == 2   # we need to have an estimate of mean and variance

        mean = pred[:, 0]
        logvar = pred[:, 1]

        loss = torch.exp(-logvar) * (mean - target) ** 2 + logvar
        return loss.mean()