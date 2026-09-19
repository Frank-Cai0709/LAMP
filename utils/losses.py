import torch
from torch import nn


class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.smooth = smooth

    def forward(self, prediction, target):
        bce = self.bce(prediction, target)
        pred = prediction.flatten(1)
        truth = target.flatten(1)
        dice = (2 * (pred * truth).sum(1) + self.smooth) / (
            pred.sum(1) + truth.sum(1) + self.smooth
        )
        return bce + 1 - dice.mean()
