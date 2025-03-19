import torch
from torch import nn

class RPN(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.cls = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=1)
        self.reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

    def forward(self, x):
        # x为FeatureMap
        x = self.relu(self.conv(x))
        cls_scores = self.cls(x)
        reg_preds = self.reg(x)
        return cls_scores, reg_preds