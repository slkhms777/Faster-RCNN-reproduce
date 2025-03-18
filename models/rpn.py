import torch
from torch import nn
import numpy as np
import torchvision


def generate_anchors(FeatureMap, scales=[128, 256, 512], ratios=[0.5, 1, 2]):
    # FeatureMa : [batch_size, 2048, 25, 19]
    anchors = torch.zeros(FeatureMap.size(0), len(scales) * len(ratios), FeatureMap.size(2), FeatureMap.size(3), 4)
    for i in range(FeatureMap.size(2)):
        for j in range(FeatureMap.size(3)):
            for k, scale in enumerate(scales):
                for l, ratio in enumerate(ratios):
                    h = scale * np.sqrt(ratio)
                    w = scale / np.sqrt(ratio)
                    anchors[:, k * len(ratios) + l, i, j, 0] = i * 32 - w / 2
                    anchors[:, k * len(ratios) + l, i, j, 1] = j * 32 - h / 2
                    anchors[:, k * len(ratios) + l, i, j, 2] = i * 32 + w / 2
                    anchors[:, k * len(ratios) + l, i, j, 3] = j * 32 + h / 2

    return anchors


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

def decode_anchors(anchors, rpn_reg_preds):
    # anchors: [batch_size, 9, 25, 19, 4]
    # rpn_reg_preds: [batch_size, 36, 25, 19]
    proposals = torch.zeros_like(anchors)
    for i in range(anchors.size(2)):
        for j in range(anchors.size(3)):
            for k in range(anchors.size(1)):
                proposals[:, k, i, j, 0] = anchors[:, k, i, j, 0] + rpn_reg_preds[:, k * 4 + 0, i, j]
                proposals[:, k, i, j, 1] = anchors[:, k, i, j, 1] + rpn_reg_preds[:, k * 4 + 1, i, j]
                proposals[:, k, i, j, 2] = anchors[:, k, i, j, 2] + rpn_reg_preds[:, k * 4 + 2, i, j]
                proposals[:, k, i, j, 3] = anchors[:, k, i, j, 3] + rpn_reg_preds[:, k * 4 + 3, i, j]
    return proposals
