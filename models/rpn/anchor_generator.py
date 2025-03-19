import torch
import numpy as np

def generate_anchors(FeatureMap, scales=[128, 256, 512], ratios=[0.5, 1, 2]):
    # FeatureMa : [batch_size, 2048, 25, 19]
    anchors = torch.zeros(FeatureMap.size(0), len(scales) * len(ratios), FeatureMap.size(2), FeatureMap.size(3), 4)
    for i in range(FeatureMap.size(2)):
        for j in range(FeatureMap.size(3)):
            for k, scale in enumerate(scales):
                for l, ratio in enumerate(ratios):
                    h = scale * np.sqrt(ratio)
                    w = scale / np.sqrt(ratio)
                    anchors[:, k * len(ratios) + l, i, j, 0] = i * 32 - w / 2 # x_min
                    anchors[:, k * len(ratios) + l, i, j, 1] = j * 32 - h / 2 # y_min
                    anchors[:, k * len(ratios) + l, i, j, 2] = i * 32 + w / 2 # x_max
                    anchors[:, k * len(ratios) + l, i, j, 3] = j * 32 + h / 2 # y_max

    return anchors