import torch
from torch import nn

def ROI_Pooling(FeatureMap, proposals, output_size=7):
    # FeatureMap: [batch_size, 2048, 25, 19]
    # proposals: [batch_size, top_n, 4]
    # output_size: 输出特征图大小
    # roi_features: [num_proposals, 2048, output_size, output_size]
    batch_size, top_n, _ = proposals.size()
    roi_features = torch.zeros(batch_size, top_n, FeatureMap.size(1), output_size, output_size)
    for i in range(batch_size):
        for j in range(top_n):
            # 检查是否是有效框
            if proposals[i, j, 0] >= 0:  # 不是填充的无效框
                # 将原始图像坐标缩放到特征图坐标
                x_min = max(0, int(proposals[i, j, 0] // 32))
                y_min = max(0, int(proposals[i, j, 1] // 32))
                x_max = min(FeatureMap.size(2)-1, int(proposals[i, j, 2] // 32))
                y_max = min(FeatureMap.size(3)-1, int(proposals[i, j, 3] // 32))
                
                # 确保区域有效
                if x_max > x_min and y_max > y_min:
                    roi_features[i, j] = nn.functional.adaptive_max_pool2d(
                        FeatureMap[i, :, y_min:y_max, x_min:x_max], output_size)
                else :
                    roi_features[i, j] = -1.0
    # 不改变形状，返回原始形状和掩码
    mask = roi_features[:, :, 0, 0, 0] >= 0  # [batch_size, top_n]
    return roi_features, mask

