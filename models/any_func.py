import torch
from torch import nn
import numpy as np
import torchvision


"""
- `RPN`: 区域提议网络类
- `generate_anchors`: 生成锚框
- `decode_anchors`: 将RPN回归值解码为实际边界框
- `clip_boxes`: 将边界框裁剪到图像范围内
- `filter_proposals`: 筛选提议框(NMS等)
- `ROI_Pooling`: ROI池化操作
"""


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


def DetectionHead(roi_features, valid_mask, num_classes=21):
    # roi_features: [batch_size, top_n, 2048, 7, 7]
    # valid_mask: [batch_size, top_n]
    batch_size, top_n, channels, height, width = roi_features.size()
    device = roi_features.device
    
    # 重塑ROI特征以便处理
    roi_features = roi_features.view(batch_size * top_n, channels, height, width)
    
    # 通过平均池化将特征图转换为特征向量
    roi_pooled = nn.functional.adaptive_avg_pool2d(roi_features, (1, 1))
    roi_pooled = roi_pooled.view(batch_size * top_n, channels)
    
    # 使用全连接层提取特征
    fc1 = nn.Linear(channels, 1024).to(device)
    fc2 = nn.Linear(1024, 1024).to(device)
    cls_layer = nn.Linear(1024, num_classes).to(device)
    reg_layer = nn.Linear(1024, 4).to(device)
    
    # 特征提取
    x = nn.functional.relu(fc1(roi_pooled))
    x = nn.functional.relu(fc2(x))
    
    # 预测分类和回归
    cls_scores = cls_layer(x)
    reg_preds = reg_layer(x)
    
    # 重塑回原始批次结构
    cls_scores = cls_scores.view(batch_size, top_n, num_classes)
    reg_preds = reg_preds.view(batch_size, top_n, 4)
    
    # 应用掩码，将无效ROI的预测设为0
    valid_mask = valid_mask.float().unsqueeze(-1)  # [batch_size, top_n, 1]
    cls_scores = cls_scores * valid_mask
    reg_preds = reg_preds * valid_mask.repeat(1, 1, reg_preds.size(-1))
    # cls_scores: [batch_size, top_n, num_classes]
    # reg_preds: [batch_size, top_n, 4]
    return cls_scores, reg_preds

def decode_proposals(proposals, reg_preds):
    # proposals torch.Size([batch_size, top_n, 4])
    # reg_preds: [batch_size, top_n, 4]

    detections = proposals + reg_preds
    # detections: [batch_size, top_n, 4]

    return detections

