import torch
from torch import nn


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

