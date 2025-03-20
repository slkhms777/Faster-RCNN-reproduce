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
    # proposals: [batch_size, top_n, 4] 无效框的坐标全是-1
    # output_size: 输出特征图大小
    batch_size, channels, height, width = FeatureMap.size()
    top_n = proposals.size(1)

    x_min = proposals[:, :, 0] // 32
    y_min = proposals[:, :, 1] // 32
    x_max = proposals[:, :, 2] // 32
    y_max = proposals[:, :, 3] // 32
    # 计算ROI的宽度和高度
    
    # 创建掩码标识有效的ROI（坐标不是-1的）
    mask = (proposals[:, :, 0] >= 0)
    
    # 确保坐标在特征图范围内
    x_min = torch.clamp(x_min, 0, width - 1)
    y_min = torch.clamp(y_min, 0, height - 1)
    x_max = torch.clamp(x_max, 0, width - 1)
    y_max = torch.clamp(y_max, 0, height - 1)
    
    # 更新掩码以排除无效的ROI（宽度或高度小于等于0）
    mask = mask & (x_max >= x_min) & (y_max >= y_min)
    
    # 初始化结果张量
    roi_features = torch.zeros(batch_size, top_n, channels, output_size, output_size, device=FeatureMap.device)
    
    # 使用torchvision的ROIAlign函数进行批量处理
    # 首先构建符合ROIAlign格式的索引张量

    valid_indices = torch.nonzero(mask, as_tuple=True)
    if len(valid_indices[0]) > 0:
        # 提取有效的批次索引和proposal索引
        batch_indices = valid_indices[0]
        prop_indices = valid_indices[1]
        
        # 构建ROIAlign所需的boxes格式: (batch_idx, x1, y1, x2, y2)
        boxes = torch.zeros(len(batch_indices), 5, device=FeatureMap.device)
        boxes[:, 0] = batch_indices
        boxes[:, 1] = x_min[batch_indices, prop_indices]
        boxes[:, 2] = y_min[batch_indices, prop_indices]
        boxes[:, 3] = x_max[batch_indices, prop_indices]
        boxes[:, 4] = y_max[batch_indices, prop_indices]
        
        # 使用torchvision的ROIAlign
        import torchvision.ops as ops
        pooled = ops.roi_align(
            FeatureMap, 
            boxes,
            output_size=(output_size, output_size),
            spatial_scale=1.0,  # 因为我们已经将坐标缩小了32倍
            sampling_ratio=-1   # 采用自适应采样
        )
        
        # 将结果填回原始张量
        for i in range(len(batch_indices)):
            b_idx, p_idx = batch_indices[i], prop_indices[i]
            roi_features[b_idx, p_idx] = pooled[i]

    # roi_features: [batch_size, top_n, channels, output_size, output_size]
    # mask: [batch_size, top_n]    
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

