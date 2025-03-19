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

def clip_boxes(proposals, img_size):
    # proposals: [batch_size, 9, 25, 19, 4]
    # img_size: [H, W]
    proposals[:, :, :, :, 0] = torch.clamp(proposals[:, :, :, :, 0], min=0, max=img_size[1])
    proposals[:, :, :, :, 1] = torch.clamp(proposals[:, :, :, :, 1], min=0, max=img_size[0])
    proposals[:, :, :, :, 2] = torch.clamp(proposals[:, :, :, :, 2], min=0, max=img_size[1])
    proposals[:, :, :, :, 3] = torch.clamp(proposals[:, :, :, :, 3], min=0, max=img_size[0])
    return proposals

def filter_proposals(proposals, foreground_probs, nms_threshold=0.7, top_n=1000):
    # proposals: [batch_size, 9, 25, 19, 4]
    # foreground_probs: [batch_size, 9, 25, 19]

    # 保留前景得分高的 top_n 个候选框
    batch_size = proposals.size(0)
    keep_indices = nms(proposals, foreground_probs, nms_threshold, top_n)
    
    # 根据NMS结果筛选候选框
    result_proposals = torch.zeros(batch_size, top_n, 4)  # 创建固定大小的输出张量
    
    for i in range(batch_size):
        # 获取当前批次保留的索引
        keep_idx = keep_indices[i]
        # 将3D索引（anchor_idx, h_idx, w_idx）转换回线性索引
        proposals_flat = proposals[i].view(-1, 4)  # [9*25*19, 4]
        
        # 计算需要保留的框数
        num_keep = min(len(keep_idx), top_n)
        
        # 保留选中的框
        if num_keep > 0:
            result_proposals[i, :num_keep] = proposals_flat[keep_idx[:num_keep]]
        
        # 将剩余位置填充为-1（表示无效框）
        if num_keep < top_n:
            result_proposals[i, num_keep:] = -1.0

    return result_proposals  # [batch_size, top_n, 4]

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





def nms(boxes, foreground_probs, threshold=0.7, top_n=1000):
    # boxes: [batch_size, 9, 25, 19, 4]
    # foreground_probs: [batch_size, 9, 25, 19]
    batch_size = boxes.size(0)
    
    # 重塑 boxes 和 foreground_probs 用于处理
    boxes_reshaped = boxes.view(batch_size, -1, 4)  # [batch_size, 4275, 4]
    foreground_probs = foreground_probs.view(batch_size, -1)  # [batch_size, 4275]
    
    # 存储每个批次的保留索引
    batch_keep_indices = []
    
    # 对每个批次单独执行 NMS
    for i in range(batch_size):
        # 按分数降序排序
        scores = foreground_probs[i]
        boxes_batch = boxes_reshaped[i]
        _, order = torch.sort(scores, descending=True)
        
        keep = []
        while order.numel() > 0:
            # 保留分数最高的框
            if len(keep) >= top_n:
                break
                
            # 当前分数最高的框
            idx = order[0]
            keep.append(idx)
            
            # 如果只剩一个框，结束循环
            if order.numel() == 1:
                break
                
            # 计算当前最高分框与其他框的IoU
            current_box = boxes_batch[idx].unsqueeze(0)  # [1, 4]
            other_boxes = boxes_batch[order[1:]]  # [n-1, 4]
            
            ious = iou(current_box, other_boxes)[0]  # [n-1]
            
            # 保留IoU低于阈值的框
            order = order[1:][ious < threshold]
            # keep_order = order[1:][ious < threshold]
            # if keep_order.numel() + len(keep) >= top_n:
            #     order = keep_order
            # else:
            #     order = order[1:]

        batch_keep_indices.append(torch.tensor(keep))
    
    # 将保留的索引打包成一个张量
    return batch_keep_indices


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

