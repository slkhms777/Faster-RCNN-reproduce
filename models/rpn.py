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
                    anchors[:, k * len(ratios) + l, i, j, 0] = i * 32 - w / 2 # x_min
                    anchors[:, k * len(ratios) + l, i, j, 1] = j * 32 - h / 2 # y_min
                    anchors[:, k * len(ratios) + l, i, j, 2] = i * 32 + w / 2 # x_max
                    anchors[:, k * len(ratios) + l, i, j, 3] = j * 32 + h / 2 # y_max

    return anchors

def iou(boxes1, boxes2):
    # boxes1: [N, 4]
    # boxes2: [M, 4]
    N = boxes1.size(0)
    M = boxes2.size(0)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2] 左上角坐标
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2] 右下角坐标
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]
    iou = inter / (area1[:, None] + area2 - inter)
    # iou: [N, M]
    return iou

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
        batch_keep_indices.append(torch.tensor(keep))
    
    # 将保留的索引打包成一个张量
    return batch_keep_indices

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
    result_proposals = []
    for i in range(batch_size):
        # 获取当前批次保留的索引
        keep_idx = keep_indices[i]
        # 将3D索引（anchor_idx, h_idx, w_idx）转换回线性索引
        proposals_flat = proposals[i].view(-1, 4)  # [9*25*19, 4]
        # 保留选中的框
        result_proposals.append(proposals_flat[keep_idx])

    # 将结果包装为批次形式 [batch_size, N, 4], N 可能对每个批次不同
    return torch.stack([r for r in result_proposals])