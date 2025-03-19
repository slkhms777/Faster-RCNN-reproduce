import torch
import numpy as np

def generate_anchors(feature_map, scales=[128, 256, 512], aspect_ratios=[0.5, 1, 2], stride=32):
    """
    为特征图的每个位置生成锚框 - 高效版本
    """
    batch_size, _, height, width = feature_map.shape
    device = feature_map.device
    num_anchors = len(scales) * len(aspect_ratios)
    
    # 生成特征图每个位置的中心坐标
    center_y = torch.arange(0, height, device=device) * stride + stride // 2
    center_x = torch.arange(0, width, device=device) * stride + stride // 2
    
    # 扩展为网格坐标 [height, width]
    center_x, center_y = torch.meshgrid(center_x, center_y, indexing='xy')
    
    # 创建用于存储所有生成的宽度和高度的列表
    widths = []
    heights = []
    
    # 计算所有尺度和宽高比组合的宽度和高度
    for scale in scales:
        for ratio in aspect_ratios:
            widths.append(scale * np.sqrt(ratio))
            heights.append(scale / np.sqrt(ratio))
    
    # 转换为张量
    widths = torch.tensor(widths, device=device)
    heights = torch.tensor(heights, device=device)
    
    # 扩展维度以便后续广播
    # center_x, center_y: [height, width] -> [height, width, 1]
    # widths, heights: [num_anchors] -> [1, 1, num_anchors]
    center_x = center_x.unsqueeze(-1)
    center_y = center_y.unsqueeze(-1)
    widths = widths.view(1, 1, -1)
    heights = heights.view(1, 1, -1)
    
    # 计算锚框坐标 [height, width, num_anchors]
    x1 = center_x - widths / 2
    y1 = center_y - heights / 2
    x2 = center_x + widths / 2
    y2 = center_y + heights / 2
    
    # 堆叠坐标 [height, width, num_anchors, 4]
    anchors_per_batch = torch.stack([x1, y1, x2, y2], dim=-1)
    
    # 扩展批次维度 [batch_size, height, width, num_anchors, 4]
    anchors = anchors_per_batch.expand(batch_size, height, width, num_anchors, 4)
    
    return anchors