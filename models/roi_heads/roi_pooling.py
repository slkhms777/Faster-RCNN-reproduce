import torch
from torch import nn

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
