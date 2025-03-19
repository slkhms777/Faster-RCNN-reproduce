import torch
import torch.nn.functional as F
from utils.nms import nms

def decode_anchors(anchors, rpn_reg_preds):
    # anchors: [batch_size, num_anchors, 4]
    # rpn_reg_preds: [batch_size, num_anchors, 4]
    proposals = anchors + rpn_reg_preds
    return proposals

def clip_boxes(proposals, img_size):
    # proposals: [batch_size, num_anchors, 4]
    proposals[:, :, 0] = torch.clamp(proposals[:, :, 0], min=0, max=img_size[1] - 1)  # x_min
    proposals[:, :, 1] = torch.clamp(proposals[:, :, 1], min=0, max=img_size[0] - 1)  # y_min
    proposals[:, :, 2] = torch.clamp(proposals[:, :, 2], min=0, max=img_size[1] - 1)  # x_max
    proposals[:, :, 3] = torch.clamp(proposals[:, :, 3], min=0, max=img_size[0] - 1)  # y_max
    return proposals

def filter_proposals(proposals, foreground_probs, nms_threshold=0.7, top_n=1000):


    # proposals: [batch_size, 25 * 19 * 9 , 4]
    # foreground_probs: [batch_size, 25 * 19 * 9]

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