import torch
import numpy as np


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