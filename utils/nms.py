import torch
from utils.box_ops import iou
def nms(boxes, foreground_probs, threshold=0.7, top_n=1000):
    # boxes: [batch_size, 25 * 19 * 9 , 4]
    # foreground_probs: [batch_size, 25 * 19 * 9]
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
