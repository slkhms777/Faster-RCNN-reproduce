import torch 
from torch import nn
import torchvision
import backbone
import rpn

def Main():
    images = torch.randn(2, 3, 800, 600)
    batch_size = images.shape[0]
    Backbone = backbone.get_net('resnet50', pretrained=True, output_size='small')
    FeatureMap = Backbone(images)
    # FeatureMap: torch.Size([batch_size, 2048, 25, 19])

    anchors = rpn.generate_anchors(FeatureMap, scales=[128, 256, 512], ratios=[0.5, 1, 2])
    # anchors: torch.Size([batch_size, 9, 25, 19]) # 每个特征点生成 9 个锚点

    RPN = rpn.RPN(in_channels=2048, num_anchors=9)
    
    rpn_cls_scores, rpn_reg_preds = RPN(FeatureMap)


    # 对前景/背景得分应用 softmax
    # 然后使用前景概率
    # rpn_cls_scores: torch.Size([batch_size, 18, 25, 19]) # 锚点的前景得分, 背景得分
    # rpn_reg_preds: torch.Size([batch_size, 36, 25, 19]) #
    # rpn_cls_probs: torch.Size([batch_size, 2, 9, 25, 19])
    # foreground_probs: torch.Size([batch_size, 9, 25, 19])
    rpn_cls_probs = torch.softmax(rpn_cls_scores.view(batch_size, 2, 9, 25, 19), dim=1)
    foreground_probs = rpn_cls_probs[:, 1]  # 索引1为前景类别概率


    proposals = rpn.decode_anchors(anchors, rpn_reg_preds)  # 用预测偏移量调整锚点坐标
    proposals = rpn.clip_boxes(proposals, images.shape[-2:])  # 限制候选框在图像范围内

    proposals = rpn.filter_proposals(
        proposals,
        foreground_probs,
        nms_threshold=0.7,
        top_n=1000
    )
    print('proposals:', proposals.shape) # proposals torch.Size([batch_size, top_n, 4])

    roi_features, valid_mask = rpn.ROI_Pooling(FeatureMap, proposals, output_size=7)  # 输出 7×7 特征
    # roi_features 尺寸: (batch_size, top_n, 2048, 7, 7)
    # valid_mask: (batch_size, top_n)  # 记录每个候选框是否有效
    print('roi_features:', roi_features.shape)
    print('valid_mask:', valid_mask.shape)

    cls_scores, reg_preds = rpn.DetectionHead(roi_features, valid_mask)
    print('cls_scores:', cls_scores.shape)
    print('reg_preds:', reg_preds.shape)
    # cls_scores: [batch_size, top_n, num_classes]
    # reg_preds: [batch_size, top_n, 4]

    detections = rpn.decode_proposals(proposals, reg_preds)  # 用回归值调整候选框
    # detections: [batch_size, top_n, 4]


if __name__ == '__main__':
    Main()


