import torch
from torch import nn
from models.backbone import backbone
from models.rpn import  anchor_generator
from models.rpn import proposal_layer
from models.rpn import rpn_head
from models.roi_heads import roi_pooling
from models.roi_heads import box_head

class FasterRCNN(nn.Module):
    def __init__(self, image_shape, num_classes=21):
        super(FasterRCNN, self).__init__()
        self.batch_size = image_shape[0]
        self.h, self.w = image_shape[-2:]

        self.Backbone = backbone.get_net('resnet50', pretrained=True, output_size='small')
        
        self.anchor_generator = anchor_generator.generate_anchors

        self.RPN = rpn_head.RPN(in_channels=2048, num_anchors=9)

        self.anchor_decoder = proposal_layer.decode_anchors #参数: anchors, rpn_reg_preds
        self.anchor_clipper = proposal_layer.clip_boxes #参数: proposals, image_shape
        self.anchor_filter = proposal_layer.filter_proposals #参数: proposals, foreground_probs, nms_threshold=0.7, top_n=1000
        
        self.ROI_Pooling = roi_pooling.ROI_Pooling
        self.DetectionHead = box_head.DetectionHead
        self.proposal_decoder = box_head.decode_proposals
    
    def forward(self, x):
        """生成特征图"""
        # x: torch.Size([batch_size, 3, 800, 600])
        FeatureMap = self.Backbone(x)
        # FeatureMap: torch.Size([batch_size, 2048, 25, 19])

        """生成锚框"""
        anchors = self.anchor_generator(FeatureMap).view(self.batch_size, -1, 4)
        # anchors: torch.Size([batch_size, 25, 19, 9, 4]) 
        # -> torch.Size([batch_size, 25 * 19 * 9, 4]) 

        """RPN预测"""
        rpn_cls_scores, rpn_reg_preds = self.RPN(FeatureMap)

        rpn_cls_scores = rpn_cls_scores.permute(0, 2, 3, 1).contiguous().view(self.batch_size, -1, 2)
        rpn_reg_preds = rpn_reg_preds.permute(0, 2, 3, 1).contiguous().view(self.batch_size, -1, 4)
        rpn_cls_probs = torch.softmax(rpn_cls_scores, dim=-1)
        # rpn_cls_scores: torch.Size([batch_size, 25 * 19 * 9, 2])
        # rpn_reg_preds: torch.Size([batch_size, 25 * 19 * 9, 4])
        # rpn_cls_probs: torch.Size([batch_size, 25 * 19 * 9, 2])

        foreground_probs = rpn_cls_probs[:, :, 1]
        # foreground_probs: torch.Size([batch_size, 25 * 19 * 9])

        """生成候选框"""
        proposals = self.anchor_decoder(anchors, rpn_reg_preds)      # 加上偏移量
        proposals = self.anchor_clipper(proposals, (self.h, self.w)) # 裁剪到图像范围
        proposals = self.anchor_filter(                              # nms筛选候选框
            proposals,
            foreground_probs,
            nms_threshold=0.7,
            top_n=1000
        )

        """ROI处理"""
        roi_features, valid_mask = self.ROI_Pooling(FeatureMap, proposals, output_size=7)
        # roi_features: torch.Size([batch_size, top_n, 2048, 7, 7])
        # valid_mask: torch.Size([batch_size, top_n])

        """检测头阶段"""
        cls_scores, reg_preds = self.DetectionHead(roi_features, valid_mask)

        """最终检测结果"""
        detections = self.proposal_decoder(proposals, reg_preds)
        return (detections, cls_scores)









