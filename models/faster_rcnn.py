import torch
from torch import nn
from backbone import backbone
import rpn
import roi_heads
from rpn import anchor_generator
from rpn import proposal_layer
from rpn import rpn_head
from roi_heads import box_heads
from roi_heads import roi_pooling



class FasterRCNN(nn.Module):
    def __init__(self, image_shape, num_classes=21):
        super(FasterRCNN, self).__init__()
        self.batch_size = image_shape[0]
        self.h, self.w = image_shape[-2:]

        self.Backbone = backbone.get_net('resnet50', pretrained=True, output_size='small')
        
        self.anchor_generator = anchor_generator.generate_anchors(
            sizes=[128, 256, 512],
            aspect_ratios=[0.5, 1, 2]
        )

        self.RPN = rpn_head.RPN(in_channels=2048, num_anchors=9)

        self.anchor_decoder = proposal_layer.decode_anchors #参数: anchors, rpn_reg_preds
        self.anchor_clipper = proposal_layer.clip_boxes #参数: proposals, image_shape
        self.anchor_filter = proposal_layer.filter_proposals #参数: proposals, foreground_probs, nms_threshold=0.7, top_n=1000
        
        self.ROI_Pooling = roi_pooling.ROI_Pooling
        self.DetectionHead = box_heads.DetectionHead
        self.proposal_decoder = box_heads.decode_proposals
    
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
        proposals = self.anchor_decoder(anchors, rpn_reg_preds)
        proposals = self.anchor_clipper(proposals, (self.h, self.w))
        proposals = self.anchor_filter(
            proposals,
            foreground_probs,
            nms_threshold=0.7,
            top_n=1000
        )
        

        """ROI处理"""
        roi_features, valid_mask = self.ROI_Pooling(FeatureMap, proposals, output_size=7)

        """检测头阶段"""
        cls_scores, reg_preds = self.DetectionHead(roi_features, valid_mask)

        """最终检测结果"""
        detections = self.proposal_decoder(proposals, reg_preds)









