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
        """"""
        self.Backbone = backbone.get_net('resnet50', pretrained=True, output_size='small')
        
        self.anchor_generator = anchor_generator.generate_anchors(
            sizes=[128, 256, 512],
            aspect_ratios=[0.5, 1, 2]
        )

        self.RPN = rpn_head.RPN(
            in_channels=2048,
            num_anchors=len(self.anchor_generator)
        )

        self.anchor_decoder = proposal_layer.decode_anchors #参数: anchors, rpn_reg_preds
        self.anchor_clipper = proposal_layer.clip_boxes(image_size=image_shape[-2:]) #参数: proposals, image_shape
        self.anchor_filter = proposal_layer.filter_proposals #参数: proposals, foreground_probs, nms_threshold=0.7, top_n=1000
        
        self.ROI_Pooling = roi_pooling.ROI_Pooling(
            output_size=7
        )
        self.DetectionHead = box_heads.DetectionHead(
            num_classes=num_classes
        )
        self.proposal_decoder = box_heads.decode_proposals
    
    def forward(self, x):
        """"""
        FeatureMap = self.Backbone(x)
        # FeatureMap: torch.Size([batch_size, 2048, 25, 19])

        anchors = self.anchor_generator(FeatureMap)

        rpn_cls_scores, rpn_reg_preds = self.RPN(FeatureMap)
        # rpn_cls_scores: torch.Size([batch_size, 18, 25, 19])
        # rpn_reg_preds: torch.Size([batch_size, 36, 25, 19])

        rpn_cls_probs = torch.softmax(rpn_cls_scores.view(x.shape[0], 2, -1), 2, 9, 25, 19, dim=1)

        proposals = self.anchor_filter(self.anchor_clipper(self.anchor_decoder(anchors, self.RPN(FeatureMap)[1]), x.shape[-2:]), FeatureMap, nms_threshold=0.7, top_n=1000)
        # proposals: torch.Size([batch_size, top_n, 4])


