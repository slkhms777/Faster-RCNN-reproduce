faster_rcnn/
├── configs/                  # 配置文件
│   └── faster_rcnn.yaml      # 超参数配置（如学习率、锚框参数等）
├── data/                     # 数据相关
│   ├── datasets/             # 数据集处理
│   │   ├── coco.py           # COCO 数据集加载
│   │   └── voc.py            # VOC 数据集加载
│   ├── transforms.py         # 数据增强和预处理
│   └── utils.py              # 数据工具（如数据加载器封装）
├── models/                   # 模型定义
│   ├── backbone/             # 骨干网络
│   │   ├── resnet.py         # ResNet 骨干网络
│   │   └── vgg.py            # VGG 骨干网络
│   ├── rpn/                  # 区域建议网络（RPN）
│   │   ├── anchor_generator.py  # 锚框生成
│   │   ├── rpn_head.py       # RPN 分类和回归头
│   │   └── proposal_layer.py # 生成候选区域（Proposals）
│   ├── roi_heads/            # ROI 处理
│   │   ├── roi_pooling.py    # ROI Pooling 或 ROI Align
│   │   └── box_head.py       # ROI 分类和回归头
│   └── faster_rcnn.py        # Faster R-CNN 主模型整合
├── utils/                    # 工具函数
│   ├── nms.py                # 非极大值抑制（NMS）
│   ├── box_ops.py            # 边界框操作（如 IoU 计算）
│   └── logger.py             # 日志记录
├── train.py                  # 训练脚本
├── eval.py                   # 评估脚本
└── inference.py              # 推理脚本（单张图像测试）