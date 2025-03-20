import os
import xml.etree.ElementTree as ET
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from typing import List, Dict, Tuple, Optional

# VOC类别名称
VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
)

# 类别名称到索引的映射
VOC_CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(VOC_CLASSES)}


def parse_voc_annotation(annotation_path: str) -> Dict:
    """解析VOC XML标注文件
    
    Args:
        annotation_path: XML文件的路径
        
    Returns:
        包含边界框和类别信息的字典
    """
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    
    boxes = []
    labels = []
    difficult = []
    
    for obj in root.findall('object'):
        # 获取类别
        class_name = obj.find('name').text
        if class_name not in VOC_CLASS_TO_IDX:
            continue
        labels.append(VOC_CLASS_TO_IDX[class_name])
        
        # 是否是difficult样本
        is_difficult = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
        difficult.append(is_difficult)
        
        # 获取边界框坐标 [xmin, ymin, xmax, ymax]
        bbox = obj.find('bndbox')
        boxes.append([
            float(bbox.find('xmin').text),
            float(bbox.find('ymin').text),
            float(bbox.find('xmax').text),
            float(bbox.find('ymax').text)
        ])
    
    return {
        'boxes': torch.FloatTensor(boxes),
        'labels': torch.LongTensor(labels),
        'difficult': torch.BoolTensor(difficult),
        'image_id': root.find('filename').text
    }


class VOCDataset(data.Dataset):
    """PASCAL VOC数据集"""
    
    def __init__(
        self,
        root: str,
        year: str = '2007',
        image_set: str = 'trainval',
        transform: Optional[transforms.Compose] = None,
        keep_difficult: bool = False
    ):
        """
        Args:
            root: 数据集根目录，如 "/path/to/VOCdevkit/"
            year: 年份 ('2007', '2012')
            image_set: 数据集类型 ('train', 'val', 'trainval', 'test')
            transform: 数据转换及增强
            keep_difficult: 是否保留标记为difficult的样本
        """
        self.root = root
        self.year = year
        self.image_set = image_set
        self.transform = transform
        self.keep_difficult = keep_difficult
        
        self.voc_root = os.path.join(root, f'VOC{year}')
        self.image_dir = os.path.join(self.voc_root, 'JPEGImages')
        self.annotation_dir = os.path.join(self.voc_root, 'Annotations')
        
        # 加载数据集索引文件
        image_sets_file = os.path.join(
            self.voc_root, 'ImageSets', 'Main', f'{image_set}.txt'
        )
        with open(image_sets_file) as f:
            self.ids = [line.strip() for line in f.readlines()]
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict]:
        """获取单个数据样本
        
        Args:
            index: 索引
            
        Returns:
            图像张量和标注信息字典
        """
        img_id = self.ids[index]
        
        # 加载图像
        img_path = os.path.join(self.image_dir, f'{img_id}.jpg')
        img = Image.open(img_path).convert('RGB')
        
        # 加载标注
        anno_path = os.path.join(self.annotation_dir, f'{img_id}.xml')
        target = parse_voc_annotation(anno_path)
        
        # 是否过滤difficult样本
        if not self.keep_difficult:
            boxes = target['boxes']
            labels = target['labels']
            difficult = target['difficult']
            
            keep = ~difficult
            target['boxes'] = boxes[keep]
            target['labels'] = labels[keep]
            target['difficult'] = difficult[keep]
        
        # 应用变换
        if self.transform is not None:
            img, target = self.transform(img, target)
        
        return img, target
    
    def __len__(self) -> int:
        return len(self.ids)


def get_voc_dataset(
    data_dir: str,
    split: str = 'trainval',
    year: str = '2007',
    transform=None,
    keep_difficult: bool = False
) -> VOCDataset:
    """创建VOC数据集
    
    Args:
        data_dir: VOCdevkit数据集根目录
        split: 数据集分割 ('train', 'val', 'trainval', 'test')
        year: 年份 ('2007', '2012')
        transform: 数据转换
        keep_difficult: 是否保留困难样本
    
    Returns:
        VOCDataset实例
    """
    return VOCDataset(
        root=data_dir,
        year=year,
        image_set=split,
        transform=transform,
        keep_difficult=keep_difficult
    )


# 常用的数据变换
class Compose:
    """组合多个变换操作"""
    
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """将PIL图像转换为张量"""
    
    def __call__(self, image, target):
        image = transforms.functional.to_tensor(image)
        return image, target


class Normalize:
    """标准化图像"""
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, image, target):
        image = transforms.functional.normalize(image, self.mean, self.std)
        return image, target


class Resize:
    """调整图像大小，同时调整边界框"""
    
    def __init__(self, size):
        self.size = size  # (h, w)
        
    def __call__(self, image, target):
        orig_width, orig_height = image.size
        image = transforms.functional.resize(image, self.size)
        
        # 调整边界框坐标
        if "boxes" in target:
            boxes = target["boxes"]
            scale_x = self.size[1] / orig_width
            scale_y = self.size[0] / orig_height
            
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            
            target["boxes"] = boxes
            
        return image, target
    

# 示例用法
import torch
from torch.utils.data import DataLoader

# 定义数据转换
transform = Compose([
    Resize((608, 800)),  # 先调整图像大小（处理PIL图像）
    ToTensor(),          # 然后转换为张量
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建训练数据集
train_dataset = get_voc_dataset(
    data_dir="/Users/gaojiaxuan/Proj/Data/VOCdevkit",
    split="trainval",
    year="2012",
    transform=transform
)

# 创建数据加载器
train_loader = DataLoader(
    train_dataset, 
    batch_size=2,
    shuffle=True,
    collate_fn=lambda batch: tuple(zip(*batch))  # 自定义收集函数处理不同大小的图像和标注
)

# 查看一个批次
images, targets = next(iter(train_loader))

print(f"图像批次形状: {[img.shape for img in images]}")
print(f"第一张图像标签: {targets[0]['labels']}")
print(f"第一张图像边界框: {targets[0]['boxes']}")