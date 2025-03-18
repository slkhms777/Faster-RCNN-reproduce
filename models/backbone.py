import torch
from torch import nn
import torchvision
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.resnet import ResNet34_Weights
from torchvision.models.resnet import ResNet18_Weights

def get_net(model_name, pretrained=True, output_size='xlarge'):
    """
    获取backbone网络
    
    Args:
        model_name: 使用的模型名称
        pretrained: 是否使用预训练权重
        output_size: 输出特征图大小 - 'small'(7x7)、'medium'(14x14)或'large'(28x28)或'xlarge'(56x56)
    
    Returns:
        特征提取网络
    """
    if model_name == 'resnet18':
        net = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    elif model_name == 'resnet34':
        net = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
    elif model_name == 'resnet50':
        net = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError('Model not available')
    
    # 根据需要的输出尺寸返回不同层次的特征
    if output_size == 'small':  # 7x7特征图
        return nn.Sequential(*list(net.children())[:-2])
    elif output_size == 'medium':  # 14x14特征图
        return nn.Sequential(*list(net.children())[:-3])
    elif output_size == 'large':  # 28x28特征图
        return nn.Sequential(*list(net.children())[:-4])
    elif output_size == 'xlarge':  # 56x56特征图
        return nn.Sequential(*list(net.children())[:-5])
    else:
        raise ValueError('Invalid output size')

