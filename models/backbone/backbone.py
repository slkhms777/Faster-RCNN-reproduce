# filepath: [backbone.py](http://_vscodecontentref_/3)
import torch
from torch import nn
import torchvision
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.resnet import ResNet34_Weights
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.vgg import VGG16_Weights

def get_net(model_name, pretrained=True, output_size='xlarge'):
    """
    获取骨干网络
    
    Args:
        model_name: 模型名称，如'resnet50'、'vgg16'等
        pretrained: 是否使用预训练权重
        output_size: 输出特征图大小
    
    Returns:
        特征提取网络
    """
    # 根据模型名称前缀选择相应的网络构建函数
    if model_name.startswith('resnet'):
        return ResNet(model_name, pretrained, output_size)
    elif model_name.startswith('vgg'):
        return VGG(model_name, pretrained, output_size)
    else:
        raise ValueError(f"不支持的模型: {model_name}")

def ResNet(model_name, pretrained, output_size):
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

def VGG(model_name, pretrained, output_size):
    """
    获取VGG骨干网络
    
    Args:
        model_name: 使用的模型名称
        pretrained: 是否使用预训练权重
        output_size: 输出特征图大小
    
    Returns:
        特征提取网络
    """
    if model_name == 'vgg16':
        net = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"不支持的VGG模型: {model_name}")
    
    # VGG的features部分包含卷积层和池化层
    features = net.features
    
    # 根据需要的输出尺寸返回不同的特征层
    if output_size == 'small':  # 较小特征图
        return nn.Sequential(*list(features)[:24])  # 到pool4
    elif output_size == 'medium':
        return nn.Sequential(*list(features)[:17])  # 到pool3
    elif output_size == 'large':
        return nn.Sequential(*list(features)[:10])  # 到pool2
    elif output_size == 'xlarge':
        return nn.Sequential(*list(features)[:5])   # 到pool1
    else:
        raise ValueError('Invalid output size')

