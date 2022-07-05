from tkinter.tix import Tree
from numpy import pad
from pyrsistent import inc
from regex import F
from sympy import inverse_laplace_transform
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    # 主分支的卷积个数的倍数
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=stride,
                                padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample #下采样参数，虚线的残差结构

    def forward(self, x):
        identity = x
        if self.downsample is not None: # 虚线的残差
            identity = self.downsample(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        out = self.relu(x)
        return out

class Bottleneck(nn.Module):

    expansion = 4
    
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=stride,
                                padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels,
                                out_channels=in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion())
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x += identity
        out = self.relu(x)

        return out


class ResNet(nn.Module):
    def __init__(self, in_channels, block: BasicBlock, block_list, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=self.in_channels,
                                kernel_size=7,
                                stride=2,
                                padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer_1 = self.make_layer(block, 64, block_list[0])
        self.layer_2 = self.make_layer(block, 128, block_list[1])
        self.layer_3 = self.make_layer(block, 256, block_list[2], stride=2)
        self.layer_4 = self.make_layer(block, 512, block_list[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Flatten(),nn.Linear(512 * block.expansion, num_classes))
        nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
    def make_layer(self, block: BasicBlock, channels, block_list, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels*block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(channels*block.expansion))
        layers = []
        layers.append(block(self.in_channels, channels, downsample=downsample, stride=stride))
        self.in_channels = channels * block.expansion

        for _ in range(1, block_list):
            layers.append(block(self.in_channels, channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)

        x = self.avgpool(x)
        x = self.fc(x)
        
        return x

def ResNet18(in_channels=3, num_classes=1000):
    return ResNet(in_channels, BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
def ResNet34(in_channels=3, num_classes=1000):
    return ResNet(in_channels, BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
def ResNet50(in_channels=3, num_classes=1000):
    return ResNet(in_channels, Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
def ResNet101(in_channels=3, num_classes=1000):
    return ResNet(in_channels, Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
def ResNet152(in_channels=3, num_classes=1000):
    return ResNet(in_channels, Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

# 测试

# if __name__ == '__main__':
#     x = torch.randn((1, 1, 28, 28))
#     model = ResNet18(in_channels=1, num_classes=10)
#     # print(model)
#     for layer in model.children():
#         x = layer(x)
#         print(layer.__class__.__name__, "shape", x.shape)
#     # y = model(x)