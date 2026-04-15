# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class LeNet(nn.Module):
    # 子模块创建
    def __init__(self, classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    # 子模块拼接
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()


class LeNetSequetial(nn.Module):
    def __init__(self, classes):
        super(LeNetSequetial, self).__init__()
        self.features = nn.Sequential( # 输入 [16, 3, 32, 32]
            nn.Conv2d(3, 6, 5), #6个不同的卷积核，每个核为3通道核大小为5；[16, 6, 28, 28]
            nn.ReLU(), # [16, 6, 28, 28]
            nn.MaxPool2d(2, 2), # [16, 6, 14, 14]
            nn.Conv2d(6, 16, 5), # [16, 16, 10, 10]
            nn.ReLU(), # [16, 16, 10, 10]
            nn.MaxPool2d(2, 2) # [16, 16, 5, 5]
        )
        self.classifier = nn.Sequential( # 输入 [16, 400]
            nn.Linear(16*5*5, 120), # [16, 120]
            nn.ReLU(), # [16, 120]
            nn.Linear(120, 84), # [16, 84]
            nn.ReLU(), # [16, 84]
            nn.Linear(84, classes) # [16, classes]
        )

    def forward(self, x):
        x = self.features(x) # [16, 16, 5, 5] [batch, 通道数, 高, 宽]
        x = x.view(x.size()[0], -1) # [16, 16*5*5] [batch, 特征数 = 通道数*高*宽]
        x = self.classifier(x) # [16, classes] 具体数值含义为属于哪个类别的概率
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()

class LeNetSequentialOrderDict(nn.Module):
    def __init__(self, classes):
        super(LeNetSequentialOrderDict, self).__init__()

        self.features = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(3, 6, 5),
            'relu1': nn.ReLU(inplace=True),
            'pool1': nn.MaxPool2d(kernel_size=2, stride=2),

            'conv2': nn.Conv2d(6, 16, 5),
            'relu2': nn.ReLU(inplace=True),
            'pool2': nn.MaxPool2d(kernel_size=2, stride=2),
        }))

        self.classifier = nn.Sequential(OrderedDict({
            'fc1': nn.Linear(16*5*5, 120),
            'relu3': nn.ReLU(),

            'fc2': nn.Linear(120, 84),
            'relu4': nn.ReLU(inplace=True),

            'fc3': nn.Linear(84, classes),
        }))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()


