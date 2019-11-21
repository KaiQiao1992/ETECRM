#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 16:32:29 2019

@author: amax
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import leastsq




def sparseMatrix(height, width, sparsRate):
    d = np.random.randint(0, 100, [height,width])
    m = np.zeros((height, width))
    threshold = sparsRate*100.0
    m[(d<threshold)] = 1
    m = torch.Tensor(m)
    return m


def sparseTensor(c, h, w, n, sparsRate):
    d = np.random.randint(0, 100, [c, n])
    m = np.zeros((c, n, h, w))
    threshold = sparsRate*100
    m[(d<threshold), :, :] = 1
    m = np.reshape(np.transpose(m, (0,3,2,1)), (-1, n))
    m = torch.Tensor(m)
    return m


def sparseChannel(c, h, w, n, c1):
    #c1 channels for encoding
    threshold = np.transpose(np.tile(np.arange(c), (n,1)), (1,0))
    for i in range(n):
        np.random.shuffle(threshold[:,i])
    m = np.zeros((c, n, h, w))
    m[(c1>threshold), :, :] = 1
    m = np.reshape(np.transpose(m, (0,3,2,1)), (-1, n))
    m = torch.Tensor(m)
    return m


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=1294):
        super(ResNet, self).__init__()
        self.inchannel = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inchannel),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 16, 1, stride=2)
        self.layer2 = self.make_layer(ResidualBlock, 16, 1, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 16, 1, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 16, 1, stride=2)
        self.fc = nn.Linear(16*8*8, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def normalize(x):
    x = (x-np.min(x))/(np.max(x)-np.min(x))
    return x


def func(p,x):
    k, b = p
    k = np.tile(k, (100,1))
    return k*x + b

def error(p,x,y):
    return func(p,x)-y


if __name__ == '__main__':
    m = sparseTensor(16, 4, 4, 10, 0.1)
    print(m.numpy())
    print(m.sum())
    
    x = np.random.rand(10,100)
    y = np.random.rand(10,100)

    k = np.zeros(10)
    b = np.zeros(10)
    p0 = [k,b]
    
    #把error函数中除了p0以外的参数打包到args中(使用要求)
    Para = leastsq(error, p0, args=(x,y))
    
    #读取结果
    k, b = Para[0]
    print("k=",k,"b=",b)
    print("cost："+str(Para[1]))
    print("求解的拟合直线为:")
    print("y="+str(round(k,2))+"x+"+str(round(b,2)))

































    