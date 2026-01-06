import torch
import torch.nn as nn
from einops import rearrange

class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        if stride != 1:
            out_channels = in_channels * stride
            self.W = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        else:
            out_channels = in_channels
            self.W = nn.Identity()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size - 1)//2, stride=stride)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size - 1)//2)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x_0):
        x = self.conv1(x_0)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.act(x + self.W(x_0))


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        layers.append(torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3))
        layers.append(torch.nn.BatchNorm2d(64))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.MaxPool2d(kernel_size=3, stride=2))
        for i in range(3):
            layers.append(ResNetBlock(64, 3, 1)) 
        layers.append(ResNetBlock(64, 3, 2)) 
        for i in range(3):
            layers.append(ResNetBlock(128, 3, 1)) 
        layers.append(ResNetBlock(128, 3, 2)) 
        for i in range(5):
            layers.append(ResNetBlock(256, 3, 1)) 
        layers.append(ResNetBlock(256, 3, 2)) 
        for i in range(2):
            layers.append(ResNetBlock(512, 3, 1)) 
       
        layers.append(nn.AvgPool2d(7))

        self.fc = nn.Linear(512, 1000)

        self.stack = nn.Sequential(*layers)


    def forward(self, x):
        x = self.stack(x)
        x = self.fc(rearrange(x, "N C H W -> N (C H W)")) 
        return x
            
class ResNetCIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        layers.append(torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1))
        layers.append(torch.nn.BatchNorm2d(64))
        layers.append(torch.nn.ReLU())
        for i in range(3):
            layers.append(ResNetBlock(64, 3, 1)) 
        layers.append(ResNetBlock(64, 3, 2)) 
        for i in range(3):
            layers.append(ResNetBlock(128, 3, 1)) 
        layers.append(ResNetBlock(128, 3, 2)) 
        for i in range(5):
            layers.append(ResNetBlock(256, 3, 1)) 
        layers.append(ResNetBlock(256, 3, 2)) 
        for i in range(2):
            layers.append(ResNetBlock(512, 3, 1)) 
       
        layers.append(nn.AvgPool2d(4))

        self.fc = nn.Linear(512, 10)

        self.stack = nn.Sequential(*layers)


    def forward(self, x):
        x = self.stack(x)
        x = self.fc(rearrange(x, "N C H W -> N (C H W)")) 
        return x
 
