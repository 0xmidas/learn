from einops.einops import rearrange
import torch
import torch.nn as nn

class MobileNetBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels: int, first_stride: int, second_stride: int):
        super().__init__()
        self.depth_wise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=first_stride, padding=1, groups=in_channels)
        self.batch1 = nn.BatchNorm2d(in_channels)
        self.a = nn.ReLU()
        self.single_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=second_stride, padding=0)
        self.batch2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depth_wise_conv(x)
        x = self.batch1(x)
        x = self.a(x)
        x = self.single_conv(x)
        x = self.batch2(x)
        return self.a(x)


class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        
        # 1. 3x3  
        layers.append(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(MobileNetBlock(32, 64, 1, 1))
        layers.append(MobileNetBlock(64, 128, 2, 1))
        layers.append(MobileNetBlock(128, 128, 1, 1))
        layers.append(MobileNetBlock(128, 256, 2, 1))
        layers.append(MobileNetBlock(256, 256, 1, 1))
        layers.append(MobileNetBlock(256, 512, 2, 1))
        for i in range(5):
            layers.append(MobileNetBlock(512, 512, 1, 1))
        layers.append(MobileNetBlock(512, 1024, 2, 1))
        layers.append(MobileNetBlock(1024, 1024, 1, 1))
        layers.append(nn.AvgPool2d(7))

        self.stack = nn.ModuleList(layers)

        self.fc = nn.Linear(1024, 1000) 


    def forward(self, x):
        for module in self.stack:
            x = module(x)
        x = rearrange(x, "b c h w -> b (c h w)")
        return self.fc(x)


class MobileNetCifar(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        
        # 1. 3x3  
        layers.append(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(MobileNetBlock(32, 64, 2, 1))
        layers.append(MobileNetBlock(64, 128, 1, 1))
        layers.append(MobileNetBlock(128, 256, 2, 1))
        layers.append(MobileNetBlock(256, 512, 1, 1))
        for i in range(1):
            layers.append(MobileNetBlock(512, 512, 1, 1))
        layers.append(MobileNetBlock(512, 1024, 1, 1))
        layers.append(nn.AvgPool2d(4))

        self.stack = nn.ModuleList(layers)

        self.fc = nn.Linear(1024, 10) 


    def forward(self, x):
        for module in self.stack:
            x = module(x)
        x = rearrange(x, "b c h w -> b (c h w)")
        return self.fc(x)


        

