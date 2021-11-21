import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsamplerBlock(nn.Module):
    def __init__(self, in_channels=32, out_channels=32):
        super(DownsamplerBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels-in_channels, kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input_tensor):
        x = torch.cat([self.conv(input_tensor), self.maxpool(input_tensor)], dim=1)
        x = self.bn(x)
        return F.relu(x)

class NB1D(nn.Module):
    def __init__(self, channels=32, dilation=1):
        super(NB1D, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 1), stride=1,
                               padding=(1, 0))
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1, 3), stride=1,
                               padding=(0, 1))

        self.bn1 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 1), stride=1,
                               padding=(1*dilation, 0), dilation=dilation)
        self.conv4 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1, 3), stride=1,
                               padding=(0, 1*dilation), dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, input_tensor):
        x = self.conv1(input_tensor)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn2(x)
        return F.relu(x+input_tensor)

class ResNet50NB1D(nn.Module):
    def __init__(self, color_channel=3, num_classes=2):
        super(ResNet50NB1D, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=color_channel, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = nn.Sequential(
            NB1D(channels=64),
            NB1D(channels=64),
            NB1D(channels=64)
        )
        self.conv3_x = nn.Sequential(
            DownsamplerBlock(in_channels=64, out_channels=128),
            NB1D(channels=128),
            NB1D(channels=128),
            # NB1D(channels=128),
            # NB1D(channels=128)
        )
        self.conv4_x = nn.Sequential(
            DownsamplerBlock(in_channels=128, out_channels=256),
            NB1D(channels=256),
            NB1D(channels=256),
            # NB1D(channels=256),
            # NB1D(channels=256),
            # NB1D(channels=256),
            # NB1D(channels=256)
        )
        self.conv5_x = nn.Sequential(
            DownsamplerBlock(in_channels=256, out_channels=512),
            NB1D(channels=512),
            NB1D(channels=512),
            # NB1D(channels=512),
        )
        
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
