# Lowest-level 3D CNN (ResNet-18 3D variant)
# models/hnet/encoder.py
import torch
import torch.nn as nn

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out

class LowLevelEncoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, out_ch=256):
        super(LowLevelEncoder, self).__init__()
        self.in_planes = base_ch

        self.conv1 = nn.Conv3d(in_ch, base_ch, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False)
        self.bn1 = nn.BatchNorm3d(base_ch)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock3D, base_ch, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock3D, base_ch * 2, 2, stride=(1,2,2))
        self.layer3 = self._make_layer(BasicBlock3D, base_ch * 4, 2, stride=1)
        self.layer4 = self._make_layer(BasicBlock3D, out_ch, 2, stride=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: B T H W C -> B C T H W
        x = x.permute(0, 4, 1, 2, 3)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out: B C T' H' W' -> B T' H' W' C
        out = out.permute(0, 2, 3, 4, 1)
        return out
