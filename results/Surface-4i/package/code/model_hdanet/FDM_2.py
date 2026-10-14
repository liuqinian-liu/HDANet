import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init

class CR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size,padding=padding, dilation=dilation, bias=False, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.act = act
    def forward(self, x):
        out = self.conv(x)
        if self.act == True:
            out = self.relu(out)
        return out

class channel_attention(nn.Module):
    def __init__(self, channel=256, ratio=8):
        super(channel_attention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.size()
        max_pool_out = self.max_pool(x).view([b, c])
        avg_pool_out = self.avg_pool(x).view([b, c])
        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)
        out = max_fc_out + avg_fc_out
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x

class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()
        padding = 7 // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool_out,_ = torch.max(x, dim=1, keepdim=True)  # 通道最大池化
        mean_pool_out = torch.mean(x, dim=1, keepdim=True)  # 通道平均池化
        pool_out = torch.cat([max_pool_out, mean_pool_out], dim=1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)
        return out * x

class Fusion(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.c1 = CR(in_c * 2 , in_c, kernel_size=1, padding=0)
        self.c2 = CR(in_c, in_c, act=False)
        self.c3 = CR(in_c, in_c, act=False)
        self.c4 = CR(in_c, in_c, kernel_size=1, padding=0, act=False)
        self.ca = channel_attention(in_c,ratio=16)
        self.sa = spatial_attention()

    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        x = self.c1(x)
        s1 = x
        x = self.c2(x)
        x = self.relu(x + s1)
        s2 = x
        x = self.c3(x)
        x = self.relu(x + s2 + s1)
        s3 = x
        x = self.c4(x)
        x = self.relu(x + s3 + s2 + s1)
        x = self.ca(x)
        x = self.sa(x)

        return x

class AGCA(nn.Module):
    def __init__(self, in_ch, ratio):
        super(AGCA, self).__init__()
        hide_ch = in_ch // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_ch, hide_ch, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(2)
        self.A0 = torch.eye(hide_ch).to('cuda')  # 创建单位矩阵
        self.A2 = nn.Parameter(torch.FloatTensor(torch.zeros((hide_ch, hide_ch))), requires_grad=True)
        init.constant_(self.A2, 1e-6)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(hide_ch, in_ch, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # [2,64,1,1]
        y = self.conv1(y)   # [2,16,1,1]
        B, C, _, _ = y.size()
        y = y.flatten(2).transpose(1, 2)
        A1 = self.softmax(self.conv2(y))
        A1 = A1.expand(B, C, C)
        A = (self.A0 * A1) + self.A2  # [2,16,16]
        y = torch.matmul(y, A)
        y = self.relu(self.conv3(y))
        y = y.transpose(1, 2).view(-1, C, 1, 1) # [1，16,1,1]
        y = self.sigmoid(self.conv4(y))     # [1，64,1,1]

        return x * y
class ASPP(nn.Module):
    def __init__(self, in_ch=256):
        super(ASPP, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=6, dilation=6, bias=True),
            nn.ReLU(),)
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=12, dilation=12, bias=True),
            nn.ReLU(),)
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=18, dilation=18, bias=True),
            nn.ReLU(),)
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_ch*5, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
        )

        self.agca = AGCA(in_ch, 4)

    def forward(self, x):
        feature_size = x.shape[-2:]
        global_feature = F.avg_pool2d(x, kernel_size=feature_size)
        global_feature = self.layer0(global_feature)
        global_feature = global_feature.expand(-1, -1, feature_size[0], feature_size[1])
        out = torch.cat([global_feature, self.layer1(x), self.layer2(x), self.layer3(x),self.layer4(x)], dim = 1)
        out = self.layer5(out)
        out = self.agca(out)+out
        return out

# Fusion Decoding Module
class FDM(nn.Module):
    def __init__(self, in_ch):
        super(FDM,self).__init__()
        self.fuse = Fusion(in_ch)
        self.res1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.res3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        # self.aspp = ASPP(in_ch)

    def forward(self, x1, x2):
        x = self.fuse(x1, x2)
        x = self.res1(x)
        x = self.res2(x) + x
        x = self.res3(x) + x
        # x = self.aspp(x)
        return x
