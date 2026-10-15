import torch
import torch.nn.functional as F
import torch.nn as nn

# Mulit-scale Attention Fusion Module
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

class cbam(nn.Module):
    def __init__(self, channel=256, ratio=16, kernel_size=7):
        super(cbam, self).__init__()
        self.channel_attention = channel_attention(channel, ratio)
        self.spatial_attention = spatial_attention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class MAF(nn.Module):
    def __init__(self,in_ch = 256,ratio=16, kernel_size=7):
        super(MAF, self).__init__()
        out_ch = in_ch
        self.res1= nn.Sequential(
            nn.Conv2d(in_ch*2+2, in_ch, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.res2= nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.cbam = cbam(channel=in_ch, ratio=16, kernel_size=7)
        # self.standard_conv = nn.Conv2d(in_channels=1, out_channels=in_ch, kernel_size=1, padding=0)
        self.conv_1_3 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 3), padding=(0, 1))
        self.conv_3_1 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3, 1), padding=(1, 0))
        self.conv_1_5 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 5), padding=(0, 2))
        self.conv_5_1 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(5, 1), padding=(2, 0))
        self.conv_1_7 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 7), padding=(0, 3))
        self.conv_7_1 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(7, 1), padding=(3, 0))
        self.dilated_conv_3 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, dilation=3, padding=3, groups=4)
        self.dilated_conv_5 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, dilation=5, padding=5, groups=4)  # 空洞率为5的卷积
        self.dilated_conv_7 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, dilation=7, padding=7, groups=4)
    def forward(self, x1,x2,x3):
        x = torch.cat([x1,x2,x3],dim=1)
        x = self.res1(x)
        x = self.res2(x)
        s1 = F.relu(self.cbam(x) + x)
        if hasattr(self, 'spatial_gate'):
            s1 = self.spatial_gate(s1, x3)
        x_size = s1.size()
        b, c, h, w = int(x_size[0]), int(x_size[1]), int(x_size[2]), int(x_size[3])
        branch_3 = self.conv_3_1(self.conv_1_3(s1))
        branch_5 = self.conv_5_1(self.conv_1_5(s1))
        branch_7 = self.conv_7_1(self.conv_1_7(s1))
        s2 = F.relu(branch_3 + branch_5 + branch_7)
        out = self.dilated_conv_3(s2) + self.dilated_conv_5(s2) + self.dilated_conv_7(s2)
        out = F.relu(out)
        return out


if __name__ == '__main__':
    x1 = torch.randn(2, 256, 25, 25)
    x2 = torch.randn(2, 256, 25, 25)
    x3 = torch.randn(2, 2, 25, 25)
    model = MAF(256)
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.4fM" % (total / 1e6))

    out = model(x1,x2,x3)
    print(out.size())