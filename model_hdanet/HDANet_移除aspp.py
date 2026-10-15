import torch
from torch import nn
import torch.nn.functional as F
import pickle

import model_hdanet.resnet as models
import model_hdanet.vgg as vgg_models

from model_hdanet.FDM_2 import FDM
from model_hdanet.weight import weight
from model_hdanet.MAF import MAF
from model_hdanet.FEM import FEM


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

def get_vgg16_layer(model):
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4


class HDANet(nn.Module):
    def __init__(self, args):
        super(HDANet, self).__init__()
        assert args.classes == 2
        from torch.nn import BatchNorm2d as BatchNorm
        self.layers = args.layers
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.csh_boundary_weight = float(getattr(args, 'csh_boundary_weight', 0.0))
        self.ignore_label = args.ignore_label
        self.last_losses = {}
        self.shot = args.shot
        self.vgg = args.vgg
        self.pretrained = True
        models.BatchNorm = BatchNorm

        if self.vgg:
            print('>>>>>>>>> Using VGG_16 bn <<<<<<<<<')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=self.pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = get_vgg16_layer(vgg16)

        else:
            print('>>>>>>>>> Using ResNet {}<<<<<<<<<'.format(self.layers))
            if self.layers == 50:
                resnet = models.resnet50(pretrained=self.pretrained)
            elif self.layers == 101:
                resnet = models.resnet101(pretrained=self.pretrained)
            else:
                resnet = models.resnet152(pretrained=self.pretrained)

            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1,
                                        resnet.conv2, resnet.bn2, resnet.relu2,
                                        resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256

        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512

        # 高维特征 #
        self.down_support_m = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),

        )
        self.down_query_m = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        # ------------------------------低维特征 --------------------------------#
        self.down_support_l = nn.Sequential(
            nn.Conv2d(128 + 256, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),

        )
        self.down_query_l = nn.Sequential(
            nn.Conv2d(128 + 256, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),

        )
        # ------------------------------ 顶级特征 --------------------------------#
        self.down_support_h = nn.Sequential(
            nn.Conv2d(2048, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),

        )
        self.down_query_h = nn.Sequential(
            nn.Conv2d(2048, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),

        )
        # --- Convolution Segmentation Head --- #
        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(reduce_dim, 2, kernel_size=1, padding=0, bias=False)
        )
        # --- Convolution Segmentation Head --- #
        self.FDM_1 = FDM(reduce_dim)
        self.FDM_2 = FDM(reduce_dim)
        self.FEM = FEM()
        self.weight = weight(shot=self.shot)
        self.maf_h = MAF(reduce_dim)
        self.maf_l = MAF(reduce_dim)
        self.maf_m = MAF(reduce_dim)
        from model_hdanet.g1_spatial_gate import configure_spatial_gate
        configure_spatial_gate(self, args)

    def get_optim(self, args):
        optimizer = torch.optim.SGD(
            [
                {'params': self.down_query_h.parameters()},
                {'params': self.down_support_h.parameters()},
                {'params': self.down_query_m.parameters()},
                {'params': self.down_support_m.parameters()},
                {'params': self.down_query_l.parameters()},
                {'params': self.down_support_l.parameters()},
                {'params': self.maf_m.parameters()},
                {'params': self.maf_h.parameters()},
                {'params': self.maf_l.parameters()},
                {'params': self.cls.parameters()},
                {'params': self.FDM_1.parameters()},
                {'params': self.FDM_2.parameters()},
                {'params': self.weight.parameters()},
            ],
            lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        return optimizer

    def forward(self, x,
                s_x=torch.FloatTensor(4, 1, 3, 200, 200).cuda(),
                s_y=torch.FloatTensor(4, 1, 200, 200).cuda(),
                y=None):

        x_size = x.size()  # [4,3,224,224]
        h = int(x_size[-1])
        w = int(x_size[-2])

        with torch.no_grad():
            query_feat_0 = self.layer0(x)  # [2,64,112,112]
            query_feat_1 = self.layer1(query_feat_0)  # [2,256,56,56]
            query_feat_2 = self.layer2(query_feat_1)  # [2,512,28,28]
            query_feat_3 = self.layer3(query_feat_2)  # [2,1024,14,14]
            query_feat_4 = self.layer4(query_feat_3)  # [4,2048,25,25]

            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2,
                                             size=(query_feat_3.size(2), query_feat_3.size(3)),
                                             mode='bilinear',
                                             align_corners=True)
            else:
                pass

        supp_feat_0_list = []
        supp_feat_1_list = []
        supp_feat_2_list = []
        supp_feat_3_list = []
        supp_feat_4_list = []

        mask_list = []  # n-shot存GT的列表
        # ----- 5 shot 支持集特征提取 start ----- #
        with torch.no_grad():
            for i in range(self.shot):
                supp_gt = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
                mask_list.append(supp_gt)
                supp_feat_0 = self.layer0(s_x[:, 0, :, :, :])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                supp_feat_4 = self.layer4(supp_feat_3)
                supp_feat_0_list.append(supp_feat_0)
                supp_feat_1_list.append(supp_feat_1)
                supp_feat_2_list.append(supp_feat_2)
                supp_feat_3_list.append(supp_feat_3)
                supp_feat_4_list.append(supp_feat_4)

        # >>>>> 先验信息获取 start <<<<< #
        fts_size = query_feat_2.size()[-2:]
        weight_soft = self.weight(query=query_feat_3, support=supp_feat_3_list)
        prior_m = self.FEM(query_feat_3, supp_feat_3_list, mask_list, fts_size, weight_soft)  # 高分辨率
        prior_l = F.interpolate(prior_m, size=(query_feat_1.size()[-2:]), mode='bilinear', align_corners=True)
        prior_h = F.interpolate(prior_m, size=(query_feat_4.size()[-2:]), mode='bilinear', align_corners=True)
        # fg_list : [1,1,25,25]

        # >>>> 低 - 维 - 特 - 征 <<<< #
        query_feat_l = torch.cat([query_feat_0, query_feat_1], dim=1)
        query_feat_l = self.down_query_l(query_feat_l)  # [4,256,25,25]
        prototype_l_list = []
        for i in range(self.shot):
            supp_feat_l = torch.cat([supp_feat_0_list[i], supp_feat_1_list[i]], dim=1)  # cat
            supp_feat_l = self.down_support_l(supp_feat_l)  # squeeze
            supp_gt_l = F.interpolate(mask_list[i], size=(supp_feat_0.size()[-2:]), mode='bilinear', align_corners=True)
            prototype_l = Weighted_GAP(supp_feat_l, supp_gt_l)
            prototype_l_list.append(prototype_l)

        prototype_l = torch.cat(prototype_l_list, 2)  # [bs, 256, shot, 1]
        prototype_l = (weight_soft.permute(0, 2, 1, 3) * prototype_l).sum(2, True)
        support_feat_l = prototype_l.expand_as(query_feat_l)

        # >>>> 高 - 维 - 特 - 征 <<<< #
        query_feat_m = torch.cat([query_feat_2, query_feat_3], dim=1)
        query_feat_m = self.down_query_m(query_feat_m)  # [4,256,25,25]
        prototype_m_list = []
        for i in range(self.shot):
            supp_feat_m = torch.cat([supp_feat_2_list[i], supp_feat_3_list[i]], dim=1)  # cat
            supp_feat_m = self.down_support_m(supp_feat_m)  # squeeze
            supp_gt_m = F.interpolate(mask_list[i], size=(supp_feat_2.size(2), supp_feat_2.size(3)), mode='bilinear',
                                      align_corners=True)
            prototype_m = Weighted_GAP(supp_feat_m, supp_gt_m)
            prototype_m_list.append(prototype_m)
        prototype_m = torch.cat(prototype_m_list, 2)  # [bs, 256, shot, 1]
        prototype_m = (weight_soft.permute(0, 2, 1, 3) * prototype_m).sum(2, True)
        support_feat_m = prototype_m.expand_as(query_feat_m)

        # --- High-level ---#
        query_feat_h = self.down_query_h(query_feat_4)
        prototype_h_list = []
        for i in range(self.shot):
            supp_feat_h = self.down_support_h(supp_feat_4_list[i])
            supp_gt_h = F.interpolate(mask_list[i], size=(supp_feat_4.size()[-2:]), mode='bilinear', align_corners=True)
            prototype_h = Weighted_GAP(supp_feat_h, supp_gt_h)
            prototype_h_list.append(prototype_h)
        prototype_h = torch.cat(prototype_h_list, 2)  # [bs, 256, shot, 1]
        prototype_h = (weight_soft.permute(0, 2, 1, 3) * prototype_h).sum(2, True)
        support_feat_h = prototype_h.expand_as(query_feat_h)

        # --- Activation ---#
        fuse_m = self.maf_m(support_feat_m, query_feat_m, prior_m)
        fuse_l = self.maf_l(support_feat_l, query_feat_l, prior_l)
        fuse_h = self.maf_h(support_feat_h, query_feat_h, prior_h)

        # ---- Decoding --- #
        out = self.FDM_1(fuse_m, fuse_h)
        out = F.interpolate(out, size=fuse_l.size()[-2:], mode='bilinear', align_corners=True)  # [1,256,200,200]
        out = self.FDM_2(out, fuse_l)
        out = self.cls(out)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)  # [1,256,200,200]

        if self.training:
            main_loss = self.criterion(out, y.long())
            if self.csh_boundary_weight:
                from model_hdanet.csh_boundary_loss import csh_boundary_loss
                boundary = csh_boundary_loss(out, y, self.ignore_label)
            else:
                boundary = out.sum() * 0
            self.last_losses = {'seg': float(main_loss.detach()),
                                'edge': 0.0,
                                'csh_boundary': float(boundary.detach())}
            main_loss = main_loss + self.csh_boundary_weight * boundary

            return out.max(1)[1], main_loss
        else:
            # 需要输出预测图
            # pred_mask_save = torch.argmax(out[0].squeeze(0).permute(1, 2, 0), axis=-1).detach().cpu().numpy()
            # pred_mask_save[pred_mask_save != 0] = 255
            # pred_mask_save[pred_mask_save == 0] = 0
            # return out, pred_mask_save
            return out
