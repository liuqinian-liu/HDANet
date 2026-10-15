import torch
import torch.nn as nn
import torch.nn.functional as F
def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

def cos_sim(query_feat_high, tmp_supp_feat, cosine_eps=1e-7):
    q = query_feat_high.flatten(2).transpose(-2, -1)
    s = tmp_supp_feat.flatten(2).transpose(-2, -1)
    tmp_query = q
    tmp_query = tmp_query.contiguous().permute(0, 2, 1)  # [bs, c, h*w]
    tmp_query_norm = torch.norm(tmp_query, 2, 1, True)
    tmp_supp = s
    tmp_supp = tmp_supp.contiguous()
    tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)
    similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
    return similarity


def generate_prior_proto(query_feat_high, final_supp_list, mask_list, fts_size,weight_soft):
    bsize, ch_sz, sp_sz, _ = query_feat_high.size()[:]
    fg_list = []
    bg_list = []
    cosine_eps = 1e-7
    for i, tmp_supp_feat in enumerate(final_supp_list):
        resize_size = tmp_supp_feat.size(2)
        tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

        max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 可调整 kernel_size 和 padding
        bg = 1 - tmp_mask
        tensor_dilate = -max_pool(-bg) #  对背景区域进行一个腐蚀操作

        fg_supp_feat = Weighted_GAP(tmp_supp_feat, tmp_mask)
        bg_supp_feat = Weighted_GAP(tmp_supp_feat, tensor_dilate)

        fg_sim = cos_sim(query_feat_high, fg_supp_feat, cosine_eps)
        bg_sim = cos_sim(query_feat_high, bg_supp_feat, cosine_eps)

        fg_sim = fg_sim.max(1)[0].view(bsize, sp_sz * sp_sz)
        bg_sim = bg_sim.max(1)[0].view(bsize, sp_sz * sp_sz)

        fg_sim = (fg_sim - fg_sim.min(1)[0].unsqueeze(1)) / (
                fg_sim.max(1)[0].unsqueeze(1) - fg_sim.min(1)[0].unsqueeze(1) + cosine_eps)

        bg_sim = (bg_sim - bg_sim.min(1)[0].unsqueeze(1)) / (
                bg_sim.max(1)[0].unsqueeze(1) - bg_sim.min(1)[0].unsqueeze(1) + cosine_eps)

        fg_sim = fg_sim.view(bsize, 1, sp_sz, sp_sz)
        bg_sim = bg_sim.view(bsize, 1, sp_sz, sp_sz)

        fg_sim = F.interpolate(fg_sim, size=fts_size, mode='bilinear', align_corners=True)
        bg_sim = F.interpolate(bg_sim, size=fts_size, mode='bilinear', align_corners=True)
        fg_list.append(fg_sim)
        bg_list.append(bg_sim)

    fg_corr = torch.cat(fg_list, 1)  # bsize, shots, h, w
    bg_corr = torch.cat(bg_list, 1)
    corr = fg_corr - bg_corr
    corr[corr < 0] = 0

    corr_min_val = torch.min(corr)
    corr_max_val = torch.max(corr)

    corr = (corr-corr_min_val)/(corr_max_val-corr_min_val).clamp_min(1e-7)
    fg_corr = (weight_soft * fg_corr).sum(1, True)
    corr = (weight_soft * corr).sum(1, True)
    prior = torch.cat([fg_corr, corr], dim=1)
    # return corr
    return prior

# --- 特征强化 -- #
class FEM(nn.Module):
    def __init__(self):
        super(FEM, self).__init__()
    def forward(self, query, supp_list, mask_list, fts_size, weight_soft):
        prior = generate_prior_proto(query, supp_list, mask_list, fts_size,weight_soft)
        return prior

if __name__ == '__main__':
    query = torch.randn(3, 2048, 25, 25)  # 输入b c h w
    support1 = torch.randn(3, 2048, 25, 25)  # 输入b c h w
    support2 = torch.randn(3, 2048, 25, 25)  # 输入b c h w
    final_supp_list = []
    final_supp_list.append(support1)
    final_supp_list.append(support2)
    mask_list = []
    mask_list_bg = []
    mask1 = torch.randn(3, 1, 25, 25)  # 输入b c h w
    mask2 = torch.randn(3, 1, 25, 25)  # 输入b c h w
    mask_list.append(mask1)
    mask_list.append(mask2)
    fts_size = query.size()[-2:]

    # fg_corr, bg_corr, corr, corr_bg, fg_sim_maxs, corr_max = generate_prior_proto(query,final_supp_list,mask_list,fts_size)
    #
    # out = cos_similarity(query, support)
    # print(out.size())
    #
    fem = FEM(256)
    out = fem(query, final_supp_list,mask_list,fts_size )

    print(out.size())


