import torch
import torch.nn as nn
class weight(nn.Module):
    def __init__(self, kshot_trans_dim=2, shot=5):
        super(weight, self).__init__()
        self.kshot_trans_dim = kshot_trans_dim
        self.shot = shot
        if self.shot > 1:
            self.kshot_rw = nn.Sequential(
                nn.Conv2d(self.shot, self.kshot_trans_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.kshot_trans_dim, self.shot, kernel_size=1))
    def get_gram_matrix(self,x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w)  # C*N
        x_T = x.permute(0, 2, 1)  # N*C
        x_norm = x.norm(2, 2, True)
        x_T_norm = x_T.norm(2, 1, True)
        gram = torch.bmm(x, x_T) / (torch.bmm(x_norm, x_T_norm) + 1e-7)  # C*C
        return gram

    def forward(self,query, support):
        b, c, h, w = query.shape
        # 查询集 gram矩阵
        que_gram = self.get_gram_matrix(query)  # [bs, C, C] in (0,1)
        norm_max = torch.ones_like(que_gram).norm(dim=(1, 2))
        est_val_list = []
        for supp_item in support:
            supp_gram = self.get_gram_matrix(supp_item)  # [2,256,256]
            gram_diff = que_gram - supp_gram  # [2,256,256]
            est_val_list.append((gram_diff.norm(dim=(1, 2)) / norm_max).reshape(b, 1, 1, 1))  # norm2
        est_val_total = torch.cat(est_val_list, 1)  # [2, 5, 1, 1]
        if self.shot > 1:
            val1, idx1 = est_val_total.sort(1)
            val2, idx2 = idx1.sort(1)
            weight = self.kshot_rw(val1)
            weight = weight.gather(1, idx2)
            weight_soft = torch.softmax(weight, 1)
        else:
            weight_soft = torch.ones_like(est_val_total)

        return weight_soft

if __name__ == '__main__':
    query_feat_high = torch.ones(2,256,25,25)
    supp_feat_list = []
    supp_pro_list = []
    shot = 1
    for i in range(shot):
        print('this is the {} feature'.format(i+1))
        supp_feat = torch.randn(2, 256, 25, 25)
        supp_feat_list.append(supp_feat)
        print(supp_feat.shape)
        # if i ==4:
        #     # print(len(supp_feat_list))
    for i in range(shot):
        print('this is the {} prototype'.format(i+1))
        supp_pro = torch.randn(2, 256, 1, 1)
        supp_pro_list.append(supp_pro)
        print(supp_pro.shape)
        # if i ==4:
        #     # print(len(supp_pro_list))

    shot = weight(shot=shot)
    out = shot(query_feat_high,supp_feat_list)
    print(out.shape)
