import os
import os.path
import numpy as np
import random
import cv2

from tqdm import tqdm
import torch
from torch.utils import data
from torch.utils.data import Dataset


def make_dataset(split=0, data_root=None, data_list=None, train_class=None):
    assert split in [0, 1, 2, 999]

    # 判断数据列表是否存在
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    img_gt_class_list = []  # 缺陷-GT-类别列表
    list_read = open(data_list, encoding='UTF-8-sig').readlines()  # 读取文件全部行
    print("数据-{}-成功读取......".format(train_class))
    class_img_gt_dict = {}  # 类：缺陷，GT 字典

    for sub_c in train_class:
        class_img_gt_dict[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):  # 共有380数据
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split()
        image_name = os.path.join(data_root, line_split[0])  # img存储路径
        temp = line_split[0].replace('Images/', 'GT/')  # 将img名中的Images修改为GT名
        gt_name = os.path.join(data_root, temp)  # gt存储路径
        gt_name = gt_name.replace('jpg', 'png')  # gt修改为.png格式
        image_class = line_split[1]  # 得到类别
        item = (image_name, gt_name, image_class)  # img_gt_class 缺陷，gt，缺陷类别

        img_gt_class_list.append(item)

        class_img_gt_dict[int(image_class)].append(item)

    print("Checking image&label pair {} list done! ".format(split))
    return img_gt_class_list, class_img_gt_dict  # 返回原图-GT-类别列表；键：原图-GT-类别列表字典


# DATA:
# data_root:    用绝对路径
# train_list:   用绝对路径
# val_list:     用绝对路径
# classes: 2
class SemData(Dataset):
    def __init__(self, split=0, shot=1, data_root=None, data_list=None, transform=None, mode='train'):
        assert mode in ['train', 'val', 'test']
        assert split in [0, 1, 2]
        self.mode = mode
        self.split = split
        self.shot = shot
        self.data_root = data_root
        self.data_list = data_list
        # print('split:', self.split)
        # print('data_root', data_root)
        # print('data_list', data_list)
        # 获取训练类和测试类 1 2 3 4 | 5 6 7 8 | 9 10 11 12
        # split = 2
        # 1 2 3 4 | 5 6 7 8用于训练     # 9 10 11 12 用于验证
        # split = 1
        # 1 2 3 4 | 9 10 11 12用于训练   # 5 6 7 8 用于验证
        # split = 0
        # 5 6 7 8 | 9 10 11 12用于训练  # 1 2 3 4 用于验证
        self.class_list = list(range(1, 13))  # 缺陷类别总数
        if self.split == 2:
            self.train_class = list(range(1, 9))
            self.val_class = list(range(9, 13))
        elif self.split == 1:
            self.train_class = list(range(1, 5)) + list(range(9, 13))
            self.val_class = list(range(5, 9))
        elif self.split == 0:
            self.train_class = list(range(5, 13))
            self.val_class = list(range(1, 5))

        # 打印训练类别列表
        # print('读取训练train_class: ', self.train_class)
        # print('读取验证val_class: ', self.val_class)

        if self.mode == 'train':
            self.img_gt_class_list, self.class_img_gt_dict = make_dataset(split, data_root, data_list, self.train_class)
            assert len(self.class_img_gt_dict.keys()) == len(self.train_class)

        elif self.mode == 'val':
            self.img_gt_class_list, self.class_img_gt_dict = make_dataset(split, data_root, data_list, self.val_class)
            assert len(self.class_img_gt_dict.keys()) == len(self.val_class)

        self.transform = transform
        # print('已加载数据增强方法...')

    def __len__(self):
        return len(self.img_gt_class_list)  # 全部训练样本数

    def __getitem__(self, index):
        # 读入image和mask
        query_img, query_gt, query_class = self.img_gt_class_list[index]  # 查询集原图路径
        # --- 查询集缺陷--- #
        query_rgb = cv2.imread(query_img, cv2.IMREAD_COLOR)
        query_rgb = cv2.cvtColor(query_rgb, cv2.COLOR_BGR2RGB)
        query_rgb = np.float32(query_rgb)
        # --- 查询集mask--- #
        query_mask = cv2.imread(query_gt, cv2.IMREAD_GRAYSCALE)
        query_mask[query_mask != 255] = 0
        query_mask[query_mask == 255] = 1
        # 验证查询集 缺陷 和 mask 尺寸匹配
        if query_rgb.shape[0] != query_mask.shape[0] or query_rgb.shape[1] != query_mask.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + query_img + " " + query_gt + "\n"))

        class_chosen = query_class
        # 通过字典类别（键），获得该类别 全部img，mask列表
        # 该列表中包括查询集/支持集全部的缺陷和gt
        all_img_gt_list = self.class_img_gt_dict[int(class_chosen)]
        num_file = len(all_img_gt_list)

        # ---- 适用于k-shot ---- #
        support_image_path_list = []  # 支持集缺陷-路径列表
        support_gt_path_list = []  # 支持集gt-路径列表
        support_idx_list = []  # 支持集索引列表

        for k in range(self.shot):
            support_idx = random.randint(1, num_file) - 1  # 获取支持集索引
            support_image_path = query_img  # 支持集缺陷路径与查询集缺陷路径相同
            support_label_path = query_gt  # 支持集gt路径与查询集gt路径相同
            while (
                    support_image_path == query_img and support_label_path == query_gt) or support_idx in support_idx_list:
                support_idx = random.randint(1, num_file) - 1
                support_image_path, support_label_path, _ = all_img_gt_list[support_idx]

            support_idx_list.append(support_idx)  # 支持集索引列表
            support_image_path_list.append(support_image_path)
            support_gt_path_list.append(support_label_path)

        support_image_list = []  # 支持集 缺陷 列表
        support_label_list = []  # 支持集 gt 列表
        subcls_list = []

        for k in range(self.shot):
            if self.mode == 'train':
                subcls_list.append(self.train_class.index(int(class_chosen)))
            else:
                subcls_list.append(self.val_class.index(int(class_chosen)))

            support_image_path = support_image_path_list[k]  # 根据shot 获取缺陷
            support_label_path = support_gt_path_list[k]  # 根据shot 获取GT
            # 支持集rgb
            support_rgb = cv2.imread(support_image_path, cv2.IMREAD_COLOR)
            support_rgb = cv2.cvtColor(support_rgb, cv2.COLOR_BGR2RGB)
            support_rgb = np.float32(support_rgb)
            # 支持集mask
            support_mask = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
            support_mask[support_mask != 255] = 0
            support_mask[support_mask == 255] = 1

            if support_rgb.shape[0] != support_mask.shape[0] or support_rgb.shape[1] != support_mask.shape[1]:
                raise (RuntimeError(
                    "Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))
            support_image_list.append(support_rgb)
            support_label_list.append(support_mask)

        assert len(support_label_list) == self.shot and len(support_image_list) == self.shot

        raw_label = query_mask.copy()

        if self.transform is not None:
            query_rgb, query_mask = self.transform(query_rgb, query_mask)
            for k in range(self.shot):
                support_image_list[k], support_label_list[k] = self.transform(support_image_list[k],
                                                                              support_label_list[k])

        s_xs = support_image_list  # 支持集缺陷列表
        s_ys = support_label_list  # 支持集gt列表
        s_x = s_xs[0].unsqueeze(0)
        # ---- k-shot ---- #
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)
        if self.mode == 'train':
            return query_rgb, query_mask, s_x, s_y, subcls_list
        else:
            return query_rgb, query_mask, s_x, s_y, subcls_list, raw_label

        # ---- 1shot测试 ---- #
        # s_x = s_xs[0]
        # for i in range(1, self.shot):
        #     s_x = torch.cat([s_xs[i], s_x], 0)
        #
        # s_y = s_ys[0]
        # for j in range(1, self.shot):
        #     s_y = torch.cat([s_ys[j], s_y], 0)
        #
        # if self.mode == 'train':
        #     return query_rgb, query_mask, s_x, s_y, subcls_list, query_img
        # else:
        #     return query_rgb, query_mask, s_x, s_y, subcls_list, raw_label


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils_my import transform, config
    import argparse
    import torchvision


    def get_parser():
        parser = argparse.ArgumentParser(description='Few-Shot Semantic Segmentation')
        parser.add_argument('--config', type=str, default='config/SSD/msd_1_r50_train.yaml', help='config file')
        args = parser.parse_args()
        cfg = config.load_cfg_from_cfg_file(args.config)
        return cfg


    args = get_parser()
    # 归一化相关
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    # 设置训练transform，train data 和trainloader
    train_transform = [
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.padding_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)]

    train_transform = transform.Compose(train_transform)
    train_data = SemData(split=args.split,
                         shot=args.shot,
                         data_root=args.data_root,
                         data_list=args.train_list,
                         transform=train_transform,
                         mode='train')

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True)

    # _,_,imgs_batch, labels_batch, _,= next(iter(train_loader))         # 返回支持集批次的数据
    imgs_batch, labels_batch, _, _, _, raw_path = next(iter(train_loader))  # 返回查询集批次的数据
    plt.figure(figsize=(8, 8))
    # ---- batch ---- #
    print(imgs_batch.shape)
    print(labels_batch.shape)
    print(len(raw_path))
    unloader = torchvision.transforms.ToPILImage()

    for i, (img, label, raw_path) in enumerate(zip(imgs_batch[2:3], labels_batch[2:3], raw_path[2:3])):
        print('img.shape:', img.shape)
        print('label.shape:', label.shape)
        print('raw_path:', raw_path)

        image = img.cpu().clone()
        image = unloader(image)

        raw_root = raw_path
        raw_img = cv2.imread(raw_root, cv2.IMREAD_COLOR)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        # raw_img = np.float32(raw_img)
        label = label.squeeze(0).numpy()
        print(np.unique(label))
        # print(np.unique(image))

        plt.subplot(1, 3, 1)
        label[label == 1] = 200  # 可视化GT
        plt.imshow(label)
        plt.title('GT')

        plt.subplot(1, 3, 2)
        plt.imshow(image)
        plt.title('Image')

        plt.subplot(1, 3, 3)
        plt.imshow(raw_img)
        plt.title('Raw-Image')
        plt.show()

