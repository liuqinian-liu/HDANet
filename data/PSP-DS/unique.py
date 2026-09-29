import os
import cv2
import numpy as np


def get_unique_pixel_values_from_png_folder(folder_path):
    unique_pixel_values_list = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否以.png结尾
        if filename.lower().endswith('.png'):
            # 构建完整的文件路径
            image_path = os.path.join(folder_path, filename)

            # 读取图像
            image = cv2.imread(image_path)

            # 检查图像是否成功加载
            if image is not None:
                if len(image.shape) == 2:  # 灰度图
                    flat_image = image.flatten()
                else:  # 彩色图
                    flat_image = image.reshape(-1, 3).reshape(-1)  # 先将每个像素的三个通道分开，再展平

                # 使用np.unique获取唯一像素值
                unique_pixel_values = np.unique(flat_image)

                # 将结果添加到列表中
                unique_pixel_values_list.append((filename, unique_pixel_values))

    return unique_pixel_values_list


# 示例用法
folder_path = '/home/abc123/tiger/002_HDANet/FS_dataset/class_2/GT'   # 替换为你的.png文件夹路径
unique_pixel_values = get_unique_pixel_values_from_png_folder(folder_path)

# 打印结果
for filename, values in unique_pixel_values:
    print(f"Unique pixel values in {filename}: {values}")