import os
import cv2

def convert_grayscale_png_to_rgb_jpg(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件是否以.png结尾
        if filename.lower().endswith('.png'):
            # 构建完整的文件路径
            input_path = os.path.join(input_folder, filename)
            # 构建输出文件路径
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.jpg')

            # 读取灰度图
            gray_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            # 检查图像是否成功加载
            if gray_img is None:
                print(f"Failed to load image: {input_path}")
                continue
            # 将灰度图转换为三通道RGB图
            # 这里使用cv2.cvtColor函数，并指定转换模式为cv2.COLOR_GRAY2BGR
            rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            # 保存为JPG格式
            # 使用cv2.imwrite函数，并指定保存路径和文件名，以及可选的保存参数
            # 这里我们设置JPEG质量为95（默认值为95，范围是0-100）
            success = cv2.imwrite(output_path, rgb_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

            # 检查保存是否成功
            if success:
                print(f'Converted {input_path} to {output_path}')
            else:
                print(f'Failed to save image: {output_path}')


# 示例用法
input_folder = '/home/abc123/tiger/002_HDANet/FS_dataset/class_7/Images'  # 替换为你的输入文件夹路径
output_folder = '/home/abc123/tiger/002_HDANet/FS_dataset/class_7/Images2'  # 替换为你的输出文件夹路径
convert_grayscale_png_to_rgb_jpg(input_folder, output_folder)