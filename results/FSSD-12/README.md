# FSSD-12 1-shot 三折验证结果

运行日期：2026-09-29。模型为标准 HDANet（ResNet-50）。结果来自训练期间对各折 `val_list` 的验证；**没有额外的独立测试集评估**。

| Fold | 训练轮数 | 最佳轮次（从 0 计数） | 最佳 class mIoU | 训练过程中最高 FB-IoU |
| --- | ---: | ---: | ---: | ---: |
| 0 | 200 | 104 | 72.07% | 84.48% |
| 1 | 200 | 20 | 74.58% | 81.68% |
| 2 | 300 | 233 | 65.03% | 81.43% |
| 平均 | — | — | **70.56%** | **82.53%** |

最佳 checkpoint 根据 class mIoU 选择；FB-IoU 是各折训练过程中的单独峰值，可能来自不同轮次。更多运行说明见 [结果摘要](./结果摘要.md)。

本目录提供运行脚本、配置、日志、TensorBoard 记录和校验信息。较大的模型文件放在 [FSSD-12 结果 Release](https://github.com/liuqinian-liu/HDANet/releases/tag/fssd12-1shot-2026-09-29)：

- `fssd12-fold0-epoch104-full.pth`
- `fssd12-fold1-epoch20-full.pth`、`fssd12-fold1-epoch20-model-only.pth`
- `fssd12-fold2-epoch233-full.pth`、`fssd12-fold2-epoch233-model-only.pth`

`full` checkpoint 包含训练状态；`model-only` 仅包含模型权重。下载后可用 [SHA256SUMS.txt](./SHA256SUMS.txt) 校验文件。Fold 0 没有单独的 `model-only` 文件。
