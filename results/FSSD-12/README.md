# FSSD-12 1-shot 源码参数复训结果（2026-10-02）

仅 fold 0 移除 ASPP；fold 1、2 保留 ASPP。
训练期间从各折验证集按 class mIoU 选择最佳 checkpoint；这些数值不是独立测试集结果。

| Fold | 最佳轮次（从 0 计数） | class mIoU | 同轮 FB-IoU | 训练过程最高 FB-IoU |
| --- | ---: | ---: | ---: | ---: |
| 0 | 133 | 72.54% | 84.83% | 84.83% |
| 1 | 141 | 72.73% | 78.86% | 79.80% |
| 2 | 246 | 63.95% | 80.15% | 80.26% |
| 平均 | — | **69.74%** | **81.28%** | **81.63%** |

运行脚本、配置、日志、指标、清单和源代码快照保存在本目录。原始模型 checkpoint 与 TensorBoard 文件在 [Release](https://github.com/liuqinian-liu/HDANet/releases/tag/fssd12-1shot-source-repeat-2026-10-02)。

| 原目录中的文件 | Release 附件 |
| --- | --- |
| `fold0/artifacts/events.out.tfevents.1790903272.I2b4bdc63b900601797` | [`fssd12-fold0-events.out.tfevents.1790903272.I2b4bdc63b900601797`](https://github.com/liuqinian-liu/HDANet/releases/download/fssd12-1shot-source-repeat-2026-10-02/fssd12-fold0-events.out.tfevents.1790903272.I2b4bdc63b900601797) |
| `fold0/artifacts/train_epoch_133_0.7254011696794718.pth` | [`fssd12-fold0-train_epoch_133_0.7254011696794718.pth`](https://github.com/liuqinian-liu/HDANet/releases/download/fssd12-1shot-source-repeat-2026-10-02/fssd12-fold0-train_epoch_133_0.7254011696794718.pth) |
| `fold1/artifacts/events.out.tfevents.1790908922.I2b4bdc63b900601797` | [`fssd12-fold1-events.out.tfevents.1790908922.I2b4bdc63b900601797`](https://github.com/liuqinian-liu/HDANet/releases/download/fssd12-1shot-source-repeat-2026-10-02/fssd12-fold1-events.out.tfevents.1790908922.I2b4bdc63b900601797) |
| `fold1/artifacts/train_epoch_141_0.7273297175705078.pth` | [`fssd12-fold1-train_epoch_141_0.7273297175705078.pth`](https://github.com/liuqinian-liu/HDANet/releases/download/fssd12-1shot-source-repeat-2026-10-02/fssd12-fold1-train_epoch_141_0.7273297175705078.pth) |
| `fold2/artifacts/events.out.tfevents.1790915343.I2b4bdc63b900601797` | [`fssd12-fold2-events.out.tfevents.1790915343.I2b4bdc63b900601797`](https://github.com/liuqinian-liu/HDANet/releases/download/fssd12-1shot-source-repeat-2026-10-02/fssd12-fold2-events.out.tfevents.1790915343.I2b4bdc63b900601797) |
| `fold2/artifacts/train_epoch_246_0.6395199550914741.pth` | [`fssd12-fold2-train_epoch_246_0.6395199550914741.pth`](https://github.com/liuqinian-liu/HDANet/releases/download/fssd12-1shot-source-repeat-2026-10-02/fssd12-fold2-train_epoch_246_0.6395199550914741.pth) |

下载后使用 [SHA256SUMS.txt](./SHA256SUMS.txt) 核验附件。源文件传输时的校验记录见 [transfer_verified.json](./transfer_verified.json)。
实验与论文及上一次运行的对比见 [结果对比.md](./结果对比.md)。
