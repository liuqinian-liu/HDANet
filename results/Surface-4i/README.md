# Surface-4i 1-shot 源码参数复训结果（2026-10-02）

三个 fold 均使用标准 HDANet（含 ASPP）。
训练期间从各折验证集按 class mIoU 选择最佳 checkpoint；这些数值不是独立测试集结果。

| Fold | 最佳轮次（从 0 计数） | class mIoU | 同轮 FB-IoU | 训练过程最高 FB-IoU |
| --- | ---: | ---: | ---: | ---: |
| 0 | 155 | 53.82% | 61.45% | 63.35% |
| 1 | 38 | 42.01% | 67.53% | 67.53% |
| 2 | 26 | 38.20% | 55.28% | 56.79% |
| 平均 | — | **44.68%** | **61.42%** | **62.56%** |

运行脚本、配置、日志、指标、清单和源代码快照保存在本目录。原始模型 checkpoint 与 TensorBoard 文件在 [Release](https://github.com/liuqinian-liu/HDANet/releases/tag/surface4i-1shot-source-repeat-2026-10-02)。

| 原目录中的文件 | Release 附件 |
| --- | --- |
| `fold0/artifacts/events.out.tfevents.1790904347.I2b4bdc63b900601797` | [`surface4i-fold0-events.out.tfevents.1790904347.I2b4bdc63b900601797`](https://github.com/liuqinian-liu/HDANet/releases/download/surface4i-1shot-source-repeat-2026-10-02/surface4i-fold0-events.out.tfevents.1790904347.I2b4bdc63b900601797) |
| `fold0/artifacts/train_epoch_155_0.5381881262711985.pth` | [`part01`](https://github.com/liuqinian-liu/HDANet/releases/download/surface4i-1shot-source-repeat-2026-10-02/surface4i-fold0-train_epoch_155_0.5381881262711985.pth.part01), [`part02`](https://github.com/liuqinian-liu/HDANet/releases/download/surface4i-1shot-source-repeat-2026-10-02/surface4i-fold0-train_epoch_155_0.5381881262711985.pth.part02), [`part03`](https://github.com/liuqinian-liu/HDANet/releases/download/surface4i-1shot-source-repeat-2026-10-02/surface4i-fold0-train_epoch_155_0.5381881262711985.pth.part03), [`part04`](https://github.com/liuqinian-liu/HDANet/releases/download/surface4i-1shot-source-repeat-2026-10-02/surface4i-fold0-train_epoch_155_0.5381881262711985.pth.part04) |
| `fold1/artifacts/events.out.tfevents.1790910937.I2b4bdc63b900601797` | [`surface4i-fold1-events.out.tfevents.1790910937.I2b4bdc63b900601797`](https://github.com/liuqinian-liu/HDANet/releases/download/surface4i-1shot-source-repeat-2026-10-02/surface4i-fold1-events.out.tfevents.1790910937.I2b4bdc63b900601797) |
| `fold1/artifacts/train_epoch_38_0.420111278585039.pth` | [`part01`](https://github.com/liuqinian-liu/HDANet/releases/download/surface4i-1shot-source-repeat-2026-10-02/surface4i-fold1-train_epoch_38_0.420111278585039.pth.part01), [`part02`](https://github.com/liuqinian-liu/HDANet/releases/download/surface4i-1shot-source-repeat-2026-10-02/surface4i-fold1-train_epoch_38_0.420111278585039.pth.part02), [`part03`](https://github.com/liuqinian-liu/HDANet/releases/download/surface4i-1shot-source-repeat-2026-10-02/surface4i-fold1-train_epoch_38_0.420111278585039.pth.part03), [`part04`](https://github.com/liuqinian-liu/HDANet/releases/download/surface4i-1shot-source-repeat-2026-10-02/surface4i-fold1-train_epoch_38_0.420111278585039.pth.part04) |
| `fold2/artifacts/events.out.tfevents.1790917513.I2b4bdc63b900601797` | [`surface4i-fold2-events.out.tfevents.1790917513.I2b4bdc63b900601797`](https://github.com/liuqinian-liu/HDANet/releases/download/surface4i-1shot-source-repeat-2026-10-02/surface4i-fold2-events.out.tfevents.1790917513.I2b4bdc63b900601797) |
| `fold2/artifacts/train_epoch_26_0.382028924898798.pth` | [`part01`](https://github.com/liuqinian-liu/HDANet/releases/download/surface4i-1shot-source-repeat-2026-10-02/surface4i-fold2-train_epoch_26_0.382028924898798.pth.part01), [`part02`](https://github.com/liuqinian-liu/HDANet/releases/download/surface4i-1shot-source-repeat-2026-10-02/surface4i-fold2-train_epoch_26_0.382028924898798.pth.part02), [`part03`](https://github.com/liuqinian-liu/HDANet/releases/download/surface4i-1shot-source-repeat-2026-10-02/surface4i-fold2-train_epoch_26_0.382028924898798.pth.part03), [`part04`](https://github.com/liuqinian-liu/HDANet/releases/download/surface4i-1shot-source-repeat-2026-10-02/surface4i-fold2-train_epoch_26_0.382028924898798.pth.part04) |

checkpoint 按原始字节拆成 64 MiB 分卷；下载同一文件的所有 `part` 附件后，用仓库的 `results/restore_checkpoints.py` 合并并校验：

```bash
python results/restore_checkpoints.py results/Surface-4i/checkpoint_parts.json DOWNLOAD_DIR OUTPUT_DIR
```

脚本先校验每个分卷，再按顺序合并并核验完整 checkpoint 的 SHA-256。分卷校验值见 [SHA256SUMS.parts.txt](./SHA256SUMS.parts.txt)，完整文件及 TensorBoard 校验值见 [SHA256SUMS.txt](./SHA256SUMS.txt)。源文件传输时的校验记录见 [transfer_verified.json](./transfer_verified.json)。
实验与论文及上一次运行的对比见 [结果对比.md](./结果对比.md)。
