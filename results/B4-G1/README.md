# B4+G1: FSSD-12 1-shot, seed 2025

These are the three completed B4+G1 combination experiments. Each fold was trained independently from the ImageNet-pretrained ResNet-50. B4 is the 0.1 CSH boundary loss; G1 is the PDM-guided AEM spatial gate. The source fold0/2 directory also contains a separate B4+M2 arm, which is excluded here. The fold1 source directory also contains B4+A and B4+C, which are excluded here.

| Fold | Best epoch (1-based) | class mIoU | Same checkpoint, G1 off | G1 on minus off (points) | Full result |
| --- | ---: | ---: | ---: | ---: | --- |
| fold0 | 134 | 74.4226% | 74.3947% | +0.0279 | [fold0 archive](https://github.com/liuqinian-liu/HDANet/releases/tag/b4-g1-fssd12-seed2025-2026-10) |
| fold1 | 50 | 74.3775% | 74.5618% | -0.1843 | [fold1 archive](https://github.com/liuqinian-liu/HDANet/releases/tag/b4-g1-fssd12-seed2025-2026-10) |
| fold2 | 129 | 65.9014% | 65.0800% | +0.8214 | [fold2 archive](https://github.com/liuqinian-liu/HDANet/releases/tag/b4-g1-fssd12-seed2025-2026-10) |

The three-fold mean class mIoU is 71.5671%. The checkpoint for each fold was selected on its validation set and then reevaluated on the same validation episodes. These are not independent test results. The G1-off values are inference diagnostics on the same checkpoint, not retrained baselines. No multiple-seed B4+G1 result is claimed.

Each `fold*` directory contains the original source configuration, best-checkpoint record, training epoch history, full per-image evaluation JSON for G1 on and off, and the available audit/preflight/status records. Paths inside these original records refer to the training server. The corresponding archive in [Release](https://github.com/liuqinian-liu/HDANet/releases/tag/b4-g1-fssd12-seed2025-2026-10) contains the exact original fold directory, including both the best and recovery checkpoints, TensorBoard events, training/evaluation logs, and preview images. Archive size and SHA-256 are in [manifest.json](manifest.json).

The integrated runnable code and three adjusted configurations are documented in [B4-G1.md](../../B4-G1.md). The archive names, source directory paths, and fold IDs are checked before upload to keep these results separate from B4-only, G1-only, B4+M2, B4+A, B4+C, and later combinations.
