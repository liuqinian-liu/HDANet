"""Collect a completed FSSD-12 rerun and compare it with the first run and paper."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE = "/hy-tmp/lx/fssd12_source_repeat_20261002_0900"
FIRST = Path(r"D:\HDANet\FSSD-12测试2\源码参数结果2（fold0移除ASPP）")
HOST = "root@i-2.gpushare.com"
KEY = str(Path.home() / ".ssh" / "id_ed25519_codex_gpushare")
OPTIONS = ["-i", KEY, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
           "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=10"]
SSH = ["ssh", *OPTIONS, "-p", "53272", HOST]
SCP = ["scp", *OPTIONS, "-P", "53272", "-q"]
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
PAPER = {"folds": [71.20, 75.42, 62.16], "mean": 69.59, "fbiou": 81.46}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def remote_files() -> list[dict]:
    source = f"""
import hashlib, json
from pathlib import Path
root = Path({REMOTE!r})
status = json.loads((root/'status.json').read_text())
assert all(row['state'] == 'complete' for row in status['jobs'].values()), status
files = [root/name for name in ('run_manifest.json','preflight.json','status.json','driver.log')]
files += list((root/'logs').glob('*.log'))
for fold in range(3):
    files += [p for p in (root/f'fold{{fold}}'/'artifacts').rglob('*') if p.is_file()]
rows = []
for path in sorted(files):
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    rows.append(dict(path=path.relative_to(root).as_posix(), size=path.stat().st_size, sha256=sha))
print(json.dumps(rows))
"""
    result = subprocess.run(SSH + ["python -"], input=source, text=True,
                            capture_output=True, encoding="utf-8", timeout=180,
                            creationflags=FLAGS)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    return json.loads(result.stdout)


def transfer(row: dict) -> None:
    dest = HERE / row["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and digest(dest) == row["sha256"]:
        return
    partial = dest.with_name(dest.name + ".part")
    source = f"{HOST}:{REMOTE}/{row['path']}"
    for attempt in range(3):
        result = subprocess.run(SCP + [source, str(partial)], capture_output=True,
                                text=True, encoding="utf-8", timeout=3600,
                                creationflags=FLAGS)
        if result.returncode == 0 and partial.is_file() and digest(partial) == row["sha256"]:
            partial.replace(dest)
            return
    raise RuntimeError(f"Unable to verify transfer: {row['path']}")


def metrics_from_logs() -> dict:
    folds = []
    for fold in range(3):
        lines = (HERE / "logs" / f"fold{fold}.log").read_text(encoding="utf-8", errors="replace").splitlines()
        best = re.findall(r"Best Epoch ([\d.]+), Best IOU ([\d.]+) Best FB-IoU\s+([\d.]+)", "\n".join(lines))
        if not best:
            raise RuntimeError(f"No final best result in fold{fold} log")
        epoch, miou, peak = map(float, best[-1])
        current_fbiou = None
        paired_fbiou = None
        for line in lines:
            found = re.search(r"FBIoU---Val result: mIoU/mAcc/allAcc ([\d.]+)/", line)
            if found:
                current_fbiou = float(found.group(1))
            found = re.search(r"Saving checkpoint to: .*train_epoch_(\d+)_([\d.]+)\.pth", line)
            if found and int(found.group(1)) == int(epoch):
                miou = float(found.group(2))
                paired_fbiou = current_fbiou
        if paired_fbiou is None:
            raise RuntimeError(f"No paired FB-IoU for fold{fold}")
        folds.append({"fold": fold, "best_epoch_zero_based": int(epoch),
                      "class_miou": miou, "paired_fbiou": paired_fbiou,
                      "peak_fbiou": peak})
    means = {key: sum(item[key] for item in folds) / 3
             for key in ("class_miou", "paired_fbiou", "peak_fbiou")}
    return {"folds": folds, "mean": means}


def comparison(first: dict, second: dict) -> str:
    rows = []
    for fold in range(3):
        a = first["folds"][fold]["class_miou"] * 100
        b = second["folds"][fold]["class_miou"] * 100
        paper = PAPER["folds"][fold]
        rows.append(f"| fold {fold} | {paper:.2f}% | {a:.2f}% | {b:.2f}% | {b-a:+.2f} | {abs(a-paper):.2f} | {abs(b-paper):.2f} |")
    a = first["mean"]["class_miou"] * 100
    b = second["mean"]["class_miou"] * 100
    paper = PAPER["mean"]
    rows.append(f"| 三折平均 | {paper:.2f}% | {a:.2f}% | {b:.2f}% | {b-a:+.2f} | {abs(a-paper):.2f} | {abs(b-paper):.2f} |")
    return f"""# FSSD-12 1-shot：两次源码配置运行与论文对比

两次运行均使用 `D:\\HDANet\\HDANet-master源码\\HDANet-master` 的训练与模型源码、相同的数据列表及三份 `移除_fold0/1/2.yaml` 的训练参数。**只有 fold 0 移除 ASPP；fold 1 和 fold 2 保留 ASPP。** 本次训练脚本和模型文件均未修改，仅将配置中的数据、列表、输出路径改为服务器绝对路径。Python 3.11 的 `collections.Iterable` 兼容别名在启动命令中设置，未改源码文件。

论文数据取自 Table III 的 HDANet 1-shot 行。两次运行的数值取每折训练过程中按 class mIoU 选出的最佳 checkpoint，均非独立测试集结果。

| 折 | 论文 mIoU | 第一次 | 这一次 | 本次减首次（百分点） | 首次距论文（百分点） | 本次距论文（百分点） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

论文三折 FBIoU 为 {PAPER['fbiou']:.2f}%；首次同最佳 mIoU 轮次的 FBIoU 为 {first['mean']['paired_fbiou']*100:.2f}%，本次为 {second['mean']['paired_fbiou']*100:.2f}%。论文未说明其 FBIoU 是否取同一轮 checkpoint，因此该项仅作参考。

源码配置各折训练参数：fold 0 为学习率 0.001、batch size 2、200 轮；fold 1 为 0.0025、2、200 轮；fold 2 为 0.0025、4、300 轮。三折随机种子均为 2025。这些源码配置与论文统一描述的学习率 0.001、batch size 2、200 轮并不完全一致，因此数值差距不能仅归因于是否移除 ASPP。

本次 GPU 运行目录：`{REMOTE}`。日志、配置、checkpoint 已传回本目录，传输文件与 GPU 端 SHA-256 校验一致。
"""


def main() -> None:
    rows = remote_files()
    for index, row in enumerate(rows, 1):
        print(f"[{index}/{len(rows)}] {row['path']} ({row['size']} bytes)", flush=True)
        transfer(row)
    (HERE / "transfer_verified.json").write_text(json.dumps({"status": "verified", "files": rows}, indent=2), encoding="utf-8")
    first = json.loads((FIRST / "metrics.json").read_text(encoding="utf-8"))
    second = metrics_from_logs()
    (HERE / "metrics.json").write_text(json.dumps(second, indent=2), encoding="utf-8")
    (HERE / "结果对比.md").write_text(comparison(first, second), encoding="utf-8")
    print(json.dumps({"status": "complete", "mean_miou": second["mean"]["class_miou"]}, indent=2))


if __name__ == "__main__":
    main()
