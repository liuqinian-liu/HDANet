from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path("/hy-tmp/ljh/lx")
PROJECT = Path("/hy-tmp/ljh/HDANet-master")
DATASET = Path("/hy-tmp/ljh/Dataset/HDANet/FSSD-12")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    assert ROOT.is_dir() and DATASET.is_dir()
    manifest = {"dataset": str(DATASET), "model": "HDANet", "shot": 1, "folds": []}

    for fold in range(3):
        script_name = f"train-fssd12-fold{fold}.py"
        source_script = PROJECT / script_name
        script = ROOT / script_name
        assert script.is_file() and source_script.is_file()
        original_sha256 = sha256(source_script)
        code = script.read_text(encoding="utf-8")
        if fold == 0:
            old = "from model_hdanet.HDANet_移除aspp import HDANet"
            assert code.count(old) == 1
            code = code.replace(old, "from model_hdanet.HDANet import HDANet")
        code = code.replace('    os.environ["CUDA_VISIBLE_DEVICES"] = \'0\'\n', "")
        script.write_text(code, encoding="utf-8")

        config = ROOT / "config" / f"fold{fold}_resnet50.yaml"
        cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
        cfg["DATA"]["data_root"] = str(DATASET)
        cfg["DATA"]["train_list"] = str(PROJECT / f"data_list/defect_fssd_12/train/fold{fold}_defect.txt")
        cfg["DATA"]["val_list"] = str(PROJECT / f"data_list/defect_fssd_12/val/fold{fold}_defect.txt")
        cfg["TRAIN"]["save_path"] = str(ROOT / f"fold{fold}" / "artifacts")
        assert cfg["TRAIN"]["shot"] == 1 and cfg["TRAIN"]["split"] == fold
        config.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        manifest["folds"].append(
            {
                "fold": fold,
                "source_script_sha256": original_sha256,
                "run_script_sha256": sha256(script),
                "config": cfg,
            }
        )

    (ROOT / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Prepared three 1-shot HDANet folds in", ROOT)


if __name__ == "__main__":
    main()
