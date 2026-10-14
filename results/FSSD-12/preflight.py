from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "run_manifest.json").read_text(encoding="utf-8"))
assert str(ROOT) == manifest["run_root"]
assert (ROOT / "initmodel" / "resnet50_v2.pth").is_file()
for relative, expected in manifest["runtime_files"].items():
    path = ROOT / (f"code/{relative}" if relative.endswith(".py") else relative)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, relative
for relative, expected in manifest["source_files"].items():
    if relative.startswith("config/"):
        continue
    actual = hashlib.sha256((ROOT / "code" / relative).read_bytes()).hexdigest()
    assert actual == expected, relative
for py in (ROOT / "code").rglob("*.py"):
    ast.parse(py.read_text(encoding="utf-8-sig"))

report = {"run_root": str(ROOT), "jobs": [], "mask_pairs_checked": 0}
for job in manifest["jobs"]:
    config = yaml.safe_load((ROOT / job["config"]).read_text(encoding="utf-8"))
    train = config["TRAIN"]
    assert train["shot"] == 1 and train["split"] == job["fold"]
    assert train["save_path"] == f"{job['work_dir']}/artifacts"
    assert config["DATA"]["data_root"] == "/hy-tmp/ljh/Dataset/HDANet/FSSD-12"
    script = (ROOT / job["script"]).read_text(encoding="utf-8")
    expected_import = (
        "from model_hdanet.HDANet_移除aspp import HDANet"
        if job["fold"] == 0 else "from model_hdanet.HDANet import HDANet"
    )
    assert expected_import in script
    classes = {}
    for split in ("train", "val"):
        rows = [line.split() for line in Path(config["DATA"][f"{split}_list"]).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        assert rows and all(len(row) == 2 for row in rows)
        classes[split] = sorted({int(cls) for _, cls in rows})
        for relative, _ in rows:
            image = Path(config["DATA"]["data_root"]) / relative
            mask = Path(config["DATA"]["data_root"]) / relative.replace("Images/", "GT/").replace("jpg", "png")
            assert image.is_file() and mask.is_file(), (image, mask)
            report["mask_pairs_checked"] += 1
    assert len(classes["train"]) == 8 and len(classes["val"]) == 4
    assert not (set(classes["train"]) & set(classes["val"]))
    report["jobs"].append({"job": job["id"], "model": job["model"],
                           "lr": train["base_lr"], "batch_size": train["batch_size"],
                           "epochs": train["epochs"], "seed": train["manual_seed"]})

import sys
sys.path[:0] = [str(ROOT / "code"), "/hy-tmp/lx/vendor"]
import torch
from util import config
from model_hdanet.HDANet_移除aspp import HDANet as NoASPP
from model_hdanet.HDANet import HDANet

for job in manifest["jobs"]:
    args = config.load_cfg_from_cfg_file(str(ROOT / job["config"]))
    model = (NoASPP if job["fold"] == 0 else HDANet)(args)
    optimizer = model.get_optim(args)
    assert all(group["lr"] == args.base_lr for group in optimizer.param_groups)
    del model, optimizer
report["torch"] = torch.__version__
report["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
report["status"] = "passed"
(ROOT / "preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
