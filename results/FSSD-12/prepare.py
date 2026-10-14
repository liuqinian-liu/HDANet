from __future__ import annotations

import ast
import hashlib
import json
import tarfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SOURCE = Path(r"D:\HDANet\HDANet-master源码\HDANet-master")
PREVIOUS = Path(r"D:\HDANet\FSSD-12测试2\源码参数结果2（fold0移除ASPP）")
REMOTE = "/hy-tmp/lx/fssd12_source_repeat_20261002_0900"
STAGE = HERE / "package"
CODE = STAGE / "code"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def copy_source(relative: Path, destination: Path, source_hashes: dict[str, str]) -> None:
    data = (SOURCE / relative).read_bytes()
    key = relative.as_posix()
    source_hashes[key] = digest(data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def main() -> None:
    previous = json.loads((PREVIOUS / "run_manifest.json").read_text(encoding="utf-8"))
    source_hashes: dict[str, str] = {}
    runtime_hashes: dict[str, str] = {}
    jobs = []
    for directory in ("model_hdanet", "util"):
        for source_file in (SOURCE / directory).rglob("*.py"):
            if "__pycache__" in source_file.parts:
                continue
            rel = source_file.relative_to(SOURCE)
            copy_source(rel, CODE / rel, source_hashes)
    for fold in range(3):
        script_name = f"train-fssd12-fold{fold}.py"
        script_rel = Path(script_name)
        original = (SOURCE / script_rel).read_text(encoding="utf-8")
        source_hashes[script_name] = digest((SOURCE / script_rel).read_bytes())
        source_import = (
            "from model_hdanet.HDANet_移除aspp import HDANet"
            if fold == 0 else "from model_hdanet.HDANet import HDANet"
        )
        assert source_import in original, script_name
        ast.parse(original)
        target = CODE / script_rel
        target.write_bytes((SOURCE / script_rel).read_bytes())
        runtime_hashes[script_name] = digest(target.read_bytes())

        cfg_rel = Path("config") / "FSSD-12" / f"移除_fold{fold}.yaml"
        source_hashes[cfg_rel.as_posix()] = digest((SOURCE / cfg_rel).read_bytes())
        cfg = yaml.safe_load((SOURCE / cfg_rel).read_text(encoding="utf-8"))
        train = cfg["TRAIN"]
        prior = previous["jobs"][fold]["settings"]["TRAIN"]
        for key, value in prior.items():
            if key not in ("save_path", "resume", "weight", "train_gpu"):
                assert train[key] == value, (fold, key, train[key], value)
        assert train["shot"] == 1 and train["split"] == fold
        work = f"{REMOTE}/fold{fold}"
        cfg["DATA"]["data_root"] = "/hy-tmp/ljh/Dataset/HDANet/FSSD-12"
        for split in ("train", "val"):
            rel = Path("data_list") / "defect_fssd_12" / split / f"fold{fold}_defect.txt"
            copy_source(rel, CODE / rel, source_hashes)
            cfg["DATA"][f"{split}_list"] = f"{REMOTE}/code/{rel.as_posix()}"
        train["save_path"] = f"{work}/artifacts"
        cfg_dest_rel = Path("config") / f"fold{fold}.yaml"
        cfg_dest = STAGE / cfg_dest_rel
        cfg_dest.parent.mkdir(parents=True, exist_ok=True)
        cfg_dest.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
        runtime_hashes[cfg_dest_rel.as_posix()] = digest(cfg_dest.read_bytes())
        jobs.append({
            "id": f"fold{fold}", "fold": fold, "model": "HDANet_移除aspp" if fold == 0 else "HDANet",
            "script": f"code/{script_name}", "config": cfg_dest_rel.as_posix(),
            "source_config": cfg_rel.as_posix(), "work_dir": work, "settings": cfg,
        })

    checked = [name for name in source_hashes if name in previous["source_files"]]
    mismatch = [name for name in checked if source_hashes[name] != previous["source_files"][name]]
    assert not mismatch, f"Source changed since first run: {mismatch}"
    manifest = {
        "run_root": REMOTE,
        "source": str(SOURCE),
        "comparison_run": str(PREVIOUS),
        "source_files": source_hashes,
        "runtime_files": runtime_hashes,
        "runtime_adjustments": [
            "Only absolute dataset/list/output paths in copied YAML files",
            "Training scripts and model source files byte-for-byte unchanged",
            "Python 3.11 launch-time compatibility alias for collections.Iterable; no source file edit",
            "Original scripts pin CUDA_VISIBLE_DEVICES=0, so jobs run sequentially on GPU 0",
        ],
        "jobs": jobs,
    }
    (STAGE / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("preflight.py", "run_all.py"):
        (STAGE / name).write_bytes((HERE / name).read_bytes())
    with tarfile.open(HERE / "package.tar.gz", "w:gz") as archive:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(STAGE).as_posix())
    print(json.dumps({"remote": REMOTE, "jobs": [j["id"] for j in jobs],
                      "source_files_checked_against_previous": len(checked),
                      "package": str(HERE / "package.tar.gz")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
