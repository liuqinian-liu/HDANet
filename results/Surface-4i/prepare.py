from __future__ import annotations

import ast
import hashlib
import json
import tarfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SOURCE = Path(r"D:\HDANet\HDANet-master源码\HDANet-master")
PREVIOUS = Path(r"D:\HDANet\Surface-4i测试2\源码测试结果2")
REMOTE = "/hy-tmp/lx/surface4i_source_repeat_20261002_0940"
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
    prior_jobs = [job for job in previous["jobs"] if job["dataset"] == "surface4i"]
    assert len(prior_jobs) == 3
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
        script_name = f"train-msdseg-fold{fold}.py"
        script_rel = Path(script_name)
        original = (SOURCE / script_rel).read_text(encoding="utf-8")
        source_hashes[script_name] = digest((SOURCE / script_rel).read_bytes())
        source_import = "from model_hdanet.HDANet import HDANet"
        assert source_import in original, script_name
        ast.parse(original)
        target = CODE / script_rel
        target.write_bytes((SOURCE / script_rel).read_bytes())
        runtime_hashes[script_name] = digest(target.read_bytes())

        cfg_rel = Path("config") / "MSD-Seg" / f"msd_{fold}_r50_train.yaml"
        source_hashes[cfg_rel.as_posix()] = digest((SOURCE / cfg_rel).read_bytes())
        cfg = yaml.safe_load((SOURCE / cfg_rel).read_text(encoding="utf-8"))
        train = cfg["TRAIN"]
        prior = prior_jobs[fold]["settings"]["TRAIN"]
        for key, value in prior.items():
            if key not in ("save_path", "resume", "weight"):
                assert train[key] == value, (fold, key, train[key], value)
        assert train["shot"] == 1 and train["split"] == fold
        work = f"{REMOTE}/fold{fold}"
        cfg["DATA"]["data_root"] = "/hy-tmp/ljh/Dataset/HDANet/Surface Defects-4i"
        for split in ("train", "val"):
            rel = Path("data_list") / "defect_msd" / split / f"fold{fold}_defect.txt"
            copy_source(rel, CODE / rel, source_hashes)
            cfg["DATA"][f"{split}_list"] = f"{REMOTE}/code/{rel.as_posix()}"
        train["save_path"] = f"{work}/artifacts"
        cfg_dest_rel = Path("config") / f"fold{fold}.yaml"
        cfg_dest = STAGE / cfg_dest_rel
        cfg_dest.parent.mkdir(parents=True, exist_ok=True)
        cfg_dest.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
        runtime_hashes[cfg_dest_rel.as_posix()] = digest(cfg_dest.read_bytes())
        jobs.append({
            "id": f"fold{fold}", "fold": fold, "model": "HDANet",
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
            "CUDA initialized with GPU 1 visible before original scripts set CUDA_VISIBLE_DEVICES=0; jobs run sequentially on GPU 1",
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
