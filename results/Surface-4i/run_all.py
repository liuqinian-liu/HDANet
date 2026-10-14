from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "run_manifest.json").read_text(encoding="utf-8"))
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)


def timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def environment() -> dict[str, str]:
    env = os.environ.copy()
    cache, temp = ROOT / "cache", ROOT / "tmp"
    cache.mkdir(exist_ok=True)
    temp.mkdir(exist_ok=True)
    env.update(CUDA_VISIBLE_DEVICES="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=f"{ROOT}/code:/hy-tmp/lx/vendor", PYTHONUNBUFFERED="1",
               TMPDIR=str(temp), TMP=str(temp), TEMP=str(temp),
               XDG_CACHE_HOME=str(cache), TORCH_HOME=str(cache / "torch"),
               CUDA_CACHE_PATH=str(cache / "cuda"), MPLCONFIGDIR=str(cache / "matplotlib"),
               PIP_CACHE_DIR=str(cache / "pip"), TRITON_CACHE_DIR=str(cache / "triton"),
               TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), HF_HOME=str(cache / "huggingface"),
               NUMBA_CACHE_DIR=str(cache / "numba"), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="1")
    return env


def main() -> None:
    status = {"started": timestamp(), "run_root": str(ROOT),
              "jobs": {job["id"]: {"state": "queued"} for job in MANIFEST["jobs"]}}

    def save() -> None:
        path = ROOT / "status.json.tmp"
        path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        path.replace(ROOT / "status.json")

    save()

    for job in MANIFEST["jobs"]:
        name = job["id"]
        work = Path(job["work_dir"])
        work.mkdir(parents=True, exist_ok=True)
        (work / "artifacts").mkdir(exist_ok=True)
        link = work / "initmodel"
        if not link.exists():
            link.symlink_to(ROOT / "initmodel", target_is_directory=True)
        compatibility = (
            "import collections,collections.abc,runpy,sys,torch;"
            "collections.Iterable=collections.abc.Iterable;"
            "torch.cuda.init();"
            "sys.argv=[sys.argv[1],'--config',sys.argv[2]];"
            "runpy.run_path(sys.argv[0],run_name='__main__')"
        )
        command = [sys.executable, "-B", "-u", "-c", compatibility,
                   str(ROOT / job["script"]), str(ROOT / job["config"])]
        with (LOGS / f"{name}.log").open("a", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=work, env=environment(),
                                       stdout=stream, stderr=subprocess.STDOUT)
            status["jobs"][name] = {"state": "running", "gpu": 1,
                                    "pid": process.pid, "started": timestamp()}
            save()
            code = process.wait()
        status["jobs"][name].update(state="complete" if code == 0 else "failed",
                                    exit_code=code, finished=timestamp())
        save()
        (LOGS / f"{name}.{'done' if code == 0 else 'failed'}").write_text(timestamp())
        if code:
            break
    status["finished"] = timestamp()
    save()
    (LOGS / ("all.done" if all(item["state"] == "complete" for item in status["jobs"].values())
             else "training.failed")).write_text(timestamp())


if __name__ == "__main__":
    main()
