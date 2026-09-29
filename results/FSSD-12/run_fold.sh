#!/usr/bin/env bash
set -euo pipefail

fold="${1:?fold required}"
gpu="${2:?GPU required}"
root=/hy-tmp/ljh/lx

cd "$root"
mkdir -p "$root/tmp" "$root/cache" "$root/logs" "$root/fold$fold/artifacts"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$root/vendor:$root"
export PYTHONDONTWRITEBYTECODE=1
export HOME="$root"
export TMPDIR="$root/tmp"
export XDG_CACHE_HOME="$root/cache"
export TORCH_HOME="$root/cache/torch"
export CUDA_CACHE_PATH="$root/cache/cuda"
export MPLCONFIGDIR="$root/cache/matplotlib"
export PIP_CACHE_DIR="$root/cache/pip"

python -u "train-fssd12-fold$fold.py" --config "$root/config/fold${fold}_resnet50.yaml"
