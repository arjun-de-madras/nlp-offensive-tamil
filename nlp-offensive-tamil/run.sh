#!/usr/bin/env bash
# Reproducible end-to-end run. Random seed is fixed to 42 (CFG["seed"] in the notebook;
# set_seed() seeds python, numpy, torch, CUDA and cudnn-deterministic mode).
#
# Usage:
#   bash run.sh            # install deps, run full pipeline (train + evaluate), launch demo
#   bash run.sh demo       # only launch the Gradio demo from existing ./artifacts
#   bash run.sh --share    # same as full run but with a public gradio.live link
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
MODE="${1:-full}"

if [ ! -d .venv ]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

if [ "$MODE" != "demo" ]; then
  echo ">> Running notebook (seed=42). A T4-class GPU is recommended (~25-35 min)."
  jupyter nbconvert --to notebook --execute NLP_MicroProject_Offensive_Tamil.ipynb \
      --output executed.ipynb --ExecutePreprocessor.timeout=-1
fi

echo ">> Launching Gradio demo"
if [ "$MODE" = "--share" ]; then python app.py --share; else python app.py; fi
