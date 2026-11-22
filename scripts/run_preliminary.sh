#!/usr/bin/env bash
# Reproduce los resultados preliminares del Entregable 2 (E1 + E2).
# Uso: bash scripts/run_preliminary.sh [semillas...]   (por defecto: 42)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
SEEDS=${@:-42}

$PY src/prepare_data.py
for s in $SEEDS; do
  for m in baseline mobilenetv2 efficientnet_b0; do
    $PY src/train.py --model "$m" --seed "$s"
    $PY src/evaluate.py --run "${m}_s${s}"
  done
done
$PY src/make_figures.py
$PY src/gradcam.py --runs baseline_s42 mobilenetv2_s42 efficientnet_b0_s42
