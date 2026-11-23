#!/usr/bin/env bash
# Reproduce exactamente los resultados del Entregable 2:
#   E1 (CNN base) y E2 (MobileNetV2, EfficientNet-B0) con el protocolo reducido, semilla 42,
#   más el control de MobileNetV2 con todos los datos de entrenamiento.
# Uso: PY=python bash scripts/run_preliminary.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p results/logs

$PY src/prepare_data.py
for m in baseline mobilenetv2 efficientnet_b0; do
  $PY -u src/train.py --model "$m" --seed 42 --subset --tag sub | tee "results/logs/${m}_s42_sub.log"
  $PY src/evaluate.py --run "${m}_s42_sub" | tee -a "results/logs/${m}_s42_sub.log"
done
# Control: MobileNetV2 con las 26 964 imágenes de entrenamiento
$PY -u src/train.py --model mobilenetv2 --seed 42 | tee results/logs/mobilenetv2_s42.log
$PY src/evaluate.py --run mobilenetv2_s42 | tee -a results/logs/mobilenetv2_s42.log

$PY src/make_figures.py
$PY src/gradcam.py --runs baseline_s42_sub mobilenetv2_s42_sub efficientnet_b0_s42_sub
$PY src/analysis.py
