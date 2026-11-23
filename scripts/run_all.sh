#!/usr/bin/env bash
# Experimentos completos con el protocolo reducido (300 img/clase de PlantVillage).
#   E1  CNN base desde cero
#   E2  MobileNetV2 y EfficientNet-B0 con transferencia
#   E3  MobileNetV2 + aumento orientado a campo
#   E4  E3 + {10, 25, 50} % de PlantDoc-train
# Uso: bash scripts/run_all.sh [semillas...]   (por defecto: 42 43 44)
# Variable ONLY_CORE=1: en esa(s) semilla(s) se omite EfficientNet-B0 (más lento en FP32).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
SEEDS=${@:-42 43 44}
mkdir -p results/logs

run() {  # run <tag> <args...>
  local tag=$1; shift
  if [ -f "results/eval/${tag}.json" ]; then echo "ya existe $tag"; return; fi
  echo "== $tag ($(date +%H:%M))"
  $PY -u src/train.py "$@" > "results/logs/${tag}.log" 2>&1 \
    && $PY src/evaluate.py --run "$tag" >> "results/logs/${tag}.log" 2>&1 \
    || echo "FALLO $tag"
}

for s in $SEEDS; do
  run "baseline_s${s}_sub"            --model baseline    --seed "$s" --subset --tag sub
  run "mobilenetv2_s${s}_sub"         --model mobilenetv2 --seed "$s" --subset --tag sub
  [ "${ONLY_CORE:-0}" = "1" ] || \
  run "efficientnet_b0_s${s}_sub"     --model efficientnet_b0 --seed "$s" --subset --tag sub
  run "mobilenetv2_s${s}_sub_field"   --model mobilenetv2 --seed "$s" --subset --aug field --tag sub_field
  for f in 10 25 50; do
    run "mobilenetv2_s${s}_sub_field_pd${f}" --model mobilenetv2 --seed "$s" --subset --aug field \
        --field-frac "0.${f}" --tag "sub_field_pd${f}"
  done
done
echo "TERMINADO $(date +%H:%M)"
