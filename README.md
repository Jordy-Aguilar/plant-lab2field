# Del laboratorio al campo: robustez de CNN ligeras para detectar enfermedades en hojas

Proyecto semestral de **Inteligencia Artificial 2** — Universidad de las Américas (UDLA).
Integrantes: Jordy Aguilar, Josué Valencia, Eric Mullo. Docente: Enrique Vinicio Carrera.

Entrenamos clasificadores con fotos de **laboratorio** (PlantVillage) y medimos cuánto caen al
evaluarlos con fotos reales de **campo** (PlantDoc), comparando una CNN entrenada desde cero
(línea base) con dos arquitecturas ligeras con transferencia de aprendizaje
(MobileNetV2 y EfficientNet-B0).

| Experimento | Configuración | Estado |
|---|---|---|
| E1 | CNN base (4 bloques) desde cero, evaluada en ambos dominios | ✅ Entregable 2 |
| E2 | MobileNetV2 y EfficientNet-B0 preentrenadas (ImageNet), ajuste fino por etapas | ✅ Entregable 2 |
| E3 | E2 + aumento de datos orientado a campo (`--aug field`) | ⏳ semanas 11–12 |
| E4 | E2/E3 + {10, 25, 50} % de imágenes de PlantDoc en el entrenamiento | ⏳ semanas 11–12 |

## Estructura

```
plant-lab2field/
├── src/
│   ├── common.py         rutas, semillas, transformaciones, dataset desde CSV
│   ├── prepare_data.py   mapeo de 28 clases, deduplicación (MD5 + pHash), particiones
│   ├── models.py         CNN base, MobileNetV2, EfficientNet-B0
│   ├── train.py          entrenamiento (AdamW + OneCycle, early stopping por F1-macro de validación)
│   ├── evaluate.py       exactitud, F1-macro, IC 95 % bootstrap, F1 por clase, eficiencia en CPU
│   ├── gradcam.py        mapas Grad-CAM laboratorio vs. campo
│   └── make_figures.py   tabla resumen y figuras del informe
├── scripts/
│   ├── extract_plantdoc.py   extrae PlantDoc en Windows (nombres de archivo inválidos en NTFS)
│   └── run_preliminary.sh    reproduce todos los resultados del Entregable 2
├── splits/               particiones exactas usadas (CSV) + clases + reporte de datos
└── results/              métricas (JSON/CSV), registros de entrenamiento y figuras
```

## 1. Instalación

Requiere Python 3.10+ y, para entrenar en tiempo razonable, una GPU NVIDIA.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   ·   Linux/Mac: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## 2. Datos

Los datos no se incluyen en el repositorio. Se descargan en una carpeta `data/` **al lado** del
repositorio (o en la ruta indicada por la variable `DATA_ROOT`):

```bash
mkdir -p ../data && cd ../data
# PlantVillage: solo las imágenes a color (~1 GB)
git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset.git
git -C PlantVillage-Dataset sparse-checkout set raw/color
# PlantDoc (~1 GB)
git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git
cd ../plant-lab2field
```

En **Windows** el checkout de PlantDoc falla porque algunos archivos tienen `?` en el nombre.
En ese caso ejecutar después del clon:

```bash
python scripts/extract_plantdoc.py ../data/PlantDoc-Dataset
```

Versiones usadas: PlantVillage commit `7f7ecc7`, PlantDoc commit `5467f60`.

## 3. Ejecución

```bash
python src/prepare_data.py                         # particiones en splits/ (ya incluidas)
python src/train.py --model baseline --seed 42     # E1
python src/train.py --model mobilenetv2 --seed 42  # E2
python src/train.py --model efficientnet_b0 --seed 42
python src/evaluate.py --run mobilenetv2_s42       # repetir para cada corrida
python src/make_figures.py
python src/gradcam.py --runs baseline_s42 mobilenetv2_s42 efficientnet_b0_s42
```

O todo junto: `bash scripts/run_preliminary.sh 42` (acepta varias semillas: `42 43 44`).

Los checkpoints se guardan en `../checkpoints/` (variable `CKPT_DIR`), fuera del repositorio.

### Google Colab

```python
!git clone <URL-de-este-repositorio> plant-lab2field
%cd plant-lab2field
!pip install -q ImageHash
# descargar los datos como en la sección 2, luego:
import os; os.environ["USE_AMP"] = "1"   # la T4 sí soporta fp16 de forma estable
!python src/prepare_data.py && python src/train.py --model mobilenetv2 --seed 42
```

## 4. Notas de reproducibilidad

- Semilla fija (42) para particiones, inicialización y orden de los lotes; `cudnn.benchmark`
  activado (puede introducir variaciones mínimas entre ejecuciones).
- **Precisión mixta desactivada por defecto** (`USE_AMP=0`): en una GTX 1660 SUPER el modo fp16
  produjo salidas `NaN` en lotes de 128 imágenes. En GPU T4/RTX puede activarse con `USE_AMP=1`.
- **`channels_last` no se usa**: en la misma GPU fue ~12 veces más lento (convoluciones depthwise).
- La selección de modelo usa solo la validación de PlantVillage. PlantDoc-test se evalúa en cada
  época únicamente para graficar las curvas laboratorio vs. campo; nunca para elegir hiperparámetros.

## 5. Datos y licencias

- PlantVillage — Hughes & Salathé (2015), https://github.com/spMohanty/PlantVillage-Dataset
- PlantDoc — Singh et al. (2020), CC BY 4.0, https://github.com/pratikkayal/PlantDoc-Dataset

## Uso de IA generativa

El código y la documentación se desarrollaron con asistencia de Claude (Anthropic). Todos los
experimentos se ejecutaron realmente y los resultados reportados provienen de los archivos en
`results/`.
