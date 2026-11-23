"""Análisis de errores en campo: exactitud a nivel de especie y tipo de error.

Para cada corrida evaluada calcula, en PlantDoc-test:
  - exactitud por especie (¿acierta al menos la planta?),
  - % de errores que son dentro de la misma especie (confunde la enfermedad)
    vs. entre especies distintas (confunde la planta),
  - clases que el modelo predice en exceso ("clases sumidero").

    python src/analysis.py
"""
import json
from collections import Counter

import numpy as np

from common import RESULTS_DIR, SPLITS_DIR


def main():
    classes = json.loads((SPLITS_DIR / "classes.json").read_text(encoding="utf-8"))
    species = np.array([c["species"] for c in classes])
    out = {}
    for f in sorted((RESULTS_DIR / "eval").glob("*_preds.npz")):
        run = f.name.replace("_preds.npz", "")
        p = np.load(f)
        y, yhat = p["pd_test_y"], p["pd_test_p"]
        err = y != yhat
        same_sp = species[y] == species[yhat]
        top = Counter(yhat.tolist()).most_common(3)
        out[run] = {
            "acc": float((~err).mean()),
            "species_acc": float(same_sp.mean()),
            "errors": int(err.sum()),
            "errors_within_species_pct": float((err & same_sp).sum() / max(1, err.sum())),
            "top_predicted": [(classes[k]["plantdoc"], v) for k, v in top],
        }
        print(f"{run:34s} acc={out[run]['acc']:.3f} especie={out[run]['species_acc']:.3f} "
              f"errores dentro de especie={out[run]['errors_within_species_pct']:.1%} top={top[0]}")
    (RESULTS_DIR / "error_analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
