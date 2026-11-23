"""Unifica clases PlantVillage↔PlantDoc, elimina duplicados y crea particiones.

Salidas en splits/:
  classes.json                 28 clases comunes (nombre PlantDoc, nombre PlantVillage, especie)
  pv_train/val/test.csv        PlantVillage (laboratorio), 70/15/15 estratificado, semilla 42
  pd_train.csv, pd_test.csv    PlantDoc (campo), partición oficial
  pd_all.csv                   PlantDoc train+test (solo para evaluar E1/E2, sin entrenamiento en campo)
  data_report.json             conteos, duplicados e imágenes descartadas

    python src/prepare_data.py
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from common import DATA_ROOT, SPLITS_DIR

SEED = 42
PV_DIR = Path("PlantVillage-Dataset/raw/color")
PD_DIR = Path("PlantDoc-Dataset")

# (carpeta PlantDoc, carpeta PlantVillage, especie)
CLASS_MAP = [
    ("Apple Scab Leaf", "Apple___Apple_scab", "Manzana"),
    ("Apple leaf", "Apple___healthy", "Manzana"),
    ("Apple rust leaf", "Apple___Cedar_apple_rust", "Manzana"),
    ("Bell_pepper leaf spot", "Pepper,_bell___Bacterial_spot", "Pimiento"),
    ("Bell_pepper leaf", "Pepper,_bell___healthy", "Pimiento"),
    ("Blueberry leaf", "Blueberry___healthy", "Arándano"),
    ("Cherry leaf", "Cherry_(including_sour)___healthy", "Cereza"),
    ("Corn Gray leaf spot", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Maíz"),
    ("Corn leaf blight", "Corn_(maize)___Northern_Leaf_Blight", "Maíz"),
    ("Corn rust leaf", "Corn_(maize)___Common_rust_", "Maíz"),
    ("Peach leaf", "Peach___healthy", "Durazno"),
    ("Potato leaf early blight", "Potato___Early_blight", "Papa"),
    ("Potato leaf late blight", "Potato___Late_blight", "Papa"),
    ("Raspberry leaf", "Raspberry___healthy", "Frambuesa"),
    ("Soyabean leaf", "Soybean___healthy", "Soya"),
    ("Squash Powdery mildew leaf", "Squash___Powdery_mildew", "Calabaza"),
    ("Strawberry leaf", "Strawberry___healthy", "Fresa"),
    ("Tomato Early blight leaf", "Tomato___Early_blight", "Tomate"),
    ("Tomato Septoria leaf spot", "Tomato___Septoria_leaf_spot", "Tomate"),
    ("Tomato leaf bacterial spot", "Tomato___Bacterial_spot", "Tomate"),
    ("Tomato leaf late blight", "Tomato___Late_blight", "Tomate"),
    ("Tomato leaf mosaic virus", "Tomato___Tomato_mosaic_virus", "Tomate"),
    ("Tomato leaf yellow virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomate"),
    ("Tomato leaf", "Tomato___healthy", "Tomate"),
    ("Tomato mold leaf", "Tomato___Leaf_Mold", "Tomate"),
    ("Tomato two spotted spider mites leaf", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomate"),
    ("grape leaf black rot", "Grape___Black_rot", "Uva"),
    ("grape leaf", "Grape___healthy", "Uva"),
]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def fingerprint(rel: str):
    """md5 de bytes + hash perceptual; None si la imagen no se puede abrir."""
    p = DATA_ROOT / rel
    try:
        raw = p.read_bytes()
        with Image.open(p) as im:
            im = im.convert("RGB")
            return rel, hashlib.md5(raw).hexdigest(), str(imagehash.phash(im))
    except Exception:
        return rel, None, None


def list_images(folder: Path):
    return sorted(str(f.relative_to(DATA_ROOT)).replace("\\", "/")
                  for f in (DATA_ROOT / folder).iterdir() if f.suffix.lower() in IMG_EXT)


def fingerprints(rels):
    with ProcessPoolExecutor() as ex:
        return list(ex.map(fingerprint, rels, chunksize=64))


def dedup(df: pd.DataFrame):
    """Elimina imágenes ilegibles y duplicados exactos (md5) y perceptuales (pHash idéntico)."""
    bad = int(df["md5"].isna().sum())
    df = df.dropna(subset=["md5"])
    n0 = len(df)
    df = df.drop_duplicates("md5")
    exact = n0 - len(df)
    # Mismo pHash con distinta etiqueta = conflicto: se descartan todas las copias.
    conflict = df.groupby("phash")["label"].transform("nunique") > 1
    n_conf = int(conflict.sum())
    df = df[~conflict]
    n1 = len(df)
    df = df.drop_duplicates("phash")
    return df.reset_index(drop=True), {"unreadable": bad, "exact_dupes": exact,
                                        "phash_dupes": n1 - len(df), "label_conflicts": n_conf}


def make_subsets(n_train: int = 300, n_val: int = 100):
    """Protocolo reducido: muestra estratificada de PlantVillage (proyecto acotado).
    Clases con menos imágenes se toman completas. El test de PlantVillage no se reduce."""
    for name, n in (("pv_train", n_train), ("pv_val", n_val)):
        d = pd.read_csv(SPLITS_DIR / f"{name}.csv")
        s = d.sample(frac=1.0, random_state=SEED).groupby("label").head(n)
        s.sort_values("path").to_csv(SPLITS_DIR / f"{name}_small.csv", index=False)
        print(name + "_small", len(s))


def main():
    import sys
    if "--subsets-only" in sys.argv:
        make_subsets()
        return
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    classes = [{"id": i, "plantdoc": d, "plantvillage": v, "species": s}
               for i, (d, v, s) in enumerate(CLASS_MAP)]
    (SPLITS_DIR / "classes.json").write_text(json.dumps(classes, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {}

    # --- PlantVillage (laboratorio) ---
    rows = [(r, c["id"]) for c in classes for r in list_images(PV_DIR / c["plantvillage"])]
    fp = dict((r, (m, h)) for r, m, h in fingerprints([r for r, _ in rows]))
    pv = pd.DataFrame([{"path": r, "label": l, "md5": fp[r][0], "phash": fp[r][1]} for r, l in rows])
    report["pv_raw"] = len(pv)
    pv, report["pv_dedup"] = dedup(pv)
    tr, tmp = train_test_split(pv, test_size=0.30, stratify=pv["label"], random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp["label"], random_state=SEED)
    for name, d in (("pv_train", tr), ("pv_val", va), ("pv_test", te)):
        d[["path", "label"]].sort_values("path").to_csv(SPLITS_DIR / f"{name}.csv", index=False)
        report[name] = len(d)

    # --- PlantDoc (campo) ---
    pd_parts = {}
    for split in ("train", "test"):
        rows = [(r, c["id"]) for c in classes
                if (DATA_ROOT / PD_DIR / split / c["plantdoc"]).exists()
                for r in list_images(PD_DIR / split / c["plantdoc"])]
        fp = dict((r, (m, h)) for r, m, h in fingerprints([r for r, _ in rows]))
        d = pd.DataFrame([{"path": r, "label": l, "md5": fp[r][0], "phash": fp[r][1]} for r, l in rows])
        report[f"pd_{split}_raw"] = len(d)
        d, report[f"pd_{split}_dedup"] = dedup(d)
        pd_parts[split] = d
    # Fuga entre train y test de PlantDoc: se quitan del train las imágenes que también están en test.
    leak = pd_parts["train"]["phash"].isin(set(pd_parts["test"]["phash"]))
    report["pd_train_test_overlap_removed"] = int(leak.sum())
    pd_parts["train"] = pd_parts["train"][~leak]
    # Imágenes de PlantDoc que coinciden con PlantVillage (no debería haber).
    report["pd_pv_overlap"] = int(pd.concat(pd_parts.values())["phash"].isin(set(pv["phash"])).sum())
    for split, d in pd_parts.items():
        d[["path", "label"]].sort_values("path").to_csv(SPLITS_DIR / f"pd_{split}.csv", index=False)
        report[f"pd_{split}"] = len(d)
    pd.concat([pd_parts["train"], pd_parts["test"]])[["path", "label"]].to_csv(SPLITS_DIR / "pd_all.csv", index=False)
    report["pd_all"] = report["pd_train"] + report["pd_test"]

    # Conteos por clase
    counts = {}
    for name in ("pv_train", "pv_val", "pv_test", "pd_train", "pd_test"):
        c = Counter(pd.read_csv(SPLITS_DIR / f"{name}.csv")["label"])
        counts[name] = [int(c.get(i, 0)) for i in range(len(classes))]
    report["per_class"] = counts
    (SPLITS_DIR / "data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_class"}, indent=2))
    make_subsets()


if __name__ == "__main__":
    main()
