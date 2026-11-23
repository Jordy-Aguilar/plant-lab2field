"""Utilidades compartidas: rutas, semillas, transformaciones y dataset."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", ROOT.parent / "data"))
SPLITS_DIR = ROOT / "splits"
RESULTS_DIR = ROOT / "results"
CKPT_DIR = Path(os.environ.get("CKPT_DIR", ROOT.parent / "checkpoints"))

IMG_SIZE = 224
# Precisión mixta (fp16). Desactivada por defecto: en GPU GTX 16xx (TU116) produce NaN
# en algunas convoluciones con lotes grandes. En GPU con Tensor Cores (T4, RTX) usar USE_AMP=1.
USE_AMP = os.environ.get("USE_AMP", "0") == "1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def train_transform(kind: str = "standard") -> T.Compose:
    """Aumento estándar (E1/E2). 'field' queda reservado para E3."""
    ops = [
        T.RandomResizedCrop(IMG_SIZE, scale=(0.3, 1.0) if kind == "field" else (0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
    ]
    if kind == "field":
        # Aumento orientado a campo (E3): iluminación/color, perspectiva, desenfoque y oclusiones,
        # para que el modelo no dependa del fondo uniforme y la luz controlada del laboratorio.
        ops += [
            T.RandomApply([T.ColorJitter(0.5, 0.5, 0.5, 0.08)], p=0.8),
            T.RandomPerspective(distortion_scale=0.3, p=0.3),
            T.RandomRotation(30),
            T.RandomApply([T.GaussianBlur(7, sigma=(0.1, 3.0))], p=0.4),
            T.RandomGrayscale(p=0.05),
        ]
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if kind == "field":
        ops += [T.RandomErasing(p=0.5, scale=(0.02, 0.15), value="random")]
    return T.Compose(ops)


def eval_transform() -> T.Compose:
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ManifestDataset(Dataset):
    """Lee un CSV con columnas path,label (ruta relativa a DATA_ROOT)."""

    def __init__(self, csv_path, transform):
        self.df = csv_path if isinstance(csv_path, pd.DataFrame) else pd.read_csv(csv_path)
        self.paths = self.df["path"].tolist()
        self.labels = self.df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(DATA_ROOT / self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]
