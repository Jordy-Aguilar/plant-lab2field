"""Grad-CAM (Selvaraju et al., 2017) sobre imágenes de PlantVillage y PlantDoc.

    python src/gradcam.py --runs baseline_s42 mobilenetv2_s42 efficientnet_b0_s42
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from common import CKPT_DIR, DATA_ROOT, IMG_SIZE, RESULTS_DIR, SPLITS_DIR, eval_transform
from models import build_model, gradcam_layer

LABELS = {"baseline": "CNN base", "mobilenetv2": "MobileNetV2", "efficientnet_b0": "EfficientNet-B0"}


def cam_for(model, layer, x, target=None):
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    model.zero_grad()
    out = model(x)
    pred = int(out.argmax(1))
    out[0, pred if target is None else target].backward()
    h1.remove(); h2.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy(), pred, float(out.softmax(1)[0, pred])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=3, help="imágenes por dominio")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    classes = json.loads((SPLITS_DIR / "classes.json").read_text(encoding="utf-8"))
    short = [c["plantdoc"] for c in classes]
    rng = np.random.default_rng(args.seed)
    pv = pd.read_csv(SPLITS_DIR / "pv_test.csv")
    pdt = pd.read_csv(SPLITS_DIR / "pd_test.csv")
    # Misma clase en ambos dominios para que la comparación sea justa.
    cls = rng.choice(sorted(set(pdt["label"])), args.n, replace=False)
    samples = [("Laboratorio", pv[pv.label == c].sample(1, random_state=args.seed).iloc[0]) for c in cls]
    samples += [("Campo", pdt[pdt.label == c].sample(1, random_state=args.seed).iloc[0]) for c in cls]

    models = []
    for run in args.runs:
        ck = torch.load(CKPT_DIR / f"{run}.pt", map_location="cpu", weights_only=False)
        m = build_model(ck["model"], ck["num_classes"], pretrained=False)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models.append((ck["model"], m, gradcam_layer(m, ck["model"])))

    tf = eval_transform()
    fig, axes = plt.subplots(len(samples), len(models) + 1, figsize=(2.2 * (len(models) + 1), 2.2 * len(samples)))
    for r, (dom, row) in enumerate(samples):
        img = Image.open(DATA_ROOT / row.path).convert("RGB")
        x = tf(img).unsqueeze(0).requires_grad_(True)
        base = np.asarray(img.resize((IMG_SIZE, IMG_SIZE))) / 255.0
        axes[r, 0].imshow(base)
        axes[r, 0].set_title(f"{dom}\n{short[row.label]}", fontsize=7)
        for c, (name, m, layer) in enumerate(models, start=1):
            cam, pred, conf = cam_for(m, layer, x)
            axes[r, c].imshow(base)
            axes[r, c].imshow(cam, cmap="jet", alpha=0.45)
            ok = "✓" if pred == row.label else "✗"
            axes[r, c].set_title(f"{LABELS[name]} {ok}\n{short[pred]} ({conf:.0%})", fontsize=7,
                                 color="green" if pred == row.label else "red")
    for a in axes.ravel():
        a.axis("off")
    plt.tight_layout()
    out = RESULTS_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "gradcam.png", dpi=200)
    print("Guardado", out / "gradcam.png")


if __name__ == "__main__":
    main()
