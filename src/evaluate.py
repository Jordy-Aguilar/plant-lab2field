"""Evalúa un checkpoint en PlantVillage-test (dentro de dominio) y PlantDoc (fuera de dominio).

Guarda: métricas (exactitud, F1-macro, IC 95 % por bootstrap, F1 por clase),
predicciones y métricas de eficiencia (parámetros, MB, ms/imagen en CPU).

    python src/evaluate.py --run mobilenetv2_s42
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from common import CKPT_DIR, IMG_SIZE, RESULTS_DIR, SPLITS_DIR, USE_AMP, ManifestDataset, eval_transform
from models import build_model

EVAL_SETS = {"pv_test": "pv_test.csv", "pd_test": "pd_test.csv", "pd_all": "pd_all.csv"}


@torch.no_grad()
def predict(model, csv, device):
    dl = DataLoader(ManifestDataset(SPLITS_DIR / csv, eval_transform()), batch_size=128, num_workers=4)
    ys, ps = [], []
    for x, y in dl:
        with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP and device.type == "cuda"):
            out = model(x.to(device))
        ys.append(y.numpy())
        ps.append(out.argmax(1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def f1m(y, p):
    return f1_score(y, p, average="macro", labels=np.unique(y), zero_division=0)


def bootstrap_ci(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    f1s, accs = [], []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        f1s.append(f1m(y[b], p[b]))
        accs.append((y[b] == p[b]).mean())
    return {"f1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
            "acc_ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]}


def cpu_latency_ms(name, n_cls, state_dict, runs=50):
    torch.set_num_threads(4)
    m = build_model(name, n_cls, pretrained=False)
    m.load_state_dict(state_dict)
    m.eval()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.inference_mode():
        for _ in range(10):
            m(x)
        ts = []
        for _ in range(runs):
            t = time.perf_counter()
            m(x)
            ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    ck = torch.load(CKPT_DIR / f"{args.run}.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ck["model"], ck["num_classes"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()

    res = {"run": args.run, "model": ck["model"]}
    preds = {}
    for key, csv in EVAL_SETS.items():
        y, p = predict(model, csv, device)
        preds[f"{key}_y"], preds[f"{key}_p"] = y, p
        res[key] = {"n": int(len(y)), "acc": float(accuracy_score(y, p)), "f1_macro": float(f1m(y, p)),
                    "f1_per_class": f1_score(y, p, average=None, labels=list(range(ck["num_classes"])),
                                             zero_division=0).tolist()}
        if key in ("pv_test", "pd_test"):
            res[key].update(bootstrap_ci(y, p))
    res["gap_f1"] = res["pv_test"]["f1_macro"] - res["pd_test"]["f1_macro"]
    res["gap_acc"] = res["pv_test"]["acc"] - res["pd_test"]["acc"]

    n_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(t.numel() * t.element_size() for t in ck["state_dict"].values()) / 1e6
    res["efficiency"] = {"params_m": n_params / 1e6, "size_mb_fp32": size_mb,
                         "cpu_ms_per_img": cpu_latency_ms(ck["model"], ck["num_classes"], ck["state_dict"])}

    out_dir = RESULTS_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.run}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / f"{args.run}_preds.npz", **preds)
    print(json.dumps({k: res[k] for k in ("run", "gap_f1", "efficiency")}, indent=2))
    for k in EVAL_SETS:
        print(f"{k:8s} n={res[k]['n']:5d} acc={res[k]['acc']:.4f} F1={res[k]['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
