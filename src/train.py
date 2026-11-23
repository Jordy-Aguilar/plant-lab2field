"""Entrena un modelo en PlantVillage (28 clases comunes) y guarda el mejor checkpoint.

Selección de modelo: F1-macro en la validación de PlantVillage (nunca en PlantDoc).
PlantDoc-test se evalúa en cada época solo para monitoreo (curvas laboratorio vs campo).

Ejemplos:
    python src/train.py --model baseline --seed 42
    python src/train.py --model mobilenetv2 --seed 42
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from common import (CKPT_DIR, RESULTS_DIR, SPLITS_DIR, USE_AMP, ManifestDataset,
                    eval_transform, set_seed, train_transform)
from models import build_model, head_parameters

DEFAULTS = {
    # modelo: (épocas, lr etapa completa, épocas solo cabeza)
    "baseline": (25, 1e-3, 0),
    "mobilenetv2": (12, 3e-4, 1),
    "efficientnet_b0": (12, 3e-4, 1),
}
# Protocolo reducido (--subset): 300 imágenes/clase para entrenar, 100/clase para validar.
DEFAULTS_SUBSET = {"baseline": (20, 1e-3, 0), "mobilenetv2": (10, 3e-4, 1), "efficientnet_b0": (10, 3e-4, 1)}


def field_sample(frac: float, seed: int) -> pd.DataFrame:
    """Fracción estratificada de PlantDoc-train (E4). Clases con <2 imágenes se toman completas."""
    d = pd.read_csv(SPLITS_DIR / "pd_train.csv")
    d = d.sample(frac=1.0, random_state=seed)
    quota = d.groupby("label")["label"].transform(lambda s: max(1, round(len(s) * frac)))
    return d[d.groupby("label").cumcount() < quota]


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    crit = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP and device.type == "cuda"):
            out = model(x)
        loss_sum += crit(out.float(), y).item()
        n += y.numel()
        ys.append(y.cpu().numpy())
        ps.append(out.argmax(1).cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return {
        "loss": loss_sum / n,
        "acc": float((y == p).mean()),
        "f1_macro": float(f1_score(y, p, average="macro", labels=np.unique(y))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(DEFAULTS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--head-epochs", type=int)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--aug", default="standard", choices=["standard", "field"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--subset", action="store_true", help="protocolo reducido (300 img/clase)")
    ap.add_argument("--field-frac", type=float, default=0.0, help="fracción de PlantDoc-train añadida (E4)")
    args = ap.parse_args()

    ep_def, lr_def, head_def = (DEFAULTS_SUBSET if args.subset else DEFAULTS)[args.model]
    epochs = args.epochs or ep_def
    lr = args.lr or lr_def
    head_epochs = head_def if args.head_epochs is None else args.head_epochs
    run = f"{args.model}_s{args.seed}{('_' + args.tag) if args.tag else ''}"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = json.loads((SPLITS_DIR / "classes.json").read_text(encoding="utf-8"))
    n_cls = len(classes)

    dl_kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    g = torch.Generator().manual_seed(args.seed)
    sfx = "_small" if args.subset else ""
    train_df = pd.read_csv(SPLITS_DIR / f"pv_train{sfx}.csv")
    n_field = 0
    if args.field_frac > 0:
        fd = field_sample(args.field_frac, args.seed)
        n_field = len(fd)
        train_df = pd.concat([train_df, fd], ignore_index=True)
    train_dl = DataLoader(ManifestDataset(train_df, train_transform(args.aug)),
                          batch_size=args.batch_size, shuffle=True, drop_last=True, generator=g, **dl_kw)
    val_dl = DataLoader(ManifestDataset(SPLITS_DIR / f"pv_val{sfx}.csv", eval_transform()),
                        batch_size=128, **dl_kw)
    pd_dl = DataLoader(ManifestDataset(SPLITS_DIR / "pd_test.csv", eval_transform()),
                       batch_size=128, num_workers=0)

    # Nota: channels_last fue ~12x más lento en una GTX 1660 SUPER (convoluciones depthwise), se usa NCHW.
    model = build_model(args.model, n_cls).to(device)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")

    history, best_f1, best_ep, bad = [], -1.0, -1, 0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{run}.pt"
    t_start = time.time()

    for ep in range(epochs):
        stage = "head" if ep < head_epochs else "full"
        if ep == 0 or ep == head_epochs:
            # Etapa 1: solo cabeza (backbone congelado). Etapa 2: todo el modelo.
            for p in model.parameters():
                p.requires_grad = stage == "full"
            for p in head_parameters(model, args.model):
                p.requires_grad = True
            params = [p for p in model.parameters() if p.requires_grad]
            stage_lr = 1e-3 if stage == "head" else lr
            stage_epochs = head_epochs if stage == "head" else epochs - head_epochs
            opt = torch.optim.AdamW(params, lr=stage_lr, weight_decay=args.weight_decay)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=stage_lr, total_steps=stage_epochs * len(train_dl), pct_start=0.1)

        model.train()
        t0, loss_sum, correct, n = time.time(), 0.0, 0, 0
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP and device.type == "cuda"):
                out = model(x)
                loss = crit(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * y.numel()
            correct += (out.argmax(1) == y).sum().item()
            n += y.numel()
        train_time = time.time() - t0

        val = predict(model, val_dl, device)
        pdm = predict(model, pd_dl, device)
        rec = {"epoch": ep + 1, "stage": stage, "train_loss": loss_sum / n, "train_acc": correct / n,
               "val_loss": val["loss"], "val_acc": val["acc"], "val_f1": val["f1_macro"],
               "pd_loss": pdm["loss"], "pd_acc": pdm["acc"], "pd_f1": pdm["f1_macro"],
               "epoch_time_s": train_time}
        history.append(rec)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)

        if val["f1_macro"] > best_f1:
            best_f1, best_ep, bad = val["f1_macro"], ep + 1, 0
            torch.save({"model": args.model, "num_classes": n_cls, "state_dict": model.state_dict()}, ckpt_path)
        else:
            bad += 1
            if stage == "full" and bad >= args.patience:
                print(f"Early stopping en la época {ep + 1}")
                break

    out = {
        "run": run, "model": args.model, "seed": args.seed, "aug": args.aug,
        "subset": args.subset, "field_frac": args.field_frac, "n_train": len(train_df), "n_field": n_field,
        "amp": USE_AMP,
        "epochs_max": epochs, "epochs_run": len(history), "best_epoch": best_ep,
        "best_val_f1": best_f1, "lr": lr, "head_epochs": head_epochs,
        "batch_size": args.batch_size, "weight_decay": args.weight_decay,
        "train_time_min": (time.time() - t_start) / 60,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0,
        "history": history,
    }
    (RESULTS_DIR / "runs").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "runs" / f"{run}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Listo: {run} | mejor F1 val = {best_f1:.4f} (época {best_ep})")


if __name__ == "__main__":
    main()
