"""Genera la tabla resumen y las figuras del informe a partir de results/.

    python src/make_figures.py
"""
from __future__ import annotations

import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from common import RESULTS_DIR, SPLITS_DIR

ORDER = ["baseline", "mobilenetv2", "efficientnet_b0"]
LABELS = {"baseline": "CNN base (E1)", "mobilenetv2": "MobileNetV2 (E2)", "efficientnet_b0": "EfficientNet-B0 (E2)"}
COLORS = {"baseline": "#8c8c8c", "mobilenetv2": "#C8102E", "efficientnet_b0": "#2b6cb0"}
FIG = RESULTS_DIR / "figures"
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load(kind):
    out = defaultdict(list)
    for f in sorted((RESULTS_DIR / kind).glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if "smoke" in d["run"]:
            continue
        out[d["model"]].append(d)
    return out


def ms(vals):
    vals = np.asarray(vals, dtype=float)
    return vals.mean(), (vals.std(ddof=1) if len(vals) > 1 else 0.0)


def summary_table(evals, runs):
    rows = []
    for m in ORDER:
        if m not in evals:
            continue
        E, R = evals[m], runs[m]
        row = {"modelo": LABELS[m], "semillas": len(E)}
        for key, col in (("pv_test", "PV"), ("pd_test", "PD"), ("pd_all", "PDall")):
            for met in ("acc", "f1_macro"):
                mu, sd = ms([e[key][met] for e in E])
                row[f"{col}_{met}"], row[f"{col}_{met}_sd"] = mu, sd
        row["PD_f1_ci_lo"] = np.mean([e["pd_test"]["f1_ci95"][0] for e in E])
        row["PD_f1_ci_hi"] = np.mean([e["pd_test"]["f1_ci95"][1] for e in E])
        row["gap_f1"], row["gap_f1_sd"] = ms([e["gap_f1"] for e in E])
        row["params_m"] = E[0]["efficiency"]["params_m"]
        row["size_mb"] = E[0]["efficiency"]["size_mb_fp32"]
        row["cpu_ms"] = np.mean([e["efficiency"]["cpu_ms_per_img"] for e in E])
        row["best_epoch"] = np.mean([r["best_epoch"] for r in R])
        row["epochs_run"] = np.mean([r["epochs_run"] for r in R])
        row["train_min"] = np.mean([r["train_time_min"] for r in R])
        row["epoch_s"] = np.mean([np.mean([h["epoch_time_s"] for h in r["history"]]) for r in R])
        row["gpu_gb"] = np.mean([r["peak_gpu_mem_gb"] for r in R])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)
    return df


def fig_curves(runs):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for m in ORDER:
        if m not in runs:
            continue
        h = pd.DataFrame(runs[m][0]["history"])
        axes[0].plot(h.epoch, h.train_loss, "-", color=COLORS[m], label=f"{LABELS[m]} train")
        axes[0].plot(h.epoch, h.val_loss, "--", color=COLORS[m], label=f"{LABELS[m]} val")
        axes[1].plot(h.epoch, h.val_f1, "-", color=COLORS[m], label=f"{LABELS[m]} · laboratorio")
        axes[1].plot(h.epoch, h.pd_f1, ":", color=COLORS[m], marker="o", ms=2.5, label=f"{LABELS[m]} · campo")
    axes[0].set(xlabel="Época", ylabel="Pérdida (entropía cruzada)", title="a) Pérdida en PlantVillage")
    axes[1].set(xlabel="Época", ylabel="F1-macro", title="b) Laboratorio (val) vs. campo (PlantDoc)", ylim=(0, 1.02))
    axes[0].legend(fontsize=6, frameon=False)
    axes[1].legend(fontsize=6, frameon=False, loc="center right")
    plt.tight_layout()
    plt.savefig(FIG / "curves.png", dpi=220)
    plt.close()


def fig_gap(df):
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.PV_f1_macro, 0.38, color="#d9d9d9", edgecolor="#555", label="Laboratorio (PlantVillage-test)")
    err = np.vstack([df.PD_f1_macro - df.PD_f1_ci_lo, df.PD_f1_ci_hi - df.PD_f1_macro])
    ax.bar(x + 0.2, df.PD_f1_macro, 0.38, color="#C8102E", yerr=err, capsize=3, label="Campo (PlantDoc-test, IC 95 %)")
    for i, r in df.iterrows():
        ax.text(i - 0.2, r.PV_f1_macro + 0.01, f"{r.PV_f1_macro:.3f}", ha="center", fontsize=7)
        ax.text(i + 0.2, r.PD_f1_ci_hi + 0.01, f"{r.PD_f1_macro:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, [LABELS[m].split(" (")[0] for m in ORDER if LABELS[m] in set(df.modelo)])
    ax.set(ylabel="F1-macro", ylim=(0, 1.12))
    ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.15))
    plt.tight_layout()
    plt.savefig(FIG / "gap.png", dpi=220)
    plt.close()


def fig_confusion_and_perclass(evals, classes):
    names = [c["plantdoc"].replace("leaf", "").replace("Leaf", "").replace("  ", " ").strip() for c in classes]
    best = max((m for m in ORDER if m in evals), key=lambda m: np.mean([e["pd_test"]["f1_macro"] for e in evals[m]]))
    run = evals[best][0]["run"]
    p = np.load(RESULTS_DIR / "eval" / f"{run}_preds.npz")
    y, yhat = p["pd_test_y"], p["pd_test_p"]
    labs = sorted(set(y))
    cm = confusion_matrix(y, yhat, labels=list(range(len(classes))))[labs][:, labs]
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cmn, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(len(labs)), [names[i] for i in labs], rotation=90, fontsize=6)
    ax.set_yticks(range(len(labs)), [names[i] for i in labs], fontsize=6)
    ax.set(xlabel="Predicción", ylabel="Clase real")
    for i in range(len(labs)):
        for j in range(len(labs)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=5,
                        color="white" if cmn[i, j] > 0.5 else "black")
    plt.colorbar(im, fraction=0.03)
    plt.tight_layout()
    plt.savefig(FIG / "confusion_pd.png", dpi=220)
    plt.close()

    # F1 por clase en campo: modelo propuesto vs línea base
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    xs = np.arange(len(labs))
    w = 0.8 / len([m for m in ORDER if m in evals])
    for k, m in enumerate([m for m in ORDER if m in evals]):
        f = np.mean([e["pd_test"]["f1_per_class"] for e in evals[m]], axis=0)
        ax.bar(xs + k * w - 0.4 + w / 2, f[labs], w, color=COLORS[m], label=LABELS[m])
    ax.set_xticks(xs, [names[i] for i in labs], rotation=75, ha="right", fontsize=6)
    ax.set(ylabel="F1 en PlantDoc-test", ylim=(0, 1))
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    plt.tight_layout()
    plt.savefig(FIG / "perclass_pd.png", dpi=220)
    plt.close()
    return best


def fig_distribution(classes):
    rep = json.loads((SPLITS_DIR / "data_report.json").read_text(encoding="utf-8"))["per_class"]
    pv = np.array(rep["pv_train"]) + np.array(rep["pv_val"]) + np.array(rep["pv_test"])
    pdd = np.array(rep["pd_train"]) + np.array(rep["pd_test"])
    order = np.argsort(-pv)
    names = [classes[i]["plantdoc"].replace("leaf", "").replace("Leaf", "").strip() for i in order]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    xs = np.arange(len(order))
    ax.bar(xs - 0.2, pv[order], 0.4, color="#bdbdbd", label="PlantVillage (laboratorio)")
    ax.bar(xs + 0.2, pdd[order], 0.4, color="#C8102E", label="PlantDoc (campo)")
    ax.set_yscale("log")
    ax.set_xticks(xs, names, rotation=75, ha="right", fontsize=6)
    ax.set(ylabel="Imágenes (escala log)")
    ax.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    plt.savefig(FIG / "distribution.png", dpi=220)
    plt.close()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    classes = json.loads((SPLITS_DIR / "classes.json").read_text(encoding="utf-8"))
    evals, runs = load("eval"), load("runs")
    df = summary_table(evals, runs)
    fig_distribution(classes)
    fig_curves(runs)
    fig_gap(df)
    best = fig_confusion_and_perclass(evals, classes)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string())
    print("Mejor modelo en campo:", best)


if __name__ == "__main__":
    main()
