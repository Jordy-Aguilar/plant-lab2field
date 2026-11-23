"""Tabla resumen (media ± desviación entre semillas) y figuras del informe.

Las corridas se agrupan por configuración: el nombre de la corrida sin la semilla
(p. ej. mobilenetv2_s43_sub_field -> mobilenetv2_sub_field).

    python src/make_figures.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from common import RESULTS_DIR, SPLITS_DIR

CONFIGS = {  # clave: (experimento, etiqueta corta, color)
    "baseline_sub": ("E1", "CNN base", "#8c8c8c"),
    "mobilenetv2_sub": ("E2", "MobileNetV2", "#C8102E"),
    "efficientnet_b0_sub": ("E2", "EfficientNet-B0", "#2b6cb0"),
    "mobilenetv2_sub_field": ("E3", "MobileNetV2 + aum. campo", "#e07a5f"),
    "mobilenetv2_sub_field_pd10": ("E4", "E3 + 10 % campo", "#81b29a"),
    "mobilenetv2_sub_field_pd25": ("E4", "E3 + 25 % campo", "#3d9970"),
    "mobilenetv2_sub_field_pd50": ("E4", "E3 + 50 % campo", "#1b5e20"),
    "mobilenetv2": ("Control", "MobileNetV2 (datos completos)", "#5c0a17"),
}
FIG = RESULTS_DIR / "figures"
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def cfg_of(run: str) -> str:
    return re.sub(r"_s\d+", "", run, count=1)


def load(kind):
    out = defaultdict(list)
    for f in sorted((RESULTS_DIR / kind).glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        c = cfg_of(d["run"])
        if c in CONFIGS:
            out[c].append(d)
    return out


def ms(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def summary_table(evals, runs):
    rows = []
    for c, (exp, label, _) in CONFIGS.items():
        if c not in evals:
            continue
        E, R = evals[c], runs[c]
        row = {"config": c, "exp": exp, "modelo": label, "semillas": len(E)}
        for key, col in (("pv_test", "PV"), ("pd_test", "PD"), ("pd_all", "PDall")):
            for met in ("acc", "f1_macro"):
                row[f"{col}_{met}"], row[f"{col}_{met}_sd"] = ms([e[key][met] for e in E])
        row["PD_f1_ci_lo"] = np.mean([e["pd_test"]["f1_ci95"][0] for e in E])
        row["PD_f1_ci_hi"] = np.mean([e["pd_test"]["f1_ci95"][1] for e in E])
        row["gap_f1"], row["gap_f1_sd"] = ms([e["gap_f1"] for e in E])
        row["params_m"] = E[0]["efficiency"]["params_m"]
        row["size_mb"] = E[0]["efficiency"]["size_mb_fp32"]
        row["cpu_ms"] = np.median([e["efficiency"]["cpu_ms_per_img"] for e in E])
        row["n_train"] = np.mean([r.get("n_train", np.nan) for r in R])
        row["best_epoch"] = np.mean([r["best_epoch"] for r in R])
        row["epochs_run"] = np.mean([r["epochs_run"] for r in R])
        row["train_min"] = np.mean([r["train_time_min"] for r in R])
        row["epoch_s"] = np.mean([np.mean([h["epoch_time_s"] for h in r["history"]]) for r in R])
        row["gpu_gb"] = np.mean([r["peak_gpu_mem_gb"] for r in R])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)
    return df.set_index("config")


def short_names(classes):
    return [c["plantdoc"].replace("leaf", "").replace("Leaf", "").replace("  ", " ").strip() for c in classes]


def fig_curves(runs, cfgs, name):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for c in cfgs:
        if c not in runs:
            continue
        r = min(runs[c], key=lambda d: d["seed"])
        h = pd.DataFrame(r["history"])
        col, lab = CONFIGS[c][2], CONFIGS[c][1]
        axes[0].plot(h.epoch, h.train_loss, "-", color=col, label=f"{lab} · train")
        axes[0].plot(h.epoch, h.val_loss, "--", color=col, label=f"{lab} · val")
        axes[1].plot(h.epoch, h.val_f1, "-", color=col, label=f"{lab} · laboratorio")
        axes[1].plot(h.epoch, h.pd_f1, ":", color=col, marker="o", ms=2.5, label=f"{lab} · campo")
    axes[0].set(xlabel="Época", ylabel="Pérdida (entropía cruzada)", title="a) Pérdida en PlantVillage", yscale="log")
    axes[1].set(xlabel="Época", ylabel="F1-macro", title="b) Laboratorio (val) vs. campo (PlantDoc-test)", ylim=(0, 1.02))
    axes[0].legend(fontsize=5.8, frameon=False)
    axes[1].legend(fontsize=5.8, frameon=False, loc="center right")
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=220)
    plt.close()


def fig_gap(df, cfgs, name, width=4.8):
    d = df.loc[[c for c in cfgs if c in df.index]]
    fig, ax = plt.subplots(figsize=(width, 2.6))
    x = np.arange(len(d))
    ax.bar(x - 0.2, d.PV_f1_macro, 0.38, yerr=d.PV_f1_macro_sd, capsize=2, color="#d9d9d9", edgecolor="#555",
           label="Laboratorio (PlantVillage-test)")
    ax.bar(x + 0.2, d.PD_f1_macro, 0.38, yerr=d.PD_f1_macro_sd, capsize=2, color="#C8102E",
           label="Campo (PlantDoc-test)")
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(i - 0.2, r.PV_f1_macro + r.PV_f1_macro_sd + 0.015, f"{r.PV_f1_macro:.3f}", ha="center", fontsize=6.5)
        ax.text(i + 0.2, r.PD_f1_macro + r.PD_f1_macro_sd + 0.015, f"{r.PD_f1_macro:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, [CONFIGS[c][1].replace(" + ", "\n+ ") for c in d.index], fontsize=7)
    ax.set(ylabel="F1-macro (media ± d.e.)", ylim=(0, 1.15))
    ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13))
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=220)
    plt.close()


def fig_field_fraction(df, name="e4_fraction.png"):
    pts = [("mobilenetv2_sub_field", 0)] + [(f"mobilenetv2_sub_field_pd{f}", f) for f in (10, 25, 50)]
    pts = [(c, f) for c, f in pts if c in df.index]
    if len(pts) < 2:
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    xs = [f for _, f in pts]
    for col, lab, color, mk in (("PD_f1_macro", "Campo (PlantDoc-test)", "#C8102E", "o"),
                                ("PV_f1_macro", "Laboratorio (PlantVillage-test)", "#555", "s")):
        y = [df.loc[c, col] for c, _ in pts]
        e = [df.loc[c, col + "_sd"] for c, _ in pts]
        ax.errorbar(xs, y, yerr=e, marker=mk, color=color, capsize=3, label=lab)
        for xv, yv in zip(xs, y):
            ax.annotate(f"{yv:.3f}", (xv, yv), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6.5)
    if "mobilenetv2_sub" in df.index:
        ax.axhline(df.loc["mobilenetv2_sub", "PD_f1_macro"], ls=":", color="#C8102E", lw=1)
        ax.text(50, df.loc["mobilenetv2_sub", "PD_f1_macro"] - 0.05, "E2 (sin aum. ni campo)", fontsize=6,
                ha="right", color="#C8102E")
    ax.set(xlabel="% de PlantDoc-train añadido al entrenamiento", ylabel="F1-macro (media ± d.e.)",
           ylim=(0, 1.08), xticks=xs)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=220)
    plt.close()


def fig_perclass(evals, classes, cfgs, name):
    names = short_names(classes)
    labs = sorted(set(np.load(RESULTS_DIR / "eval" / f"{evals[cfgs[0]][0]['run']}_preds.npz")["pd_test_y"]))
    cfgs = [c for c in cfgs if c in evals]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    xs = np.arange(len(labs))
    w = 0.8 / len(cfgs)
    for k, c in enumerate(cfgs):
        f = np.mean([e["pd_test"]["f1_per_class"] for e in evals[c]], axis=0)
        ax.bar(xs + k * w - 0.4 + w / 2, f[labs], w, color=CONFIGS[c][2], label=CONFIGS[c][1])
    ax.set_xticks(xs, [names[i] for i in labs], rotation=75, ha="right", fontsize=6)
    ax.set(ylabel="F1 en PlantDoc-test", ylim=(0, 1))
    ax.legend(fontsize=6.5, frameon=False, ncol=len(cfgs), loc="upper center", bbox_to_anchor=(0.5, 1.13))
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=220)
    plt.close()


def fig_confusion(evals, classes, cfg, name):
    names = short_names(classes)
    run = min(evals[cfg], key=lambda d: d["run"])["run"]
    p = np.load(RESULTS_DIR / "eval" / f"{run}_preds.npz")
    y, yhat = p["pd_test_y"], p["pd_test_p"]
    labs = sorted(set(y))
    cm = confusion_matrix(y, yhat, labels=list(range(len(classes))))[labs][:, labs]
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
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
    plt.savefig(FIG / name, dpi=220)
    plt.close()


def fig_distribution(classes):
    rep = json.loads((SPLITS_DIR / "data_report.json").read_text(encoding="utf-8"))["per_class"]
    pv = np.array(rep["pv_train"]) + np.array(rep["pv_val"]) + np.array(rep["pv_test"])
    pdd = np.array(rep["pd_train"]) + np.array(rep["pd_test"])
    order = np.argsort(-pv)
    names = short_names(classes)
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    xs = np.arange(len(order))
    ax.bar(xs - 0.2, pv[order], 0.4, color="#bdbdbd", label="PlantVillage (laboratorio)")
    ax.bar(xs + 0.2, pdd[order], 0.4, color="#C8102E", label="PlantDoc (campo)")
    ax.axhline(300, ls="--", lw=0.8, color="#333")
    ax.text(len(order) - 0.5, 330, "300 img/clase (protocolo reducido)", fontsize=6, ha="right")
    ax.set_yscale("log")
    ax.set_xticks(xs, [names[i] for i in order], rotation=75, ha="right", fontsize=6)
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
    e12 = ["baseline_sub", "mobilenetv2_sub", "efficientnet_b0_sub"]
    fig_curves(runs, e12, "curves_e1e2.png")
    fig_gap(df, e12, "gap_e1e2.png")
    if all(c in evals for c in e12):
        fig_perclass(evals, classes, e12, "perclass_e1e2.png")
    allc = [c for c in CONFIGS if c != "mobilenetv2"]
    fig_gap(df, allc, "gap_all.png", width=7.2)
    fig_field_fraction(df)
    fig_curves(runs, ["mobilenetv2_sub", "mobilenetv2_sub_field", "mobilenetv2_sub_field_pd50"], "curves_e3e4.png")
    best = df.loc[[c for c in df.index if c != "mobilenetv2"], "PD_f1_macro"].idxmax()
    fig_confusion(evals, classes, best, "confusion_best.png")
    if "mobilenetv2_sub" in evals:
        fig_confusion(evals, classes, "mobilenetv2_sub", "confusion_e2.png")
    pd.set_option("display.width", 250)
    cols = ["exp", "semillas", "PV_f1_macro", "PV_f1_macro_sd", "PD_f1_macro", "PD_f1_macro_sd", "PD_acc",
            "PDall_f1_macro", "gap_f1", "params_m", "size_mb", "cpu_ms", "n_train", "epochs_run", "epoch_s", "train_min"]
    print(df[cols].round(4).to_string())
    print("Mejor configuración en campo:", best)


if __name__ == "__main__":
    main()
