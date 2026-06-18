from __future__ import annotations

import json
import runpy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parent
MODEL_SCRIPTS = [
    "01_fob_random_forest.py",
    "02_fob_hist_gradient_boosting.py",
    "03_fob_elasticnet.py",
    "04_margen_random_forest.py",
    "05_margen_hist_gradient_boosting.py",
    "06_margen_elasticnet.py",
    "07_escenario_random_forest.py",
    "08_escenario_hist_gradient_boosting.py",
    "09_escenario_logistic_regression.py",
    "10_fob_prophet.py",
    "11_margen_prophet.py",
    "12_escenario_prophet.py",
    "13_fob_ensemble.py",
]


if __name__ == "__main__":
    for script in MODEL_SCRIPTS:
        print(f"\n=== Ejecutando {script} ===")
        runpy.run_path(str(ROOT / script), run_name="__main__")

    metrics = {}
    for path in sorted((ROOT.parent / "models").glob("*_metrics.json")):
        metrics[path.stem.replace("_metrics", "")] = json.loads(path.read_text(encoding="utf-8"))
    summary_path = ROOT.parent / "models" / "training_summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    figures_dir = ROOT.parent / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    regression_rows = []
    classifier_rows = []
    for name, values in metrics.items():
        if "mae" in values:
            solution = "FOB" if "_fob_" in f"_{name}_" else "Margen"
            regression_rows.append(
                {
                    "modelo": name.split("_", 2)[-1].replace("_", " ").title(),
                    "solucion": solution,
                    "MAE": values["mae"],
                    "RMSE": values.get("rmse"),
                    "R2": values.get("r2"),
                }
            )
        elif "roc_auc" in values:
            classifier_rows.append(
                {
                    "modelo": name.split("_", 2)[-1].replace("_", " ").title(),
                    "Accuracy": values.get("accuracy"),
                    "Precision": values.get("precision"),
                    "Recall": values.get("recall"),
                    "ROC AUC": values.get("roc_auc"),
                }
            )

    sns.set_theme(style="whitegrid")
    reg_df = pd.DataFrame(regression_rows)
    if not reg_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, solution, color in zip(axes, ["FOB", "Margen"], ["#176B87", "#2E8B57"]):
            subset = reg_df[reg_df["solucion"] == solution].sort_values("MAE")
            sns.barplot(data=subset, x="MAE", y="modelo", ax=ax, color=color)
            ax.set_title(f"Comparación de MAE - {solution}")
            ax.set_xlabel("MAE (menor es mejor)")
            ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(figures_dir / "comparacion_modelos_regresion.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        reg_df.to_csv(ROOT.parent / "outputs" / "comparacion_modelos_regresion.csv", index=False, encoding="utf-8-sig")

    cls_df = pd.DataFrame(classifier_rows)
    if not cls_df.empty:
        long_df = cls_df.melt(id_vars="modelo", var_name="metrica", value_name="valor")
        fig, ax = plt.subplots(figsize=(12, 5.5))
        sns.barplot(data=long_df, x="valor", y="modelo", hue="metrica", ax=ax)
        ax.set_xlim(0, 1)
        ax.set_title("Comparación de modelos de riesgo de margen bajo")
        ax.set_xlabel("Valor de la métrica")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(figures_dir / "comparacion_modelos_clasificacion.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        cls_df.to_csv(ROOT.parent / "outputs" / "comparacion_modelos_clasificacion.csv", index=False, encoding="utf-8-sig")

    output_dir = ROOT.parent / "outputs"
    display_names = {
        "01_fob_random_forest": "Random Forest",
        "02_fob_hist_gradient_boosting": "HistGradientBoosting",
        "03_fob_elasticnet": "ElasticNet",
        "10_fob_prophet": "Prophet",
        "13_fob_ensemble": "Ensamble",
        "04_margen_random_forest": "Random Forest",
        "05_margen_hist_gradient_boosting": "HistGradientBoosting",
        "06_margen_elasticnet": "ElasticNet",
        "11_margen_prophet": "Prophet",
        "07_escenario_random_forest": "Random Forest",
        "08_escenario_hist_gradient_boosting": "HistGradientBoosting",
        "09_escenario_logistic_regression": "Regresión logística",
        "12_escenario_prophet": "Prophet",
    }

    fob_names = [
        "01_fob_random_forest",
        "02_fob_hist_gradient_boosting",
        "03_fob_elasticnet",
        "10_fob_prophet",
        "13_fob_ensemble",
    ]
    fob_frames = {}
    for name in fob_names:
        frame = pd.read_csv(output_dir / f"{name}_predicciones.csv", parse_dates=["fecha"])
        fob_frames[name] = frame.groupby("fecha", as_index=False)[["real", "predicho"]].mean().sort_values("fecha")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    reference = fob_frames[fob_names[0]]
    axes[0].plot(reference["fecha"], reference["real"], color="#111827", linewidth=2.2, label="Precio real")
    palette = ["#176B87", "#2E8B57", "#8B5E3C", "#C65911", "#7B2CBF"]
    for name, color in zip(fob_names, palette):
        frame = fob_frames[name]
        axes[0].plot(frame["fecha"], frame["predicho"], linewidth=1.45, color=color, label=display_names[name])
    axes[0].set_title("Precio FOB real y predicciones de todos los modelos")
    axes[0].set_ylabel("USD/kg")
    axes[0].legend(ncol=3, fontsize=9)
    fob_metric = reg_df[reg_df["solucion"] == "FOB"].copy()
    fob_metric["modelo"] = fob_metric["modelo"].replace(
        {"Hist Gradient Boosting": "HistGradientBoosting", "Elasticnet": "ElasticNet"}
    )
    sns.barplot(data=fob_metric.sort_values("MAE"), x="MAE", y="modelo", ax=axes[1], color="#176B87")
    axes[1].set_title("MAE de los modelos FOB")
    axes[1].set_xlabel("MAE en USD/kg (menor es mejor)")
    axes[1].set_ylabel("")
    fig.tight_layout()
    fig.savefig(figures_dir / "comparacion_completa_modelos_fob.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    margin_names = [
        "04_margen_random_forest",
        "05_margen_hist_gradient_boosting",
        "06_margen_elasticnet",
        "11_margen_prophet",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, name, color in zip(axes.flat, margin_names, palette):
        frame = pd.read_csv(output_dir / f"{name}_predicciones.csv")
        low = float(np.nanmin([frame["real"].min(), frame["predicho"].min()]))
        high = float(np.nanmax([frame["real"].max(), frame["predicho"].max()]))
        ax.scatter(frame["real"], frame["predicho"], alpha=0.58, s=22, color=color)
        ax.plot([low, high], [low, high], "--", color="#B42318", linewidth=1.2)
        ax.set_title(display_names[name])
        ax.set_xlabel("Margen real (S/kg)")
        ax.set_ylabel("Margen predicho (S/kg)")
    fig.suptitle("Comparación de los cuatro modelos de margen exportador", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "comparacion_completa_modelos_margen.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    risk_names = [
        "07_escenario_random_forest",
        "08_escenario_hist_gradient_boosting",
        "09_escenario_logistic_regression",
        "12_escenario_prophet",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for name, color in zip(risk_names, palette):
        frame = pd.read_csv(output_dir / f"{name}_predicciones.csv")
        fpr, tpr, _ = roc_curve(frame["real"], frame["probabilidad_riesgo"])
        auc = roc_auc_score(frame["real"], frame["probabilidad_riesgo"])
        axes[0].plot(fpr, tpr, linewidth=2, color=color, label=f"{display_names[name]} (AUC={auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_title("Curvas ROC de todos los modelos de riesgo")
    axes[0].set_xlabel("Tasa de falsos positivos")
    axes[0].set_ylabel("Tasa de verdaderos positivos")
    axes[0].legend(fontsize=8)
    risk_long = cls_df.melt(
        id_vars="modelo",
        value_vars=["Precision", "Recall", "ROC AUC"],
        var_name="metrica",
        value_name="valor",
    )
    risk_long["modelo"] = risk_long["modelo"].replace(
        {
            "Hist Gradient Boosting": "HistGradientBoosting",
            "Logistic Regression": "Regresión logística",
        }
    )
    sns.barplot(data=risk_long, x="valor", y="modelo", hue="metrica", ax=axes[1])
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Precisión, recall y ROC AUC")
    axes[1].set_xlabel("Valor")
    axes[1].set_ylabel("")
    fig.tight_layout()
    fig.savefig(figures_dir / "comparacion_completa_modelos_riesgo.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\nResumen guardado en: {summary_path}")
