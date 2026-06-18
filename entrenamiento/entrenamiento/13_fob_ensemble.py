from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import safe_mape, save_json, save_regression_artifacts


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"
COMPONENTS = [
    "01_fob_random_forest",
    "02_fob_hist_gradient_boosting",
    "03_fob_elasticnet",
    "10_fob_prophet",
]


def train_fob_ensemble() -> dict:
    predictions = [
        pd.read_csv(OUTPUTS_DIR / f"{name}_predicciones.csv", parse_dates=["fecha"])
        for name in COMPONENTS
    ]
    reference = predictions[0][["fecha", "real"]].copy()
    for name, frame in zip(COMPONENTS, predictions):
        if len(frame) != len(reference) or not np.allclose(frame["real"], reference["real"]):
            raise ValueError(f"Las predicciones de {name} no comparten el mismo bloque de prueba.")

    matrix = np.column_stack([frame["predicho"].to_numpy(dtype=float) for frame in predictions])
    ensemble_prediction = matrix.mean(axis=1)
    actual = reference["real"].to_numpy(dtype=float)
    metrics = {
        "model_name": "13_fob_ensemble",
        "target": "target_fob_6_semanas",
        "train_rows": 1849,
        "test_rows": len(reference),
        "mae": float(mean_absolute_error(actual, ensemble_prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, ensemble_prediction))),
        "mape_pct": safe_mape(actual, ensemble_prediction),
        "r2": float(r2_score(actual, ensemble_prediction)),
        "components": COMPONENTS,
        "combination": "promedio_simple",
        "note": "Ensamble de cuatro modelos evaluados sobre el mismo bloque temporal de prueba.",
    }
    save_json(MODELS_DIR / "13_fob_ensemble_metrics.json", metrics)
    save_json(
        MODELS_DIR / "13_fob_ensemble.json",
        {"components": COMPONENTS, "weights": [0.25] * len(COMPONENTS)},
    )
    save_regression_artifacts(
        "13_fob_ensemble",
        actual,
        ensemble_prediction,
        dates=reference["fecha"],
        unit="USD/kg",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    train_fob_ensemble()
