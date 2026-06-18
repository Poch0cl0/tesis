from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
from prophet import Prophet
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from common import safe_mape, save_classifier_artifacts, save_joblib_atomic, save_regression_artifacts


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"


def _save(name: str, bundle: dict, metrics: dict):
    MODELS_DIR.mkdir(exist_ok=True)
    save_joblib_atomic(MODELS_DIR / f"{name}.joblib", bundle)
    (MODELS_DIR / f"{name}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=True))


def _encode_categoricals(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    combined = pd.concat([train[categorical], test[categorical]], ignore_index=True).fillna("No reportado").astype(str)
    for col in categorical:
        combined[col] = combined[col].map(
            lambda value: "".join(ch if ch.isalnum() or ch in " _-" else "_" for ch in value).strip() or "No reportado"
        )
    encoded = pd.get_dummies(combined, columns=categorical, dtype=float)
    encoded.columns = [
        "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in col).strip("_") or f"categoria_{idx}"
        for idx, col in enumerate(encoded.columns)
    ]
    train_encoded = encoded.iloc[: len(train)].set_index(train.index)
    test_encoded = encoded.iloc[len(train) :].set_index(test.index)
    return train_encoded, test_encoded, encoded.columns.tolist()


def train_fob_prophet():
    train_prophet_regression(
        name="10_fob_prophet",
        csv_name="dataset_prediccion_fob_minimo.csv",
        target="target_fob_6_semanas",
        numeric=[
            "precio_fob_por_kilogramo",
            "semana_iso",
            "log_volumen_exportado",
            "precio_lag_1",
            "precio_prom_movil_4",
        ],
        categorical=["destino", "temporada"],
        yearly_seasonality=True,
        time_split=True,
    )


def train_prophet_regression(
    name: str,
    csv_name: str,
    target: str,
    numeric: list[str],
    categorical: list[str],
    yearly_seasonality: bool = False,
    time_split: bool = False,
):
    df = pd.read_csv(OUTPUTS_DIR / csv_name, parse_dates=["fecha"]).sort_values("fecha").copy()
    if time_split:
        split = max(1, int(len(df) * 0.8))
        train, test = df.iloc[:split].copy(), df.iloc[split:].copy()
        split_method = "temporal_80_20"
    else:
        train, test = train_test_split(df, test_size=0.25, random_state=42)
        train = train.sort_values("fecha").copy()
        test = test.sort_values("fecha").copy()
        split_method = "aleatoria_75_25_random_state_42"
    train_cat, test_cat, dummy_cols = _encode_categoricals(train, test, categorical)
    train_model = pd.concat([train[["fecha", target] + numeric], train_cat], axis=1)
    test_model = pd.concat([test[["fecha", target] + numeric], test_cat], axis=1)
    regressors = numeric + dummy_cols
    for col in numeric:
        median = pd.to_numeric(train_model[col], errors="coerce").median()
        train_model[col] = pd.to_numeric(train_model[col], errors="coerce").fillna(median)
        test_model[col] = pd.to_numeric(test_model[col], errors="coerce").fillna(median)
    model = Prophet(
        weekly_seasonality=False,
        daily_seasonality=False,
        yearly_seasonality=yearly_seasonality,
        seasonality_mode="additive",
        changepoint_prior_scale=0.08,
        seasonality_prior_scale=5.0,
    )
    for col in regressors:
        model.add_regressor(col)
    fit = train_model.rename(columns={"fecha": "ds", target: "y"})[["ds", "y"] + regressors]
    model.fit(fit)
    future = test_model.rename(columns={"fecha": "ds"})[["ds"] + regressors]
    pred = model.predict(future)["yhat"].to_numpy()
    actual = test_model[target].astype(float).to_numpy()
    metrics = {
        "model_name": name,
        "target": target,
        "train_rows": len(train),
        "test_rows": len(test),
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "mape_pct": safe_mape(actual, pred),
        "r2": float(r2_score(actual, pred)),
        "split_method": split_method,
        "regressors": regressors,
    }
    _save(
        name,
        {
            "model": model,
            "numeric": numeric,
            "categorical": categorical,
            "dummy_columns": dummy_cols,
        },
        metrics,
    )
    unit = "USD/kg" if "fob" in name else "S/kg"
    save_regression_artifacts(name, actual, pred, dates=test_model["fecha"], unit=unit)


def train_prophet_scenario():
    name = "12_escenario_prophet"
    target = "riesgo_margen_bajo"
    numeric = ["precio_fob_usd_kg", "rendimiento_kg_ha", "porcentaje_vendido"]
    categorical = ["region", "sequia", "plagas_enfermedades"]
    df = pd.read_csv(OUTPUTS_DIR / "dataset_prediccion_escenarios_minimo.csv", parse_dates=["fecha"])
    df = df.sort_values("fecha").copy()
    train, test = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df[target].astype(int),
    )
    train = train.sort_values("fecha").copy()
    test = test.sort_values("fecha").copy()
    train_cat, test_cat, dummy_cols = _encode_categoricals(train, test, categorical)
    train_model = pd.concat([train[["fecha", target] + numeric], train_cat], axis=1)
    test_model = pd.concat([test[["fecha", target] + numeric], test_cat], axis=1)
    regressors = numeric + dummy_cols
    for col in numeric:
        median = pd.to_numeric(train_model[col], errors="coerce").median()
        train_model[col] = pd.to_numeric(train_model[col], errors="coerce").fillna(median)
        test_model[col] = pd.to_numeric(test_model[col], errors="coerce").fillna(median)
    model = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=False)
    for col in regressors:
        model.add_regressor(col)
    fit = train_model.rename(columns={"fecha": "ds", target: "y"})[["ds", "y"] + regressors]
    model.fit(fit)
    future = test_model.rename(columns={"fecha": "ds"})[["ds"] + regressors]
    probability = np.clip(model.predict(future)["yhat"].to_numpy(), 0, 1)
    prediction = (probability >= 0.5).astype(int)
    actual = test_model[target].astype(int).to_numpy()
    metrics = {
        "model_name": name,
        "target": target,
        "train_rows": len(train),
        "test_rows": len(test),
        "accuracy": float(accuracy_score(actual, prediction)),
        "precision": float(precision_score(actual, prediction, zero_division=0)),
        "recall": float(recall_score(actual, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probability)),
        "regressors": regressors,
        "note": "Prophet se usa como modelo probabilistico continuo con umbral 0.5.",
    }
    _save(
        name,
        {
            "model": model,
            "numeric": numeric,
            "categorical": categorical,
            "dummy_columns": dummy_cols,
            "classification_threshold": 0.5,
        },
        metrics,
    )
    save_classifier_artifacts(name, actual, prediction, probability)
