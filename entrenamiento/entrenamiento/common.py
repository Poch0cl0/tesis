from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "figures"
YEARS = ["2021", "2022", "2023", "2024"]
PALTO_CODE = "1162"
DEFAULT_TIPO_CAMBIO = 3.75


def clean_columns(columns):
    return [col.encode("latin1").replace(b"\xef\xbb\xbf", b"").decode("latin1").strip() for col in columns]


def normalize_geo(value):
    if pd.isna(value):
        return value
    text = str(value).strip().upper()
    replacements = {
        "APUR?MAC": "APURIMAC",
        "HU?NUCO": "HUANUCO",
        "JUN?N": "JUNIN",
        "SAN MART?N": "SAN MARTIN",
        "APURIMAC": "APURIMAC",
        "HUANUCO": "HUANUCO",
        "JUNIN": "JUNIN",
        "SAN MARTIN": "SAN MARTIN",
        "Ñ": "N",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    return " ".join(text.split())


def read_columns(path: Path, wanted: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, dtype=str, encoding="latin1", nrows=0)
    raw_to_clean = dict(zip(header.columns, clean_columns(header.columns)))
    usecols = [raw for raw, clean in raw_to_clean.items() if clean in wanted]
    df = pd.read_csv(path, dtype=str, encoding="latin1", low_memory=False, usecols=usecols)
    df.columns = clean_columns(df.columns)
    return df


def numeric(series: pd.Series | None, index=None) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index)
    return pd.to_numeric(series.replace("", np.nan).replace(" ", np.nan), errors="coerce")


def amount_with_decimal(df: pd.DataFrame, whole: str, decimal: str, equiv_kg: str | None = None) -> pd.Series:
    value = numeric(df.get(whole), df.index).fillna(0) + numeric(df.get(decimal), df.index).fillna(0) / 1000
    if equiv_kg:
        value = value * numeric(df.get(equiv_kg), df.index).fillna(0)
    return value


def detect_factor(df: pd.DataFrame) -> pd.DataFrame:
    factor_cols = [col for col in df.columns if "FACTOR" in col.upper()]
    if factor_cols and factor_cols[0] != "FACTOR_PRODUCTOR":
        return df.rename(columns={factor_cols[0]: "FACTOR_PRODUCTOR"})
    return df


def detect_key_cols(df: pd.DataFrame) -> list[str]:
    keys_new = ["ANIO", "CCDD", "CCPP", "CCDI", "NSEGM", "ID_PROD", "UA"]
    keys_old = ["ANIO", "CCDD", "CCPP", "CCDI", "CONGLOMERADO", "NSELUA", "UA"]
    if all(col in df.columns for col in keys_new):
        return keys_new
    return [col for col in keys_old if col in df.columns]


def first_non_empty(series: pd.Series):
    non_empty = series.dropna()
    non_empty = non_empty[~non_empty.isin(["", " "])]
    return non_empty.iloc[0] if len(non_empty) else np.nan


def make_preprocessor(num_features: list[str], cat_features: list[str], scale_numeric: bool = False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), num_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_features,
            ),
        ],
        remainder="drop",
    )


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_joblib_atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    joblib.dump(payload, temporary)
    os.replace(temporary, path)


def safe_mape(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted) & (np.abs(actual) > 1e-8)
    if not mask.any():
        return float("nan")
    return float(mean_absolute_percentage_error(actual[mask], predicted[mask]) * 100)


def save_regression_artifacts(
    model_name: str,
    actual,
    predicted,
    dates=None,
    unit: str = "",
):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residuals = actual - predicted
    result = pd.DataFrame({"real": actual, "predicho": predicted, "residuo": residuals})
    if dates is not None:
        result.insert(0, "fecha", pd.to_datetime(dates).to_numpy())
    result.to_csv(OUTPUTS_DIR / f"{model_name}_predicciones.csv", index=False, encoding="utf-8-sig")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    low = float(np.nanmin([actual.min(), predicted.min()]))
    high = float(np.nanmax([actual.max(), predicted.max()]))
    axes[0].scatter(actual, predicted, alpha=0.55, s=24, color="#176B87")
    axes[0].plot([low, high], [low, high], "--", color="#B42318", linewidth=1.5)
    axes[0].set_title("Valores reales frente a predichos")
    axes[0].set_xlabel(f"Real {unit}".strip())
    axes[0].set_ylabel(f"Predicho {unit}".strip())
    sns.histplot(residuals, bins=25, kde=True, ax=axes[1], color="#2E8B57")
    axes[1].axvline(0, color="#B42318", linestyle="--", linewidth=1.5)
    axes[1].set_title("Distribución de residuos")
    axes[1].set_xlabel(f"Error real - predicho {unit}".strip())
    fig.suptitle(model_name.replace("_", " ").title(), fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{model_name}_resultados.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    if dates is not None:
        ordered = (
            result.groupby("fecha", as_index=False)[["real", "predicho"]]
            .mean()
            .sort_values("fecha")
            .reset_index(drop=True)
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(ordered["fecha"], ordered["real"], label="Real", linewidth=1.7, color="#1F4E79")
        ax.plot(ordered["fecha"], ordered["predicho"], label="Predicho", linewidth=1.5, color="#C65911")
        ax.set_title(f"Serie de prueba - {model_name.replace('_', ' ').title()}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel(unit)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{model_name}_serie_prueba.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def save_classifier_artifacts(model_name: str, actual, predicted, probability):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    probability = np.asarray(probability, dtype=float)
    pd.DataFrame(
        {"real": actual, "predicho": predicted, "probabilidad_riesgo": probability}
    ).to_csv(OUTPUTS_DIR / f"{model_name}_predicciones.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(actual, predicted, labels=[0, 1])
    fpr, tpr, _ = roc_curve(actual, probability)
    auc = roc_auc_score(actual, probability)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
    axes[0].set_title("Matriz de confusión")
    axes[0].set_xlabel("Predicción")
    axes[0].set_ylabel("Valor real")
    axes[0].set_xticklabels(["Normal", "Riesgo"])
    axes[0].set_yticklabels(["Normal", "Riesgo"], rotation=0)
    axes[1].plot(fpr, tpr, color="#176B87", linewidth=2, label=f"AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set_title("Curva ROC")
    axes[1].set_xlabel("Tasa de falsos positivos")
    axes[1].set_ylabel("Tasa de verdaderos positivos")
    axes[1].legend(loc="lower right")
    fig.suptitle(model_name.replace("_", " ").title(), fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{model_name}_resultados.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def weighted_average(group: pd.DataFrame, value_col: str, weight_col: str):
    values = pd.to_numeric(group[value_col], errors="coerce")
    weights = pd.to_numeric(group[weight_col], errors="coerce").fillna(0)
    mask = values.notna() & (weights > 0)
    if mask.any():
        return float(np.average(values[mask], weights=weights[mask]))
    return float(values.mean())


def load_fob_dataset(horizon_weeks: int = 6) -> tuple[pd.DataFrame, list[str], list[str], str]:
    df = pd.read_csv(ROOT / "precio_palta_semanal.csv", parse_dates=["fecha"])
    df = df.sort_values(["destino", "fecha"]).copy()
    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month
    df["semana_iso"] = df["fecha"].dt.isocalendar().week.astype(int)
    df["trimestre"] = df["fecha"].dt.quarter
    df["log_volumen_exportado"] = np.log1p(df["volumen_exportado"])
    df["log_operaciones"] = np.log1p(df["operaciones"])
    grouped = df.groupby("destino", group_keys=False)
    df["precio_lag_1"] = grouped["precio_fob_por_kilogramo"].shift(1)
    df["precio_prom_movil_4"] = grouped["precio_fob_por_kilogramo"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )
    df["volatilidad_movil_4"] = grouped["precio_fob_por_kilogramo"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=2).std()
    )
    target_frames = []
    tolerance = pd.Timedelta(days=14)
    for destino, group in df.groupby("destino", sort=False):
        group = group.sort_values("fecha").copy()
        future = group[["fecha", "precio_fob_por_kilogramo"]].rename(
            columns={"fecha": "fecha_futura", "precio_fob_por_kilogramo": "target_fob_6_semanas"}
        )
        requested = group[["fecha"]].copy()
        requested["fecha_objetivo"] = requested["fecha"] + pd.Timedelta(weeks=horizon_weeks)
        matched = pd.merge_asof(
            requested.sort_values("fecha_objetivo"),
            future.sort_values("fecha_futura"),
            left_on="fecha_objetivo",
            right_on="fecha_futura",
            direction="nearest",
            tolerance=tolerance,
        )
        group["fecha_objetivo"] = matched["fecha_objetivo"].to_numpy()
        group["fecha_futura"] = matched["fecha_futura"].to_numpy()
        group["target_fob_6_semanas"] = matched["target_fob_6_semanas"].to_numpy()
        target_frames.append(group)
    df = pd.concat(target_frames, ignore_index=True)
    median_price = df["precio_fob_por_kilogramo"].median()
    df["precio_lag_1"] = df["precio_lag_1"].fillna(median_price)
    df["precio_prom_movil_4"] = df["precio_prom_movil_4"].fillna(median_price)
    df["volatilidad_movil_4"] = df["volatilidad_movil_4"].fillna(df["precio_fob_por_kilogramo"].std())
    df = df.dropna(subset=["target_fob_6_semanas"]).copy()
    num_features = [
        "precio_fob_por_kilogramo",
        "anio",
        "mes",
        "semana_iso",
        "trimestre",
        "log_volumen_exportado",
        "log_operaciones",
        "precio_lag_1",
        "precio_prom_movil_4",
        "volatilidad_movil_4",
    ]
    cat_features = ["temporada", "destino"]
    return df, num_features, cat_features, "target_fob_6_semanas"


def load_fob_minimal_dataset() -> tuple[pd.DataFrame, list[str], list[str], str]:
    df = pd.read_csv(OUTPUTS_DIR / "dataset_prediccion_fob_minimo.csv", parse_dates=["fecha"])
    df = df.sort_values(["fecha", "destino"]).reset_index(drop=True)
    num_features = [
        "precio_fob_por_kilogramo",
        "semana_iso",
        "log_volumen_exportado",
        "precio_lag_1",
        "precio_prom_movil_4",
    ]
    cat_features = ["destino", "temporada"]
    return df, num_features, cat_features, "target_fob_6_semanas"


def load_margin_minimal_dataset() -> tuple[pd.DataFrame, list[str], list[str], str]:
    df = pd.read_csv(OUTPUTS_DIR / "dataset_prediccion_margen_minimo.csv", parse_dates=["fecha"])
    num_features = ["precio_fob_usd_kg", "rendimiento_kg_ha", "porcentaje_vendido"]
    cat_features = ["region", "provincia", "tipo_conduccion_cultivo"]
    return df, num_features, cat_features, "margen_exportador_soles_kg"


def load_scenario_minimal_dataset() -> tuple[pd.DataFrame, list[str], list[str], str]:
    df = pd.read_csv(OUTPUTS_DIR / "dataset_prediccion_escenarios_minimo.csv", parse_dates=["fecha"])
    num_features = ["precio_fob_usd_kg", "rendimiento_kg_ha", "porcentaje_vendido"]
    cat_features = ["region", "sequia", "plagas_enfermedades"]
    return df, num_features, cat_features, "riesgo_margen_bajo"


def monthly_fob_prices() -> pd.DataFrame:
    price = pd.read_csv(ROOT / "precio_palta_semanal.csv", parse_dates=["fecha"])
    price["ANIO"] = price["fecha"].dt.year.astype(str)
    price["mes_venta"] = price["fecha"].dt.month.astype(int)
    rows = []
    for (year, month), group in price.groupby(["ANIO", "mes_venta"]):
        rows.append(
            {
                "ANIO": year,
                "mes_venta": month,
                "precio_fob_usd_kg": weighted_average(group, "precio_fob_por_kilogramo", "volumen_exportado"),
                "volumen_exportado_mes": float(group["volumen_exportado"].sum()),
                "operaciones_mes": float(group["operaciones"].sum()),
                "destinos_activos_mes": int(group["destino"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def build_export_margin_dataset() -> tuple[pd.DataFrame, list[str], list[str], str]:
    frames = []
    cost_cols = [
        "ANIO",
        "CCDD",
        "NOMBREDD",
        "CCPP",
        "NOMBREPV",
        "CCDI",
        "NOMBREDI",
        "CONGLOMERADO",
        "NSELUA",
        "NSEGM",
        "ID_PROD",
        "UA",
        "FACTOR",
        "FACTOR_PRODUCTOR",
        "P234_COD",
        "P235_VAL",
        "P237_VAL",
        "P239",
        "P241",
    ]
    prod_cols = [
        "ANIO",
        "CCDD",
        "NOMBREDD",
        "CCPP",
        "NOMBREPV",
        "CCDI",
        "NOMBREDI",
        "CONGLOMERADO",
        "NSELUA",
        "NSEGM",
        "ID_PROD",
        "UA",
        "FACTOR",
        "FACTOR_PRODUCTOR",
        "P204_COD",
        "P217_SUP_ha",
        "P218",
        "P219_CANT_1",
        "P219_CANT_2",
        "P219_EQUIV_KG",
        "P220_1_CANT_1",
        "P220_1_CANT_2",
        "P220_1_VAL",
        "P220_1_PRE_KG",
        "P206_INI_MES",
        "P206_INI_ANIO",
        "P207_FIN_MES",
        "P207_FIN_ANIO",
        "P208",
        "P211_8",
        "P212",
        "P213",
        "P223_3",
        "P223A",
        "P223B_1",
        "P223B_2",
        "P223B_3",
        "P223B_6",
        "P223B_7",
    ]
    for year in YEARS:
        folder = ROOT / "costos_produccion" / year
        cost = detect_factor(read_columns(folder / "Costos_Produccion.csv", cost_cols))
        prod = detect_factor(read_columns(folder / "Produccion.csv", prod_cols))
        cost = cost[cost["P234_COD"] == PALTO_CODE].copy()
        if "P223_3" not in prod.columns:
            continue
        prod = prod[(prod["P204_COD"] == PALTO_CODE) & (prod["P223_3"].astype(str).str.strip() == "1")].copy()
        if cost.empty or prod.empty:
            continue
        keys = detect_key_cols(cost)
        for col in ["P235_VAL", "P237_VAL", "P239", "P241"]:
            cost[col] = numeric(cost.get(col), cost.index).fillna(0)
        cost["factor_cost"] = numeric(cost.get("FACTOR_PRODUCTOR"), cost.index)
        cost_agg = cost.groupby(keys, dropna=False).agg(
            costo_semilla_soles=("P235_VAL", "sum"),
            costo_abono_soles=("P237_VAL", "sum"),
            costo_fertilizantes_soles=("P239", "sum"),
            costo_plaguicidas_soles=("P241", "sum"),
            factor_cost=("factor_cost", "first"),
        ).reset_index()

        prod["produccion_kg"] = amount_with_decimal(prod, "P219_CANT_1", "P219_CANT_2", "P219_EQUIV_KG")
        prod["venta_kg"] = amount_with_decimal(prod, "P220_1_CANT_1", "P220_1_CANT_2", "P219_EQUIV_KG")
        prod["area_cosechada_ha"] = numeric(prod.get("P217_SUP_ha"), prod.index)
        prod["plantas_cosechadas"] = numeric(prod.get("P218"), prod.index)
        prod["precio_chacra_soles_kg"] = numeric(prod.get("P220_1_PRE_KG"), prod.index)
        prod["valor_venta_soles"] = numeric(prod.get("P220_1_VAL"), prod.index)
        prod["factor_prod"] = numeric(prod.get("FACTOR_PRODUCTOR"), prod.index)
        prod["mes_venta"] = numeric(prod.get("P207_FIN_MES"), prod.index).fillna(numeric(prod.get("P206_INI_MES"), prod.index))
        prod["mes_venta"] = prod["mes_venta"].clip(lower=1, upper=12).fillna(6).astype(int)
        prod["duracion_cosecha_meses"] = (
            (numeric(prod.get("P207_FIN_ANIO"), prod.index) - numeric(prod.get("P206_INI_ANIO"), prod.index)) * 12
            + numeric(prod.get("P207_FIN_MES"), prod.index)
            - numeric(prod.get("P206_INI_MES"), prod.index)
            + 1
        )
        prod_agg = prod.groupby(keys, dropna=False).agg(
            region=("NOMBREDD", first_non_empty),
            provincia=("NOMBREPV", first_non_empty),
            distrito=("NOMBREDI", first_non_empty),
            mes_venta=("mes_venta", "first"),
            produccion_kg=("produccion_kg", "sum"),
            venta_kg=("venta_kg", "sum"),
            area_cosechada_ha=("area_cosechada_ha", "sum"),
            plantas_cosechadas=("plantas_cosechadas", "sum"),
            precio_chacra_soles_kg=("precio_chacra_soles_kg", "mean"),
            valor_venta_soles=("valor_venta_soles", "sum"),
            duracion_cosecha_meses=("duracion_cosecha_meses", "first"),
            tipo_conduccion_cultivo=("P208", "first"),
            considero_fertilidad_suelo=("P211_8", "first"),
            fuente_agua_riego=("P212", "first"),
            sistema_riego=("P213", "first"),
            produccion_afectada=("P223A", "first"),
            sequia=("P223B_1", "first"),
            bajas_temperaturas=("P223B_2", "first"),
            heladas=("P223B_3", "first"),
            lluvias_destiempo=("P223B_6", "first"),
            plagas_enfermedades=("P223B_7", "first"),
            factor_prod=("factor_prod", "first"),
        ).reset_index()

        merged = cost_agg.merge(prod_agg, on=keys, how="inner")
        merged["costo_total_soles"] = merged[
            ["costo_semilla_soles", "costo_abono_soles", "costo_fertilizantes_soles", "costo_plaguicidas_soles"]
        ].sum(axis=1)
        merged = merged[(merged["costo_total_soles"] > 0) & (merged["produccion_kg"] > 0)].copy()
        merged["costo_soles_kg"] = merged["costo_total_soles"] / merged["produccion_kg"]
        merged["rendimiento_kg_ha"] = merged["produccion_kg"] / merged["area_cosechada_ha"].replace(0, np.nan)
        merged["porcentaje_vendido"] = merged["venta_kg"] / merged["produccion_kg"].replace(0, np.nan)
        merged["densidad_plantas_ha"] = merged["plantas_cosechadas"] / merged["area_cosechada_ha"].replace(0, np.nan)
        merged["FACTOR_PRODUCTOR"] = merged["factor_cost"].fillna(merged["factor_prod"])
        frames.append(merged)

    if not frames:
        raise RuntimeError("No se encontraron registros de palto con mercado exterior.")

    df = pd.concat(frames, ignore_index=True)
    for col in ["region", "provincia", "distrito"]:
        df[col] = df[col].map(normalize_geo)
    df = df.merge(monthly_fob_prices(), on=["ANIO", "mes_venta"], how="left")
    df = df.dropna(subset=["precio_fob_usd_kg", "costo_soles_kg"]).copy()
    df["tipo_cambio"] = DEFAULT_TIPO_CAMBIO
    df["margen_exportador_soles_kg"] = df["precio_fob_usd_kg"] * df["tipo_cambio"] - df["costo_soles_kg"]
    q1, q3 = df["costo_soles_kg"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df["es_outlier_costo_kg"] = (
        (df["costo_soles_kg"] < max(0, q1 - 1.5 * iqr)) | (df["costo_soles_kg"] > q3 + 1.5 * iqr)
    ).astype(int)

    value_maps = {
        "tipo_conduccion_cultivo": {"1": "Homogeneo", "2": "Asociado", "3": "Disperso", "4": "Vergel"},
        "considero_fertilidad_suelo": {"1": "Si", "0": "No"},
        "fuente_agua_riego": {
            "1": "Lluvia/secano",
            "2": "Rio",
            "3": "Manantial/puquio",
            "4": "Pozo/agua subterranea",
            "5": "Reservorio",
            "6": "Embalse estacional",
            "7": "Otro",
        },
        "sistema_riego": {
            "1": "Exudacion",
            "2": "Goteo",
            "3": "Microaspersion",
            "4": "Aspersion",
            "5": "Multicompuertas",
            "6": "Mangas",
            "7": "Gravedad",
            "8": "Otro",
        },
        "produccion_afectada": {"1": "Si", "2": "No"},
        "sequia": {"1": "Si", "0": "No"},
        "bajas_temperaturas": {"1": "Si", "0": "No"},
        "heladas": {"1": "Si", "0": "No"},
        "lluvias_destiempo": {"1": "Si", "0": "No"},
        "plagas_enfermedades": {"1": "Si", "0": "No"},
    }
    for col, mapping in value_maps.items():
        df[col] = df[col].astype("string").map(mapping).fillna(df[col])

    num_features = [
        "precio_fob_usd_kg",
        "tipo_cambio",
        "rendimiento_kg_ha",
        "porcentaje_vendido",
        "area_cosechada_ha",
        "densidad_plantas_ha",
        "duracion_cosecha_meses",
        "volumen_exportado_mes",
        "operaciones_mes",
        "destinos_activos_mes",
    ]
    cat_features = [
        "ANIO",
        "mes_venta",
        "region",
        "provincia",
        "tipo_conduccion_cultivo",
        "considero_fertilidad_suelo",
        "fuente_agua_riego",
        "sistema_riego",
        "produccion_afectada",
        "sequia",
        "bajas_temperaturas",
        "heladas",
        "lluvias_destiempo",
        "plagas_enfermedades",
    ]
    return df, num_features, cat_features, "margen_exportador_soles_kg"


def load_scenario_dataset() -> tuple[pd.DataFrame, list[str], list[str], str]:
    df, num_features, cat_features, _ = build_export_margin_dataset()
    threshold = df["margen_exportador_soles_kg"].quantile(0.25)
    df["riesgo_margen_bajo"] = (df["margen_exportador_soles_kg"] <= threshold).astype(int)
    return df, num_features, cat_features, "riesgo_margen_bajo"


def train_regression(
    model_name: str,
    estimator,
    dataset_loader,
    scale_numeric: bool = False,
    time_split: bool = False,
) -> dict:
    MODELS_DIR.mkdir(exist_ok=True)
    df, num_features, cat_features, target = dataset_loader()
    df = df.dropna(subset=[target]).copy()
    X = df[num_features + cat_features]
    y = df[target].astype(float)
    if time_split and "fecha" in df.columns:
        ordered = df.sort_values("fecha").index
        split_at = max(1, int(len(ordered) * 0.8))
        train_idx, test_idx = ordered[:split_at], ordered[split_at:]
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    elif time_split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    pipeline = Pipeline(
        [
            ("preprocess", make_preprocessor(num_features, cat_features, scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    dates = df.loc[X_test.index, "fecha"] if "fecha" in df.columns else None
    metrics = {
        "model_name": model_name,
        "target": target,
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mape_pct": safe_mape(y_test, pred),
        "r2": float(r2_score(y_test, pred)),
        "features_num": num_features,
        "features_cat": cat_features,
    }
    save_joblib_atomic(MODELS_DIR / f"{model_name}.joblib", pipeline)
    save_json(MODELS_DIR / f"{model_name}_metrics.json", metrics)
    unit = "USD/kg" if "fob" in model_name else "S/kg"
    save_regression_artifacts(model_name, y_test, pred, dates=dates, unit=unit)
    return metrics


def train_classifier(
    model_name: str,
    estimator,
    dataset_loader,
    scale_numeric: bool = False,
) -> dict:
    MODELS_DIR.mkdir(exist_ok=True)
    df, num_features, cat_features, target = dataset_loader()
    df = df.dropna(subset=[target]).copy()
    X = df[num_features + cat_features]
    y = df[target].astype(int)
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
    pipeline = Pipeline(
        [
            ("preprocess", make_preprocessor(num_features, cat_features, scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else pred.astype(float)
    metrics = {
        "model_name": model_name,
        "target": target,
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "class_balance": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "features_num": num_features,
        "features_cat": cat_features,
    }
    if y_test.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, prob))
    save_joblib_atomic(MODELS_DIR / f"{model_name}.joblib", pipeline)
    save_json(MODELS_DIR / f"{model_name}_metrics.json", metrics)
    save_classifier_artifacts(model_name, y_test, pred, prob)
    return metrics


def print_metrics(metrics: dict):
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
