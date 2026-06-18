from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PALTO_CODE = "1162"
LA_LIBERTAD_CODE = "13"
YEARS = ["2021", "2022", "2023", "2024"]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = []
    for col in df.columns:
        cleaned.append(col.encode("latin1").replace(b"\xef\xbb\xbf", b"").decode("latin1").strip())
    df.columns = cleaned
    return df


def read_source_csv(path: Path) -> pd.DataFrame:
    return clean_columns(pd.read_csv(path, dtype=str, encoding="latin1", low_memory=False))


def numeric(series: pd.Series | None, index=None) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index)
    return pd.to_numeric(series.replace("", np.nan).replace(" ", np.nan), errors="coerce")


def detect_factor(df: pd.DataFrame) -> pd.DataFrame:
    factor_cols = [col for col in df.columns if "FACTOR" in col.upper()]
    if factor_cols and factor_cols[0] != "FACTOR_PRODUCTOR":
        df = df.rename(columns={factor_cols[0]: "FACTOR_PRODUCTOR"})
    return df


def detect_key_cols(df: pd.DataFrame) -> list[str]:
    keys_2024 = ["ANIO", "CCDD", "CCPP", "CCDI", "NSEGM", "ID_PROD", "UA"]
    keys_2021 = ["ANIO", "CCDD", "CCPP", "CCDI", "CONGLOMERADO", "NSELUA", "UA"]
    if all(col in df.columns for col in keys_2024):
        return keys_2024
    return [col for col in keys_2021 if col in df.columns]


def amount_with_decimal(df: pd.DataFrame, whole: str, decimal: str, equiv_kg: str | None = None) -> pd.Series:
    value = numeric(df.get(whole), df.index).fillna(0) + numeric(df.get(decimal), df.index).fillna(0) / 1000
    if equiv_kg:
        value = value * numeric(df.get(equiv_kg), df.index).fillna(0)
    return value


def first_non_empty(series: pd.Series):
    non_empty = series.dropna()
    non_empty = non_empty[~non_empty.isin(["", " "])]
    return non_empty.iloc[0] if len(non_empty) else np.nan


def build_enriched_cost_dataset(base_dir: str | Path = ".", department_code: str | None = LA_LIBERTAD_CODE) -> pd.DataFrame:
    base_dir = Path(base_dir)
    frames = []

    for year in YEARS:
        folder = base_dir / "costos_produccion" / year
        prod = detect_factor(read_source_csv(folder / "Produccion.csv"))
        cost = detect_factor(read_source_csv(folder / "Costos_Produccion.csv"))

        prod_filter = prod["P204_COD"] == PALTO_CODE
        cost_filter = cost["P234_COD"] == PALTO_CODE
        if department_code is not None:
            prod_filter &= prod["CCDD"] == department_code
            cost_filter &= cost["CCDD"] == department_code
        prod = prod[prod_filter].copy()
        cost = cost[cost_filter].copy()
        if prod.empty or cost.empty:
            continue

        key_cols = detect_key_cols(cost)

        prod["produccion_kg"] = amount_with_decimal(prod, "P219_CANT_1", "P219_CANT_2", "P219_EQUIV_KG")
        prod["venta_kg"] = amount_with_decimal(prod, "P220_1_CANT_1", "P220_1_CANT_2", "P219_EQUIV_KG")
        prod["area_cosechada_ha"] = numeric(prod.get("P217_SUP_ha"), prod.index)
        prod["plantas_cosechadas"] = numeric(prod.get("P218"), prod.index)
        prod["precio_chacra_kg"] = numeric(prod.get("P220_1_PRE_KG"), prod.index)
        prod["venta_valor_soles"] = numeric(prod.get("P220_1_VAL"), prod.index)
        prod["area_sembrada_ha"] = amount_with_decimal(prod, "P210_SUP_1", "P210_SUP_2")
        prod["factor_prod"] = numeric(prod.get("FACTOR_PRODUCTOR"), prod.index)

        keep_first = [
            "P204_TIPO", "P205_TOT", "P206_INI_MES", "P206_INI_ANIO", "P207_FIN_MES", "P207_FIN_ANIO",
            "P208", "P209_MES", "P209_ANIO", "P211_1", "P211_2", "P211_3", "P211_4", "P211_5",
            "P211_8", "P212", "P213", "P214", "P215", "P221_1", "P221_2", "P222_1", "P222_2",
            "P222_3", "P222_4", "P222_5", "P222_6", "P222_7", "P223_1", "P223_2", "P223_3",
            "P223_4", "P223_5", "P223_6", "P223A", "P223B_1", "P223B_2", "P223B_3", "P223B_4",
            "P223B_5", "P223B_6", "P223B_7", "P223B_8", "P219_UM",
        ]
        aggregations = {
            "region": ("NOMBREDD", first_non_empty),
            "provincia": ("NOMBREPV", first_non_empty),
            "distrito": ("NOMBREDI", first_non_empty),
            "produccion_kg": ("produccion_kg", "sum"),
            "venta_kg": ("venta_kg", "sum"),
            "area_cosechada_ha": ("area_cosechada_ha", "sum"),
            "area_sembrada_ha": ("area_sembrada_ha", "sum"),
            "plantas_cosechadas": ("plantas_cosechadas", "sum"),
            "venta_valor_soles": ("venta_valor_soles", "sum"),
            "precio_chacra_kg": ("precio_chacra_kg", "mean"),
            "factor_prod": ("factor_prod", "first"),
        }
        aggregations.update({col: (col, first_non_empty) for col in keep_first if col in prod.columns})
        prod_agg = prod.groupby(key_cols).agg(**aggregations).reset_index()

        for col in ["P235_VAL", "P237_VAL", "P239", "P241"]:
            cost[col] = numeric(cost.get(col), cost.index).fillna(0)
        cost["factor_cost"] = numeric(cost.get("FACTOR_PRODUCTOR"), cost.index)
        cost_agg = cost.groupby(key_cols).agg(
            costo_semilla=("P235_VAL", "sum"),
            costo_abono=("P237_VAL", "sum"),
            costo_fertilizantes=("P239", "sum"),
            costo_plaguicidas=("P241", "sum"),
            factor_cost=("factor_cost", "first"),
        ).reset_index()

        merged = cost_agg.merge(prod_agg, on=key_cols, how="inner")
        merged["costo_total"] = (
            merged["costo_semilla"]
            + merged["costo_abono"]
            + merged["costo_fertilizantes"]
            + merged["costo_plaguicidas"]
        )
        merged = merged[(merged["costo_total"] > 0) & (merged["produccion_kg"] > 0)].copy()
        merged["costo_por_kg"] = merged["costo_total"] / merged["produccion_kg"]
        merged["FACTOR_PRODUCTOR"] = merged["factor_cost"].fillna(merged["factor_prod"])
        merged["rendimiento_kg_ha"] = merged["produccion_kg"] / merged["area_cosechada_ha"].replace(0, np.nan)
        merged["densidad_plantas_ha"] = merged["plantas_cosechadas"] / merged["area_cosechada_ha"].replace(0, np.nan)
        merged["porcentaje_vendido"] = merged["venta_kg"] / merged["produccion_kg"].replace(0, np.nan)
        merged["costo_abono_kg"] = merged["costo_abono"] / merged["produccion_kg"]
        merged["costo_fertilizantes_kg"] = merged["costo_fertilizantes"] / merged["produccion_kg"]
        merged["costo_plaguicidas_kg"] = merged["costo_plaguicidas"] / merged["produccion_kg"]
        merged["participacion_fertilizantes"] = merged["costo_fertilizantes"] / merged["costo_total"]
        merged["participacion_plaguicidas"] = merged["costo_plaguicidas"] / merged["costo_total"]
        merged["duracion_cosecha_meses"] = (
            (numeric(merged.get("P207_FIN_ANIO"), merged.index) - numeric(merged.get("P206_INI_ANIO"), merged.index)) * 12
            + numeric(merged.get("P207_FIN_MES"), merged.index)
            - numeric(merged.get("P206_INI_MES"), merged.index)
            + 1
        )
        q1, q3 = merged["costo_por_kg"].quantile([0.25, 0.75])
        iqr = q3 - q1
        merged["es_outlier"] = ((merged["costo_por_kg"] < max(0, q1 - 1.5 * iqr)) | (merged["costo_por_kg"] > q3 + 1.5 * iqr)).astype(int)
        name_cols = [col for col in ["region", "provincia", "distrito"] if col in merged.columns]
        other_cols = [col for col in merged.columns if col not in name_cols]
        merged = merged[name_cols + other_cols]
        frames.append(merged)

    return pd.concat(frames, ignore_index=True)


def build_price_features(base_dir: str | Path = ".") -> pd.DataFrame:
    df = pd.read_csv(Path(base_dir) / "precio_palta_semanal.csv", parse_dates=["fecha"])
    df = df.sort_values(["destino", "fecha"]).copy()
    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month
    df["semana_iso"] = df["fecha"].dt.isocalendar().week.astype(int)
    df["trimestre"] = df["fecha"].dt.quarter
    df["log_volumen_exportado"] = np.log1p(df["volumen_exportado"])
    df["log_operaciones"] = np.log1p(df["operaciones"])
    df["precio_lag_1"] = df.groupby("destino")["precio_fob_por_kilogramo"].shift(1)
    df["precio_prom_movil_4"] = df.groupby("destino")["precio_fob_por_kilogramo"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )
    df["volatilidad_movil_4"] = df.groupby("destino")["precio_fob_por_kilogramo"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=2).std()
    )
    global_median = df["precio_fob_por_kilogramo"].median()
    df["precio_lag_1"] = df["precio_lag_1"].fillna(global_median)
    df["precio_prom_movil_4"] = df["precio_prom_movil_4"].fillna(global_median)
    df["volatilidad_movil_4"] = df["volatilidad_movil_4"].fillna(df["precio_fob_por_kilogramo"].std())
    return df


def train_models(base_dir: str | Path = ".", output_dir: str | Path = "models") -> dict:
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "outputs").mkdir(exist_ok=True)

    cost_df = build_enriched_cost_dataset(base_dir)
    price_df = build_price_features(base_dir)
    cost_df.to_csv(base_dir / "outputs" / "palto_modeling_base_la_libertad.csv", index=False)
    price_df.to_csv(base_dir / "outputs" / "precio_fob_features_semanal.csv", index=False)

    cost_features_num = [
        "produccion_kg", "venta_kg", "area_cosechada_ha", "plantas_cosechadas", "rendimiento_kg_ha",
        "densidad_plantas_ha", "porcentaje_vendido", "duracion_cosecha_meses", "precio_chacra_kg",
        "venta_valor_soles",
    ]
    cost_features_cat = [
        "ANIO", "CCPP", "CCDI", "P205_TOT", "P208", "P212", "P213", "P211_1", "P211_2", "P211_4",
        "P221_1", "P221_2", "P222_1", "P222_2", "P222_3", "P222_5", "P223_1", "P223_2", "P223_3",
        "P223A", "P223B_1", "P223B_6", "P223B_7",
    ]
    cost_model_df = cost_df[cost_df["es_outlier"] == 0].copy()
    X_cost = cost_model_df[cost_features_num + cost_features_cat]
    y_cost = np.log1p(cost_model_df["costo_por_kg"])
    X_train, X_test, y_train, y_test = train_test_split(X_cost, y_cost, test_size=0.25, random_state=42)
    cost_model = make_regression_pipeline(cost_features_num, cost_features_cat)
    cost_model.fit(X_train, y_train)
    pred_cost = np.expm1(cost_model.predict(X_test))
    y_cost_test = np.expm1(y_test)

    price_features_num = [
        "anio", "mes", "semana_iso", "trimestre", "log_volumen_exportado", "log_operaciones",
        "precio_lag_1", "precio_prom_movil_4", "volatilidad_movil_4",
    ]
    price_features_cat = ["temporada", "destino"]
    X_price = price_df[price_features_num + price_features_cat]
    y_price = price_df["precio_fob_por_kilogramo"]
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_price, y_price, test_size=0.20, random_state=42)
    price_model = make_regression_pipeline(price_features_num, price_features_cat, n_estimators=300)
    price_model.fit(X_train_p, y_train_p)
    pred_price = price_model.predict(X_test_p)

    risk_model = None
    risk_metrics = None
    risk_target = cost_df["P223A"].replace({"1": 1, "2": 0})
    risk_mask = risk_target.isin([0, 1])
    if risk_mask.sum() >= 100 and risk_target[risk_mask].nunique() == 2:
        risk_features_num = ["area_cosechada_ha", "plantas_cosechadas", "rendimiento_kg_ha", "densidad_plantas_ha", "duracion_cosecha_meses"]
        risk_features_cat = ["ANIO", "CCPP", "CCDI", "P208", "P212", "P213", "P211_1", "P211_2", "P211_4", "P211_8"]
        X_risk = cost_df.loc[risk_mask, risk_features_num + risk_features_cat]
        y_risk = risk_target.loc[risk_mask].astype(int)
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_risk, y_risk, test_size=0.25, random_state=42, stratify=y_risk)
        risk_model = make_classifier_pipeline(risk_features_num, risk_features_cat)
        risk_model.fit(X_train_r, y_train_r)
        pred_risk = risk_model.predict(X_test_r)
        prob_risk = risk_model.predict_proba(X_test_r)[:, 1]
        risk_metrics = {
            "accuracy": float(accuracy_score(y_test_r, pred_risk)),
            "roc_auc": float(roc_auc_score(y_test_r, prob_risk)),
            "rows": int(risk_mask.sum()),
        }
        joblib.dump(risk_model, output_dir / "risk_model.joblib")

    joblib.dump(cost_model, output_dir / "cost_model.joblib")
    joblib.dump(price_model, output_dir / "price_model.joblib")

    metadata = {
        "cost_features_num": cost_features_num,
        "cost_features_cat": cost_features_cat,
        "price_features_num": price_features_num,
        "price_features_cat": price_features_cat,
        "risk_features_num": ["area_cosechada_ha", "plantas_cosechadas", "rendimiento_kg_ha", "densidad_plantas_ha", "duracion_cosecha_meses"],
        "risk_features_cat": ["ANIO", "CCPP", "CCDI", "P208", "P212", "P213", "P211_1", "P211_2", "P211_4", "P211_8"],
        "destinos": sorted(price_df["destino"].dropna().unique().tolist()),
        "temporadas": sorted(price_df["temporada"].dropna().unique().tolist()),
        "latest_price_date": str(price_df["fecha"].max().date()),
        "cost_rows": int(len(cost_df)),
        "price_rows": int(len(price_df)),
    }
    metrics = {
        "cost_model": {
            "mae_soles_kg": float(mean_absolute_error(y_cost_test, pred_cost)),
            "r2": float(r2_score(y_cost_test, pred_cost)),
            "rows_trainable_no_outliers": int(len(cost_model_df)),
        },
        "price_model": {
            "mae_usd_kg": float(mean_absolute_error(y_test_p, pred_price)),
            "r2": float(r2_score(y_test_p, pred_price)),
            "rows": int(len(price_df)),
        },
        "risk_model": risk_metrics,
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"metrics": metrics, "metadata": metadata, "cost_df": cost_df, "price_df": price_df}


def make_regression_pipeline(num_features: list[str], cat_features: list[str], n_estimators: int = 250) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_features),
        ]
    )
    model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=3, random_state=42, n_jobs=-1)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def make_classifier_pipeline(num_features: list[str], cat_features: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_features),
        ]
    )
    model = RandomForestClassifier(n_estimators=250, min_samples_leaf=3, random_state=42, n_jobs=-1, class_weight="balanced")
    return Pipeline([("preprocess", preprocess), ("model", model)])


if __name__ == "__main__":
    result = train_models(".")
    print(json.dumps(result["metrics"], indent=2))
