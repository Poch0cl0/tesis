from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import build_export_margin_dataset, load_fob_dataset, load_scenario_dataset


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def save_fob_dataset():
    df, _, _, target = load_fob_dataset()
    cols = [
        "fecha",
        "fecha_objetivo",
        "fecha_futura",
        "destino",
        "temporada",
        "precio_fob_por_kilogramo",
        "semana_iso",
        "log_volumen_exportado",
        "precio_lag_1",
        "precio_prom_movil_4",
        target,
    ]
    out = df[cols].sort_values(["fecha", "destino"]).reset_index(drop=True).copy()
    path = OUTPUTS / "dataset_prediccion_fob_minimo.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path, out.shape


def save_margin_dataset():
    df, _, _, target = build_export_margin_dataset()
    cols = [
        "ANIO",
        "mes_venta",
        "precio_fob_usd_kg",
        "region",
        "provincia",
        "rendimiento_kg_ha",
        "porcentaje_vendido",
        "tipo_conduccion_cultivo",
        target,
    ]
    out = df[cols].dropna(subset=[target]).copy()
    out["fecha"] = pd.to_datetime(
        out["ANIO"].astype(str) + "-" + out["mes_venta"].astype(int).astype(str).str.zfill(2) + "-01"
    )
    out = out.drop(columns=["ANIO", "mes_venta"])
    out.insert(0, "fecha", out.pop("fecha"))
    out["region"] = out["region"].fillna("No reportado")
    out["provincia"] = out["provincia"].fillna("No reportado")
    out["tipo_conduccion_cultivo"] = out["tipo_conduccion_cultivo"].fillna("No reportado")
    path = OUTPUTS / "dataset_prediccion_margen_minimo.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path, out.shape


def save_scenario_dataset():
    df, _, _, target = load_scenario_dataset()
    cols = [
        "ANIO",
        "mes_venta",
        "precio_fob_usd_kg",
        "region",
        "rendimiento_kg_ha",
        "porcentaje_vendido",
        "sequia",
        "plagas_enfermedades",
        target,
    ]
    out = df[cols].dropna(subset=[target]).copy()
    out["fecha"] = pd.to_datetime(
        out["ANIO"].astype(str) + "-" + out["mes_venta"].astype(int).astype(str).str.zfill(2) + "-01"
    )
    out = out.drop(columns=["ANIO", "mes_venta"])
    out.insert(0, "fecha", out.pop("fecha"))
    out["region"] = out["region"].fillna("No reportado")
    for col in ["sequia", "plagas_enfermedades"]:
        out[col] = out[col].astype("string").str.strip().replace("", "No reportado").fillna("No reportado")
    path = OUTPUTS / "dataset_prediccion_escenarios_minimo.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path, out.shape


if __name__ == "__main__":
    OUTPUTS.mkdir(exist_ok=True)
    for path, shape in [save_fob_dataset(), save_margin_dataset(), save_scenario_dataset()]:
        print(f"{path} -> filas={shape[0]}, columnas={shape[1]}")
