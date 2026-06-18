from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"


P208_LABELS = {"1": "Homogéneo", "2": "Asociado", "3": "Disperso", "4": "Vergel"}
P212_LABELS = {"1": "Lluvia/secano", "2": "Río", "3": "Manantial/puquio", "4": "Pozo", "5": "Reservorio", "6": "Embalse estacional", "7": "Otro"}
P213_LABELS = {"1": "Exudación", "2": "Goteo", "3": "Microaspersión", "4": "Aspersión", "5": "Multicompuertas", "6": "Mangas", "7": "Gravedad", "8": "Otro"}


@st.cache_resource
def load_artifacts():
    cost_model = joblib.load(MODELS_DIR / "cost_model.joblib")
    price_model = joblib.load(MODELS_DIR / "price_model.joblib")
    risk_model_path = MODELS_DIR / "risk_model.joblib"
    risk_model = joblib.load(risk_model_path) if risk_model_path.exists() else None
    metadata = json.loads((MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))
    return cost_model, price_model, risk_model, metadata, metrics


@st.cache_data
def load_data():
    cost_df = pd.read_csv(OUTPUTS_DIR / "palto_modeling_base_la_libertad.csv")
    price_df = pd.read_csv(OUTPUTS_DIR / "precio_fob_features_semanal.csv", parse_dates=["fecha"])
    location_df = build_location_mapping()
    return cost_df, price_df, location_df


@st.cache_data
def build_location_mapping():
    frames = []
    for path in sorted((ROOT / "costos_produccion").glob("*/Produccion.csv")):
        df = pd.read_csv(path, dtype=str, encoding="latin1", low_memory=False)
        df.columns = [col.encode("latin1").replace(b"\xef\xbb\xbf", b"").decode("latin1").strip() for col in df.columns]
        subset = df[(df["P204_COD"] == "1162") & (df["CCDD"] == "13")].copy()
        if subset.empty:
            continue
        frames.append(subset[["CCDD", "NOMBREDD", "CCPP", "NOMBREPV", "CCDI", "NOMBREDI"]])
    if not frames:
        return pd.DataFrame(
            [{"CCDD": "13", "NOMBREDD": "LA LIBERTAD", "CCPP": "12", "NOMBREPV": "VIRU", "CCDI": "2", "NOMBREDI": "CHAO"}]
        )
    locations = pd.concat(frames, ignore_index=True).drop_duplicates()
    for col in ["CCDD", "CCPP", "CCDI"]:
        locations[col] = locations[col].astype(str).str.zfill(2)
    for col in ["NOMBREDD", "NOMBREPV", "NOMBREDI"]:
        locations[col] = locations[col].astype(str).str.strip()
    return locations.sort_values(["NOMBREPV", "NOMBREDI"]).reset_index(drop=True)


def season_from_month(month: int) -> str:
    if month in [12, 1, 2]:
        return "Verano"
    if month in [3, 4, 5]:
        return "Otoño"
    if month in [6, 7, 8]:
        return "Invierno"
    return "Primavera"


def week_to_month(week: int) -> int:
    return int(np.clip(round(week / 4.345), 1, 12))


def default_price_context(price_df: pd.DataFrame, destino: str, semana: int) -> dict:
    subset = price_df[(price_df["destino"] == destino) & (price_df["semana_iso"].between(max(1, semana - 2), min(53, semana + 2)))]
    if subset.empty:
        subset = price_df[price_df["destino"] == destino]
    if subset.empty:
        subset = price_df
    return {
        "volumen_exportado": float(subset["volumen_exportado"].median()),
        "operaciones": float(subset["operaciones"].median()),
        "precio_lag_1": float(subset["precio_fob_por_kilogramo"].median()),
        "precio_prom_movil_4": float(subset["precio_fob_por_kilogramo"].median()),
        "volatilidad_movil_4": float(subset["precio_fob_por_kilogramo"].std() or price_df["precio_fob_por_kilogramo"].std()),
    }


def make_cost_row(values: dict) -> pd.DataFrame:
    produccion_kg = values["area_cosechada_ha"] * values["rendimiento_kg_ha"]
    venta_kg = produccion_kg * values["porcentaje_vendido"] / 100
    row = {
        "produccion_kg": produccion_kg,
        "venta_kg": venta_kg,
        "area_cosechada_ha": values["area_cosechada_ha"],
        "plantas_cosechadas": values["plantas_cosechadas"],
        "rendimiento_kg_ha": values["rendimiento_kg_ha"],
        "densidad_plantas_ha": values["plantas_cosechadas"] / max(values["area_cosechada_ha"], 0.001),
        "porcentaje_vendido": values["porcentaje_vendido"] / 100,
        "duracion_cosecha_meses": values["duracion_cosecha_meses"],
        "precio_chacra_kg": values["precio_chacra_kg"],
        "venta_valor_soles": values["precio_chacra_kg"] * venta_kg,
        "ANIO": str(values["anio"]),
        "CCPP": str(values["provincia"]),
        "CCDI": str(values["distrito"]),
        "P205_TOT": str(values["numero_cosechas"]),
        "P208": values["tipo_conduccion"],
        "P212": values["fuente_agua"],
        "P213": values["sistema_riego"],
        "P211_1": "1" if values["considera_clima"] else "0",
        "P211_2": "1" if values["considera_agua"] else "0",
        "P211_4": "1" if values["considera_suelo"] else "0",
        "P211_8": "1" if values["considera_fertilidad"] else "0",
        "P221_1": "1" if values["venta_chacra"] else "0",
        "P221_2": "1" if values["venta_fuera"] else "0",
        "P222_1": "1" if values["comprador"] == "Acopiador" else "0",
        "P222_2": "1" if values["comprador"] == "Mayorista" else "0",
        "P222_3": "1" if values["comprador"] == "Minorista" else "0",
        "P222_5": "1" if values["comprador"] == "Agroindustria" else "0",
        "P223_1": "1" if values["mercado"] == "Local" else "0",
        "P223_2": "1" if values["mercado"] == "Regional" else "0",
        "P223_3": "1" if values["mercado"] == "Exterior" else "0",
        "P223A": "1" if values["produccion_afectada"] else "2",
        "P223B_1": "1" if values["sequia"] else "0",
        "P223B_6": "1" if values["lluvia_destiempo"] else "0",
        "P223B_7": "1" if values["plagas"] else "0",
    }
    return pd.DataFrame([row])


def make_price_row(price_df: pd.DataFrame, values: dict) -> pd.DataFrame:
    month = week_to_month(values["semana"])
    context = default_price_context(price_df, values["destino"], values["semana"])
    volumen = values["volumen_exportado_ref"] or context["volumen_exportado"]
    operaciones = values["operaciones_ref"] or context["operaciones"]
    row = {
        "anio": values["anio"],
        "mes": month,
        "semana_iso": values["semana"],
        "trimestre": int(np.ceil(month / 3)),
        "log_volumen_exportado": np.log1p(volumen),
        "log_operaciones": np.log1p(operaciones),
        "precio_lag_1": context["precio_lag_1"],
        "precio_prom_movil_4": context["precio_prom_movil_4"],
        "volatilidad_movil_4": context["volatilidad_movil_4"],
        "temporada": season_from_month(month),
        "destino": values["destino"],
    }
    return pd.DataFrame([row])


def calc_profit(produccion_kg, exportable_pct, merma_pct, price_usd, cost_pen, fx, extra_cost_pen):
    volumen_vendido = produccion_kg * exportable_pct / 100 * (1 - merma_pct / 100)
    ingreso_pen = volumen_vendido * price_usd * fx
    costo_kg = cost_pen + extra_cost_pen
    costo_total = produccion_kg * costo_kg
    margen = ingreso_pen - costo_total
    margen_kg = margen / max(volumen_vendido, 1)
    return {
        "volumen_vendido": volumen_vendido,
        "ingreso_pen": ingreso_pen,
        "costo_total": costo_total,
        "costo_kg": costo_kg,
        "margen": margen,
        "margen_kg": margen_kg,
        "margen_venta": margen / ingreso_pen if ingreso_pen else 0,
        "rentabilidad_costo": margen / costo_total if costo_total else 0,
        "precio_equilibrio_usd": costo_kg / fx if fx else 0,
    }


def shared_inputs(cost_df: pd.DataFrame, location_df: pd.DataFrame, key_prefix: str):
    defaults = {
        "area": float(cost_df["area_cosechada_ha"].median()),
        "rend": float(cost_df["rendimiento_kg_ha"].median()),
        "plantas": int(cost_df["plantas_cosechadas"].median()),
        "precio_chacra": float(cost_df["precio_chacra_kg"].median()),
    }
    cols = st.columns(4)
    anio = cols[0].selectbox("Año campaña", [2026, 2025, 2024, 2023, 2022, 2021], key=f"{key_prefix}_anio")
    region_name = cols[1].selectbox("Región", sorted(location_df["NOMBREDD"].unique().tolist()), key=f"{key_prefix}_region")
    region_locations = location_df[location_df["NOMBREDD"] == region_name]
    provincia_name = cols[2].selectbox("Provincia", sorted(region_locations["NOMBREPV"].unique().tolist()), key=f"{key_prefix}_provincia")
    province_locations = region_locations[region_locations["NOMBREPV"] == provincia_name]
    district_options = province_locations.assign(label=province_locations["NOMBREDI"] + " (" + province_locations["CCDI"] + ")")
    district_label = cols[3].selectbox("Distrito", district_options["label"].tolist(), key=f"{key_prefix}_distrito")
    selected_location = district_options[district_options["label"] == district_label].iloc[0]

    numero_cosechas = st.number_input("N° cosechas", 1, 4, 1, key=f"{key_prefix}_numero_cosechas")

    cols = st.columns(4)
    area = cols[0].number_input("Área cosechada (ha)", min_value=0.001, value=max(defaults["area"], 0.1), step=0.1, key=f"{key_prefix}_area")
    rendimiento = cols[1].number_input("Rendimiento (kg/ha)", min_value=100.0, value=max(defaults["rend"], 1000.0), step=100.0, key=f"{key_prefix}_rendimiento")
    plantas = cols[2].number_input("Plantas cosechadas", min_value=1, value=max(defaults["plantas"], 100), step=10, key=f"{key_prefix}_plantas")
    duracion = cols[3].number_input("Duración cosecha (meses)", min_value=1, max_value=24, value=4, key=f"{key_prefix}_duracion")

    cols = st.columns(3)
    tipo_conduccion = cols[0].selectbox("Tipo conducción", list(P208_LABELS), format_func=lambda x: P208_LABELS[x], index=0, key=f"{key_prefix}_tipo_conduccion")
    fuente_agua = cols[1].selectbox("Fuente de agua", list(P212_LABELS), format_func=lambda x: P212_LABELS[x], index=1, key=f"{key_prefix}_fuente_agua")
    sistema_riego = cols[2].selectbox("Sistema de riego", list(P213_LABELS), format_func=lambda x: P213_LABELS[x], index=6, key=f"{key_prefix}_sistema_riego")

    cols = st.columns(6)
    considera_clima = cols[0].checkbox("Considera clima", True, key=f"{key_prefix}_considera_clima")
    considera_agua = cols[1].checkbox("Considera agua", True, key=f"{key_prefix}_considera_agua")
    considera_suelo = cols[2].checkbox("Considera suelo", True, key=f"{key_prefix}_considera_suelo")
    considera_fertilidad = cols[3].checkbox("Considera fertilidad", True, key=f"{key_prefix}_considera_fertilidad")
    venta_chacra = cols[4].checkbox("Venta en chacra", True, key=f"{key_prefix}_venta_chacra")
    venta_fuera = cols[5].checkbox("Venta fuera", False, key=f"{key_prefix}_venta_fuera")

    cols = st.columns(4)
    comprador = cols[0].selectbox("Comprador", ["Acopiador", "Mayorista", "Minorista", "Agroindustria", "Otro"], key=f"{key_prefix}_comprador")
    mercado = cols[1].selectbox("Mercado", ["Local", "Regional", "Exterior", "Agroindustria"], key=f"{key_prefix}_mercado")
    porcentaje_vendido = cols[2].slider("% vendido", 0, 100, 85, key=f"{key_prefix}_porcentaje_vendido")
    precio_chacra = cols[3].number_input("Precio chacra (S//kg)", min_value=0.0, value=max(defaults["precio_chacra"], 0.5), step=0.1, key=f"{key_prefix}_precio_chacra")

    cols = st.columns(4)
    produccion_afectada = cols[0].checkbox("Producción afectada", False, key=f"{key_prefix}_produccion_afectada")
    sequia = cols[1].checkbox("Sequía", False, key=f"{key_prefix}_sequia")
    lluvia_destiempo = cols[2].checkbox("Lluvias a destiempo", False, key=f"{key_prefix}_lluvia_destiempo")
    plagas = cols[3].checkbox("Plagas/enfermedades", False, key=f"{key_prefix}_plagas")

    return {
        "anio": anio,
        "region_nombre": region_name,
        "provincia_nombre": provincia_name,
        "distrito_nombre": selected_location["NOMBREDI"],
        "provincia": str(selected_location["CCPP"]),
        "distrito": str(selected_location["CCDI"]),
        "numero_cosechas": numero_cosechas,
        "area_cosechada_ha": area, "rendimiento_kg_ha": rendimiento, "plantas_cosechadas": plantas,
        "duracion_cosecha_meses": duracion, "tipo_conduccion": tipo_conduccion, "fuente_agua": fuente_agua,
        "sistema_riego": sistema_riego, "considera_clima": considera_clima, "considera_agua": considera_agua,
        "considera_suelo": considera_suelo, "considera_fertilidad": considera_fertilidad,
        "venta_chacra": venta_chacra, "venta_fuera": venta_fuera,
        "comprador": comprador, "mercado": mercado, "porcentaje_vendido": porcentaje_vendido,
        "precio_chacra_kg": precio_chacra, "produccion_afectada": produccion_afectada, "sequia": sequia,
        "lluvia_destiempo": lluvia_destiempo, "plagas": plagas,
    }


def main():
    st.set_page_config(page_title="Modelos Palta La Libertad", layout="wide")
    cost_model, price_model, risk_model, metadata, metrics = load_artifacts()
    cost_df, price_df, location_df = load_data()

    st.title("Modelos de rentabilidad y escenarios de campaña - Palta")
    st.caption(f"Datos de costos 2021-2024 y precios FOB hasta {metadata['latest_price_date']}.")

    with st.expander("Métricas de entrenamiento", expanded=False):
        st.json(metrics)

    tab1, tab2, tab3 = st.tabs(["S-1 Rentabilidad", "S-2 Escenarios", "Datos"])

    with tab1:
        st.subheader("Cálculo de rentabilidad")
        values = shared_inputs(cost_df, location_df, "s1")
        cols = st.columns(5)
        destino = cols[0].selectbox("Destino FOB", metadata["destinos"], index=metadata["destinos"].index("NETHERLANDS") if "NETHERLANDS" in metadata["destinos"] else 0)
        semana = cols[1].slider("Semana de venta", 1, 53, 22)
        exportable_pct = cols[2].slider("% exportable", 0, 100, 80)
        merma_pct = cols[3].slider("% merma/descarte", 0, 60, 8)
        fx = cols[4].number_input("Tipo de cambio S//USD", min_value=0.1, value=3.75, step=0.05)

        cols = st.columns(4)
        packing = cols[0].number_input("Packing S//kg", min_value=0.0, value=0.35, step=0.05)
        logistica = cols[1].number_input("Logística S//kg", min_value=0.0, value=0.25, step=0.05)
        comision = cols[2].number_input("Comisión S//kg", min_value=0.0, value=0.05, step=0.01)
        otros = cols[3].number_input("Otros S//kg", min_value=0.0, value=0.05, step=0.01)

        cost_row = make_cost_row(values)
        price_row = make_price_row(price_df, {"anio": values["anio"], "destino": destino, "semana": semana, "volumen_exportado_ref": 0, "operaciones_ref": 0})
        cost_pred = float(np.expm1(cost_model.predict(cost_row))[0])
        price_pred = float(price_model.predict(price_row)[0])
        risk_prob = float(risk_model.predict_proba(cost_row[metadata["risk_features_num"] + metadata["risk_features_cat"]])[:, 1][0]) if risk_model else np.nan

        produccion_kg = values["area_cosechada_ha"] * values["rendimiento_kg_ha"]
        extra_cost = packing + logistica + comision + otros
        result = calc_profit(produccion_kg, exportable_pct, merma_pct, price_pred, cost_pred, fx, extra_cost)

        cols = st.columns(5)
        cols[0].metric("Producción kg", f"{produccion_kg:,.0f}")
        cols[1].metric("Costo productivo estimado", f"S/ {cost_pred:,.2f}/kg")
        cols[2].metric("Precio FOB estimado", f"USD {price_pred:,.2f}/kg")
        cols[3].metric("Margen total", f"S/ {result['margen']:,.0f}")
        cols[4].metric("Riesgo afectación", f"{risk_prob:.1%}" if not np.isnan(risk_prob) else "N/D")

        detail = pd.DataFrame(
            {
                "Indicador": ["Volumen vendido", "Ingreso", "Costo total", "Costo total kg", "Margen kg", "Margen sobre venta", "Rentabilidad sobre costo", "Precio equilibrio FOB"],
                "Valor": [
                    f"{result['volumen_vendido']:,.0f} kg",
                    f"S/ {result['ingreso_pen']:,.0f}",
                    f"S/ {result['costo_total']:,.0f}",
                    f"S/ {result['costo_kg']:,.2f}",
                    f"S/ {result['margen_kg']:,.2f}",
                    f"{result['margen_venta']:.1%}",
                    f"{result['rentabilidad_costo']:.1%}",
                    f"USD {result['precio_equilibrio_usd']:,.2f}/kg",
                ],
            }
        )
        st.dataframe(detail, hide_index=True, use_container_width=True)

    with tab2:
        st.subheader("Simulador de escenarios")
        base_values = shared_inputs(cost_df, location_df, "s2")
        cols = st.columns(5)
        destino = cols[0].selectbox("Destino escenario", metadata["destinos"], index=metadata["destinos"].index("NETHERLANDS") if "NETHERLANDS" in metadata["destinos"] else 0, key="dest_s2")
        semana = cols[1].slider("Semana base", 1, 53, 22, key="week_s2")
        exportable_pct = cols[2].slider("% exportable base", 0, 100, 80, key="export_s2")
        fx = cols[3].number_input("TC base S//USD", min_value=0.1, value=3.75, step=0.05, key="fx_s2")
        extra_cost = cols[4].number_input("Costos extra S//kg", min_value=0.0, value=0.70, step=0.05, key="extra_s2")

        cost_row = make_cost_row(base_values)
        price_row = make_price_row(price_df, {"anio": base_values["anio"], "destino": destino, "semana": semana, "volumen_exportado_ref": 0, "operaciones_ref": 0})
        base_cost = float(np.expm1(cost_model.predict(cost_row))[0])
        base_price = float(price_model.predict(price_row)[0])
        base_production = base_values["area_cosechada_ha"] * base_values["rendimiento_kg_ha"]

        scenario_defs = pd.DataFrame(
            [
                {"Escenario": "Conservador", "Precio": -0.12, "Costo": 0.10, "Volumen": -0.10, "Merma": 15},
                {"Escenario": "Base", "Precio": 0.00, "Costo": 0.00, "Volumen": 0.00, "Merma": 8},
                {"Escenario": "Optimista", "Precio": 0.10, "Costo": -0.05, "Volumen": 0.08, "Merma": 5},
                {"Escenario": "Estrés", "Precio": -0.22, "Costo": 0.20, "Volumen": -0.20, "Merma": 22},
            ]
        )
        rows = []
        for _, scenario in scenario_defs.iterrows():
            production = base_production * (1 + scenario["Volumen"])
            price = base_price * (1 + scenario["Precio"])
            cost = base_cost * (1 + scenario["Costo"])
            calc = calc_profit(production, exportable_pct, scenario["Merma"], price, cost, fx, extra_cost)
            rows.append(
                {
                    "Escenario": scenario["Escenario"],
                    "Precio USD/kg": price,
                    "Costo S//kg": cost + extra_cost,
                    "Producción kg": production,
                    "Merma": scenario["Merma"],
                    "Ingreso S/": calc["ingreso_pen"],
                    "Costo S/": calc["costo_total"],
                    "Margen S/": calc["margen"],
                    "Rentabilidad": calc["rentabilidad_costo"],
                }
            )
        scenario_df = pd.DataFrame(rows)
        st.dataframe(
            scenario_df.style.format(
                {
                    "Precio USD/kg": "{:,.2f}",
                    "Costo S//kg": "{:,.2f}",
                    "Producción kg": "{:,.0f}",
                    "Merma": "{:,.0f}%",
                    "Ingreso S/": "S/ {:,.0f}",
                    "Costo S/": "S/ {:,.0f}",
                    "Margen S/": "S/ {:,.0f}",
                    "Rentabilidad": "{:.1%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        chart = scenario_df.set_index("Escenario")[["Ingreso S/", "Costo S/", "Margen S/"]]
        st.bar_chart(chart)

    with tab3:
        st.subheader("Datos generados")
        cols = st.columns(2)
        with cols[0]:
            st.write("Base enriquecida de costos")
            st.dataframe(cost_df.head(50), use_container_width=True)
        with cols[1]:
            st.write("Features de precio FOB")
            st.dataframe(price_df.head(50), use_container_width=True)


if __name__ == "__main__":
    main()
