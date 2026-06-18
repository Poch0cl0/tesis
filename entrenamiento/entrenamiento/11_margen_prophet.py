from prophet_common import train_prophet_regression


if __name__ == "__main__":
    train_prophet_regression(
        name="11_margen_prophet",
        csv_name="dataset_prediccion_margen_minimo.csv",
        target="margen_exportador_soles_kg",
        numeric=["precio_fob_usd_kg", "rendimiento_kg_ha", "porcentaje_vendido"],
        categorical=["region", "provincia", "tipo_conduccion_cultivo"],
        time_split=False,
    )
