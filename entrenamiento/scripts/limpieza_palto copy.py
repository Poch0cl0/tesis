import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
BASE_DIR = r'E:\Proyectos\Universidad\Analitica\Extract data\costos_produccion'
YEARS = ['2021', '2022', '2023', '2024']

CODIGO_PALTO = '1162'
CODIGO_LA_LIBERTAD = '13'
DEPARTAMENTO_NOMBRE = 'LA LIBERTAD'

OUTPUT_NAME = 'palto_costo_por_kg_la_libertad.csv'
CONSOLIDATED_PATH = r'E:\Proyectos\Universidad\Analitica\Extract data\palto_costo_por_kg_la_libertad_consolidado.csv'

ENCODING = 'latin1'


def _limpiar_columnas(df):
    """Elimina BOM y espacios de los nombres de columnas."""
    # Algunos CSVs se leyeron como latin1 pero tienen BOM UTF-8 (EF BB BF)
    cleaned = []
    for col in df.columns:
        # Convertir a bytes vía latin1, quitar BOM si existe, volver a string
        b = col.encode('latin1').replace(b'\xef\xbb\xbf', b'').decode('latin1')
        cleaned.append(b.strip())
    df.columns = cleaned
    return df


def _detectar_factor_col(df):
    """Detecta si la columna factor se llama FACTOR_PRODUCTOR o FACTOR y la renombra."""
    cols = [c for c in df.columns if 'FACTOR' in c.upper()]
    if not cols:
        raise KeyError("No se encontró columna FACTOR en el dataframe.")
    factor_col = cols[0]
    if factor_col != 'FACTOR_PRODUCTOR':
        df = df.rename(columns={factor_col: 'FACTOR_PRODUCTOR'})
    return df


def _detectar_key_cols(df):
    """Detecta las columnas de identificación del productor según el año."""
    posibles_2024 = ['ANIO', 'CCDD', 'CCPP', 'CCDI', 'NSEGM', 'ID_PROD', 'UA']
    posibles_2021 = ['ANIO', 'CCDD', 'CCPP', 'CCDI', 'CONGLOMERADO', 'NSELUA', 'UA']

    cols_presentes = [c for c in posibles_2024 if c in df.columns]
    if len(cols_presentes) >= 7:  # Todas las de 2024/2023
        return cols_presentes

    cols_presentes = [c for c in posibles_2021 if c in df.columns]
    if len(cols_presentes) >= 6:  # Las de 2021/2022
        return cols_presentes

    raise KeyError(f"No se pudieron detectar columnas clave. Columnas disponibles: {list(df.columns)}")


def procesar_ano(year):
    """Procesa un año: carga, limpia, filtra PALTO + La Libertad, calcula costo/kg."""

    year_path = os.path.join(BASE_DIR, year)
    costos_path = os.path.join(year_path, 'Costos_Produccion.csv')
    prod_path = os.path.join(year_path, 'Produccion.csv')
    output_path = os.path.join(year_path, OUTPUT_NAME)

    print(f"\n{'='*60}")
    print(f"[INFO] Procesando ao {year}")
    print(f"{'='*60}")

    # Verificar existencia de archivos
    if not os.path.exists(costos_path):
        print(f"[ERROR] No existe: {costos_path}")
        return None
    if not os.path.exists(prod_path):
        print(f"[ERROR] No existe: {prod_path}")
        return None

    # -----------------------------------------------------------------------
    # 1. Cargar datos
    # -----------------------------------------------------------------------
    print(f"[INFO] Cargando Costos_Produccion.csv ...")
    df_costos = pd.read_csv(costos_path, dtype=str, low_memory=False, encoding=ENCODING)
    df_costos = _limpiar_columnas(df_costos)

    print(f"[INFO] Cargando Produccion.csv ...")
    df_prod = pd.read_csv(prod_path, dtype=str, low_memory=False, encoding=ENCODING)
    df_prod = _limpiar_columnas(df_prod)

    # Detectar y renombrar columna factor
    df_costos = _detectar_factor_col(df_costos)
    df_prod = _detectar_factor_col(df_prod)

    # Detectar columnas clave del productor
    key_cols = _detectar_key_cols(df_costos)
    print(f"       Columnas clave detectadas: {key_cols}")

    # -----------------------------------------------------------------------
    # 2. Filtrar PALTO y La Libertad
    # -----------------------------------------------------------------------
    df_costos = df_costos[
        (df_costos['P234_COD'] == CODIGO_PALTO) &
        (df_costos['CCDD'] == CODIGO_LA_LIBERTAD)
    ].copy()

    df_prod = df_prod[
        (df_prod['P204_COD'] == CODIGO_PALTO) &
        (df_prod['CCDD'] == CODIGO_LA_LIBERTAD)
    ].copy()

    print(f"       Registros en costos (PALTO + La Libertad): {len(df_costos):,}")
    print(f"       Registros en prod   (PALTO + La Libertad): {len(df_prod):,}")

    if len(df_costos) == 0 or len(df_prod) == 0:
        print(f"[WARN] Sin datos suficientes para el ao {year}. Saltando...")
        return None

    # -----------------------------------------------------------------------
    # 3. Limpiar y convertir columnas numéricas en COSTOS
    # -----------------------------------------------------------------------
    cols_costo = ['P235_VAL', 'P237_VAL', 'P239', 'P241']
    for col in cols_costo:
        df_costos[col] = pd.to_numeric(
            df_costos[col].replace('', np.nan).replace(' ', np.nan),
            errors='coerce'
        ).fillna(0)

    df_costos['FACTOR_PRODUCTOR'] = pd.to_numeric(
        df_costos['FACTOR_PRODUCTOR'].replace('', np.nan).replace(' ', np.nan),
        errors='coerce'
    )

    df_costos['costo_total'] = df_costos[cols_costo].sum(axis=1)

    # -----------------------------------------------------------------------
    # 4. Limpiar y calcular producción en kg en PRODUCCION
    # -----------------------------------------------------------------------
    df_prod['P219_CANT_1'] = pd.to_numeric(
        df_prod['P219_CANT_1'].replace('', np.nan).replace(' ', np.nan),
        errors='coerce'
    ).fillna(0)

    df_prod['P219_CANT_2'] = pd.to_numeric(
        df_prod['P219_CANT_2'].replace('', np.nan).replace(' ', np.nan),
        errors='coerce'
    ).fillna(0)

    df_prod['P219_EQUIV_KG'] = pd.to_numeric(
        df_prod['P219_EQUIV_KG'].replace('', np.nan).replace(' ', np.nan),
        errors='coerce'
    ).fillna(0)

    df_prod['produccion_kg'] = (df_prod['P219_CANT_1'] + df_prod['P219_CANT_2'] / 1000) * df_prod['P219_EQUIV_KG']

    df_prod['FACTOR_PRODUCTOR'] = pd.to_numeric(
        df_prod['FACTOR_PRODUCTOR'].replace('', np.nan).replace(' ', np.nan),
        errors='coerce'
    )

    # -----------------------------------------------------------------------
    # 5. Agregar producción por clave productor (sumar si hay varias parcelas)
    # -----------------------------------------------------------------------
    agg_prod = df_prod.groupby(key_cols).agg(
        produccion_kg=('produccion_kg', 'sum'),
        FACTOR_PRODUCTOR=('FACTOR_PRODUCTOR', 'first')
    ).reset_index()

    # -----------------------------------------------------------------------
    # 6. Preparar costos (sumar si hay duplicados)
    # -----------------------------------------------------------------------
    if df_costos.duplicated(subset=key_cols).any():
        print("[WARN] Duplicados en costos por clave productor. Se suman insumos.")
        df_costos = df_costos.groupby(key_cols).agg(
            costo_total=('costo_total', 'sum'),
            FACTOR_PRODUCTOR=('FACTOR_PRODUCTOR', 'first')
        ).reset_index()
    else:
        df_costos = df_costos[key_cols + ['costo_total', 'FACTOR_PRODUCTOR']].copy()

    # -----------------------------------------------------------------------
    # 7. Merge entre costos y producción
    # -----------------------------------------------------------------------
    print(f"[INFO] Emparejando costos con producción ...")
    df_merged = pd.merge(
        df_costos, agg_prod,
        on=key_cols,
        how='inner',
        suffixes=('_costos', '_prod')
    )

    print(f"       Productores emparejados: {len(df_merged):,}")

    # Unificar factor de expansión
    df_merged['FACTOR_PRODUCTOR'] = df_merged['FACTOR_PRODUCTOR_costos'].fillna(
        df_merged['FACTOR_PRODUCTOR_prod']
    )
    df_merged = df_merged.drop(columns=['FACTOR_PRODUCTOR_costos', 'FACTOR_PRODUCTOR_prod'])

    # -----------------------------------------------------------------------
    # 8. Filtrar solo registros con costo > 0 y producción > 0
    # -----------------------------------------------------------------------
    df_valid = df_merged[(df_merged['costo_total'] > 0) & (df_merged['produccion_kg'] > 0)].copy()
    print(f"       Registros válidos (costo>0 y kg>0): {len(df_valid):,}")

    if len(df_valid) == 0:
        print(f"[WARN] Sin registros válidos para el ao {year}. Saltando...")
        return None

    # -----------------------------------------------------------------------
    # 9. Calcular costo por kg
    # -----------------------------------------------------------------------
    df_valid['costo_por_kg'] = df_valid['costo_total'] / df_valid['produccion_kg']

    # -----------------------------------------------------------------------
    # 10. Marcar outliers con el método IQR
    # -----------------------------------------------------------------------
    Q1 = df_valid['costo_por_kg'].quantile(0.25)
    Q3 = df_valid['costo_por_kg'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(0, Q1 - 1.5 * IQR)
    upper_bound = Q3 + 1.5 * IQR

    df_valid['es_outlier'] = (
        (df_valid['costo_por_kg'] < lower_bound) | (df_valid['costo_por_kg'] > upper_bound)
    ).astype(int)

    outliers_count = df_valid['es_outlier'].sum()
    print(f"[INFO] Outliers detectados (IQR): {outliers_count:,} ({outliers_count/len(df_valid)*100:.1f}%)")
    print(f"       Límites IQR: [{lower_bound:.4f}, {upper_bound:.4f}]")

    # Estadísticas básicas
    print(f"\n[INFO] Estadísticas costo_por_kg (S/ por kg) - Ao {year}:")
    print(df_valid['costo_por_kg'].describe(percentiles=[.05, .25, .5, .75, .95]).to_string())

    mask = df_valid['FACTOR_PRODUCTOR'].notna() & (df_valid['FACTOR_PRODUCTOR'] > 0)
    if mask.sum() > 0:
        weighted_avg = np.average(
            df_valid.loc[mask, 'costo_por_kg'],
            weights=df_valid.loc[mask, 'FACTOR_PRODUCTOR']
        )
        print(f"       Costo/kg ponderado: {weighted_avg:.4f}")

    # -----------------------------------------------------------------------
    # 11. Exportar CSV del ao
    # -----------------------------------------------------------------------
    output_cols = key_cols + ['costo_total', 'produccion_kg', 'costo_por_kg', 'FACTOR_PRODUCTOR', 'es_outlier']
    df_valid[output_cols].to_csv(output_path, index=False)
    print(f"[OK] CSV del ao {year} guardado en: {output_path}")
    print(f"     Filas exportadas: {len(df_valid):,}")

    return df_valid[output_cols]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    dataframes = []

    for year in YEARS:
        df_year = procesar_ano(year)
        if df_year is not None:
            dataframes.append(df_year)

    # -----------------------------------------------------------------------
    # Consolidado de todos los aos
    # -----------------------------------------------------------------------
    if dataframes:
        print(f"\n{'='*60}")
        print(f"[INFO] Generando CSV consolidado de todos los aos ...")
        print(f"{'='*60}")

        df_consolidado = pd.concat(dataframes, ignore_index=True)

        # Asegurar que ANIO sea numérico/entero para orden
        df_consolidado['ANIO'] = pd.to_numeric(df_consolidado['ANIO'], errors='coerce')

        df_consolidado.to_csv(CONSOLIDATED_PATH, index=False)

        print(f"[OK] Consolidado guardado en: {CONSOLIDATED_PATH}")
        print(f"     Total de productores en consolidado: {len(df_consolidado):,}")
        print(f"\n[INFO] Estadísticas consolidadas (todos los aos):")
        print(df_consolidado['costo_por_kg'].describe(percentiles=[.05, .25, .5, .75, .95]).to_string())

        # Distribución por ao
        print(f"\n[INFO] Distribución por ao:")
        print(df_consolidado.groupby('ANIO').agg(
            n=('costo_por_kg', 'count'),
            mediana=('costo_por_kg', 'median'),
            media=('costo_por_kg', 'mean')
        ).to_string())
    else:
        print("[ERROR] No se procesó ningún ao. No se generó consolidado.")
