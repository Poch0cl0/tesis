import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
BASE_DIR = r'E:\Proyectos\Universidad\Analitica\Extract data - copia\costos_produccion'
YEARS = ['2021', '2022', '2023', '2024']

CODIGO_PALTO = '1162'
CODIGO_LA_LIBERTAD = '13'
DEPARTAMENTO_NOMBRE = 'LA LIBERTAD'

OUTPUT_NAME = 'palto_costo_por_kg_la_libertad.csv'
CONSOLIDATED_PATH = r'E:\Proyectos\Universidad\Analitica\Extract data - copia\palto_costo_por_kg_la_libertad_consolidado.csv'

ENCODING = 'latin1'

# Metadatos de variables para las tablas de calidad (orden fijo entre años)
VARS_NULOS = {
    'P235_VAL (Costos_Produccion)':         'Sí — fillna(0)',
    'P237_VAL (Costos_Produccion)':         'Sí — fillna(0)',
    'P239 (Costos_Produccion)':             'Sí — fillna(0)',
    'P241 (Costos_Produccion)':             'Sí — fillna(0)',
    'P219_CANT_1 (Produccion)':             'Sí — fillna(0)',
    'P219_CANT_2 (Produccion)':             'Sí — fillna(0)',
    'P219_EQUIV_KG (Produccion)':           'Sí — fillna(0)',
    'FACTOR_PRODUCTOR (Costos_Produccion)': 'No — mantiene NaN',
    'FACTOR_PRODUCTOR (Produccion)':        'No — mantiene NaN',
}

VARS_DOMINIO = {
    'P235_VAL (Costos_Produccion)':         'Numérico >= 0',
    'P237_VAL (Costos_Produccion)':         'Numérico >= 0',
    'P239 (Costos_Produccion)':             'Numérico >= 0',
    'P241 (Costos_Produccion)':             'Numérico >= 0',
    'P219_CANT_1 (Produccion)':             'Numérico >= 0',
    'P219_CANT_2 (Produccion)':             'Numérico >= 0',
    'P219_EQUIV_KG (Produccion)':           'Numérico >= 0',
    'FACTOR_PRODUCTOR (Costos_Produccion)': 'Numérico > 0',
    'FACTOR_PRODUCTOR (Produccion)':        'Numérico > 0',
}


# ---------------------------------------------------------------------------
# Helpers de calidad
# ---------------------------------------------------------------------------

def _print_tabla_calidad(titulo, headers, filas):
    """Imprime una tabla de calidad de datos en consola."""
    col_widths = [len(h) for h in headers]
    for fila in filas:
        for i, val in enumerate(fila):
            col_widths[i] = max(col_widths[i], len(str(val)))

    sep = '  +' + '+'.join('-' * (w + 2) for w in col_widths) + '+'
    header_line = '  |' + '|'.join(f' {h.ljust(col_widths[i])} ' for i, h in enumerate(headers)) + '|'

    print(f"\n  {titulo}")
    print(sep)
    print(header_line)
    print(sep)
    for fila in filas:
        print('  |' + '|'.join(f' {str(v).ljust(col_widths[i])} ' for i, v in enumerate(fila)) + '|')
    print(sep)


def _contar_nulos(serie):
    """Cuenta valores nulos, vacíos o espacios en una Serie de strings."""
    return int((serie.isna() | (serie == '') | (serie == ' ')).sum())


def _contar_invalidos_dominio(serie):
    """Cuenta valores no nulos/vacíos que no son convertibles a numérico."""
    serie_limpia = serie.replace('', np.nan).replace(' ', np.nan)
    nulos_antes = int(serie_limpia.isna().sum())
    nulos_despues = int(pd.to_numeric(serie_limpia, errors='coerce').isna().sum())
    return nulos_despues - nulos_antes


def _imprimir_calidad_consolidada(metrics_list):
    """Imprime las 4 tablas de calidad con una columna por año y columna Total."""
    years = [m['year'] for m in metrics_list]
    yh = years + ['TOTAL']

    print(f"\n{'='*60}")
    print(f"[CALIDAD] Reporte consolidado — todos los años")
    print(f"{'='*60}")

    # Integridad de Entidad — filas agrupadas por patrón de key_cols
    patterns = {}
    for i, m in enumerate(metrics_list):
        k = tuple(m['key_cols'])
        if k not in patterns:
            patterns[k] = []
        patterns[k].append(i)

    filas_entidad = []
    for pattern, indices in patterns.items():
        label = ' + '.join(pattern)
        row_c = [f'{label} (Costos_Produccion)', 'Unicidad por productor']
        row_p = [f'{label} (Produccion)',         'Unicidad por productor']
        total_c = total_p = 0
        for i in range(len(metrics_list)):
            if i in indices:
                row_c.append(metrics_list[i]['dup_costos'])
                row_p.append(metrics_list[i]['dup_prod'])
                total_c += metrics_list[i]['dup_costos']
                total_p += metrics_list[i]['dup_prod']
            else:
                row_c.append('-')
                row_p.append('-')
        row_c.append(total_c)
        row_p.append(total_p)
        filas_entidad.append(tuple(row_c))
        filas_entidad.append(tuple(row_p))

    _print_tabla_calidad(
        "CALIDAD DE DATOS — Integridad de Entidad [Consolidado]",
        ['Variable', 'Integridad entidad'] + yh,
        filas_entidad
    )

    # Evaluación de Nulos
    filas_nulos = []
    for var, valida in VARS_NULOS.items():
        counts = [m['nulos'].get(var, 0) for m in metrics_list]
        filas_nulos.append((var, valida, *counts, sum(counts)))
    _print_tabla_calidad(
        "CALIDAD DE DATOS — Evaluación de Nulos [Consolidado]",
        ['Variable', 'Valida Nulos?'] + yh,
        filas_nulos
    )

    # Integridad de Dominio
    filas_dominio = []
    for var, integridad in VARS_DOMINIO.items():
        counts = [m['dominio'].get(var, 0) for m in metrics_list]
        filas_dominio.append((var, integridad, *counts, sum(counts)))
    _print_tabla_calidad(
        "CALIDAD DE DATOS — Integridad de Dominio [Consolidado]",
        ['Variable', 'Integridad'] + yh,
        filas_dominio
    )

    # Validaciones del Negocio
    _print_tabla_calidad(
        "CALIDAD DE DATOS — Validaciones del Negocio [Consolidado]",
        ['Variable', 'Rango de valores'] + yh,
        [
            ('costo_total',   '> 0',
             *[m['costo_cero'] for m in metrics_list],
             sum(m['costo_cero'] for m in metrics_list)),
            ('produccion_kg', '> 0',
             *[m['prod_cero'] for m in metrics_list],
             sum(m['prod_cero'] for m in metrics_list)),
            ('costo_por_kg',  'Fuera de IQR (por año)',
             *[m['outliers'] for m in metrics_list],
             sum(m['outliers'] for m in metrics_list)),
        ]
    )


def _imprimir_calidad_consolidado_df(df_consolidado):
    """Muestra las inconsistencias que QUEDAN en el dataset final consolidado."""
    total         = len(df_consolidado)
    factor_nulos  = int(df_consolidado['FACTOR_PRODUCTOR'].isna().sum())
    outliers      = int(df_consolidado['es_outlier'].sum())

    _print_tabla_calidad(
        "CALIDAD DEL CONSOLIDADO — Inconsistencias residuales en el CSV final",
        ['Variable', 'Condición', 'Registros inconsistentes', '% sobre total'],
        [
            ('FACTOR_PRODUCTOR', 'Es NaN (sin factor de expansión)',
             factor_nulos, f'{factor_nulos / total * 100:.1f}%'),
            ('costo_por_kg', 'Outlier IQR (es_outlier = 1)',
             outliers, f'{outliers / total * 100:.1f}%'),
        ]
    )


# ---------------------------------------------------------------------------
# Helpers de procesamiento
# ---------------------------------------------------------------------------

def _limpiar_columnas(df):
    """Elimina BOM y espacios de los nombres de columnas."""
    cleaned = []
    for col in df.columns:
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
    if len(cols_presentes) >= 7:
        return cols_presentes

    cols_presentes = [c for c in posibles_2021 if c in df.columns]
    if len(cols_presentes) >= 6:
        return cols_presentes

    raise KeyError(f"No se pudieron detectar columnas clave. Columnas disponibles: {list(df.columns)}")


def procesar_ano(year):
    """
    Procesa un año: carga, limpia, filtra PALTO + La Libertad, calcula costo/kg.
    Retorna (df_resultado, metrics_dict) o (None, None) si no hay datos.
    """
    year_path = os.path.join(BASE_DIR, year)
    costos_path = os.path.join(year_path, 'Costos_Produccion.csv')
    prod_path = os.path.join(year_path, 'Produccion.csv')
    output_path = os.path.join(year_path, OUTPUT_NAME)

    print(f"\n{'='*60}")
    print(f"[INFO] Procesando año {year}")
    print(f"{'='*60}")

    if not os.path.exists(costos_path):
        print(f"[ERROR] No existe: {costos_path}")
        return None, None
    if not os.path.exists(prod_path):
        print(f"[ERROR] No existe: {prod_path}")
        return None, None

    # -----------------------------------------------------------------------
    # 1. Cargar datos
    # -----------------------------------------------------------------------
    print(f"[INFO] Cargando Costos_Produccion.csv ...")
    df_costos = pd.read_csv(costos_path, dtype=str, low_memory=False, encoding=ENCODING)
    df_costos = _limpiar_columnas(df_costos)

    print(f"[INFO] Cargando Produccion.csv ...")
    df_prod = pd.read_csv(prod_path, dtype=str, low_memory=False, encoding=ENCODING)
    df_prod = _limpiar_columnas(df_prod)

    df_costos = _detectar_factor_col(df_costos)
    df_prod = _detectar_factor_col(df_prod)

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
        print(f"[WARN] Sin datos suficientes para el año {year}. Saltando...")
        return None, None

    # -----------------------------------------------------------------------
    # Métricas de calidad — capturadas ANTES de convertir tipos
    # -----------------------------------------------------------------------
    cols_costo = ['P235_VAL', 'P237_VAL', 'P239', 'P241']
    cols_prod  = ['P219_CANT_1', 'P219_CANT_2', 'P219_EQUIV_KG']

    metrics = {
        'year':      year,
        'key_cols':  key_cols,
        'dup_costos': int(df_costos.duplicated(subset=key_cols).sum()),
        'dup_prod':   int(df_prod.duplicated(subset=key_cols).sum()),
        'nulos':   {},
        'dominio': {},
    }

    for col in cols_costo:
        k = f'{col} (Costos_Produccion)'
        metrics['nulos'][k]   = _contar_nulos(df_costos[col])
        metrics['dominio'][k] = _contar_invalidos_dominio(df_costos[col])

    for col in cols_prod:
        k = f'{col} (Produccion)'
        metrics['nulos'][k]   = _contar_nulos(df_prod[col])
        metrics['dominio'][k] = _contar_invalidos_dominio(df_prod[col])

    metrics['nulos']['FACTOR_PRODUCTOR (Costos_Produccion)']   = _contar_nulos(df_costos['FACTOR_PRODUCTOR'])
    metrics['nulos']['FACTOR_PRODUCTOR (Produccion)']          = _contar_nulos(df_prod['FACTOR_PRODUCTOR'])
    metrics['dominio']['FACTOR_PRODUCTOR (Costos_Produccion)'] = _contar_invalidos_dominio(df_costos['FACTOR_PRODUCTOR'])
    metrics['dominio']['FACTOR_PRODUCTOR (Produccion)']        = _contar_invalidos_dominio(df_prod['FACTOR_PRODUCTOR'])

    # -----------------------------------------------------------------------
    # 3. Limpiar y convertir columnas numéricas en COSTOS
    # -----------------------------------------------------------------------
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

    df_merged['FACTOR_PRODUCTOR'] = df_merged['FACTOR_PRODUCTOR_costos'].fillna(
        df_merged['FACTOR_PRODUCTOR_prod']
    )
    df_merged = df_merged.drop(columns=['FACTOR_PRODUCTOR_costos', 'FACTOR_PRODUCTOR_prod'])

    # -----------------------------------------------------------------------
    # 8. Filtrar solo registros con costo > 0 y producción > 0
    # -----------------------------------------------------------------------
    metrics['costo_cero'] = int((df_merged['costo_total'] <= 0).sum())
    metrics['prod_cero']  = int((df_merged['produccion_kg'] <= 0).sum())

    df_valid = df_merged[(df_merged['costo_total'] > 0) & (df_merged['produccion_kg'] > 0)].copy()
    print(f"       Registros válidos (costo>0 y kg>0): {len(df_valid):,}")

    if len(df_valid) == 0:
        print(f"[WARN] Sin registros válidos para el año {year}. Saltando...")
        return None, None

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

    metrics['outliers'] = int(df_valid['es_outlier'].sum())

    print(f"[INFO] Outliers detectados (IQR): {metrics['outliers']:,} ({metrics['outliers']/len(df_valid)*100:.1f}%)")
    print(f"       Límites IQR: [{lower_bound:.4f}, {upper_bound:.4f}]")

    # Estadísticas básicas
    print(f"\n[INFO] Estadísticas costo_por_kg (S/ por kg) - Año {year}:")
    print(df_valid['costo_por_kg'].describe(percentiles=[.05, .25, .5, .75, .95]).to_string())

    mask = df_valid['FACTOR_PRODUCTOR'].notna() & (df_valid['FACTOR_PRODUCTOR'] > 0)
    if mask.sum() > 0:
        weighted_avg = np.average(
            df_valid.loc[mask, 'costo_por_kg'],
            weights=df_valid.loc[mask, 'FACTOR_PRODUCTOR']
        )
        print(f"       Costo/kg ponderado: {weighted_avg:.4f}")

    # -----------------------------------------------------------------------
    # 11. Exportar CSV del año
    # -----------------------------------------------------------------------
    output_cols = key_cols + ['costo_total', 'produccion_kg', 'costo_por_kg', 'FACTOR_PRODUCTOR', 'es_outlier']
    df_valid[output_cols].to_csv(output_path, index=False)
    print(f"[OK] CSV del año {year} guardado en: {output_path}")
    print(f"     Filas exportadas: {len(df_valid):,}")

    return df_valid[output_cols], metrics


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    dataframes  = []
    all_metrics = []

    for year in YEARS:
        df_year, metrics = procesar_ano(year)
        if df_year is not None:
            dataframes.append(df_year)
            all_metrics.append(metrics)

    if dataframes:
        print(f"\n{'='*60}")
        print(f"[INFO] Generando CSV consolidado de todos los años ...")
        print(f"{'='*60}")

        df_consolidado = pd.concat(dataframes, ignore_index=True)
        df_consolidado['ANIO'] = pd.to_numeric(df_consolidado['ANIO'], errors='coerce')
        df_consolidado.to_csv(CONSOLIDATED_PATH, index=False)

        print(f"[OK] Consolidado guardado en: {CONSOLIDATED_PATH}")
        print(f"     Total de productores en consolidado: {len(df_consolidado):,}")
        print(f"\n[INFO] Estadísticas consolidadas (todos los años):")
        print(df_consolidado['costo_por_kg'].describe(percentiles=[.05, .25, .5, .75, .95]).to_string())

        print(f"\n[INFO] Distribución por año:")
        print(df_consolidado.groupby('ANIO').agg(
            n=('costo_por_kg', 'count'),
            mediana=('costo_por_kg', 'median'),
            media=('costo_por_kg', 'mean')
        ).to_string())

        # Reporte de calidad al final — todos los años juntos
        _imprimir_calidad_consolidada(all_metrics)
        _imprimir_calidad_consolidado_df(df_consolidado)

    else:
        print("[ERROR] No se procesó ningún año. No se generó consolidado.")
