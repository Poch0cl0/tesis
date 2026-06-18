# Entrenamiento de modelos de palta

## 1. Objetivo general

Se construyeron 12 modelos de aprendizaje automático para tres soluciones:

1. Predecir el precio FOB de la palta con un horizonte de seis semanas.
2. Predecir el margen exportador en soles por kilogramo.
3. Predecir el riesgo de obtener un margen exportador bajo ante distintos escenarios.

Para cada solución se compararon cuatro algoritmos:

- Random Forest.
- HistGradientBoosting.
- Modelo lineal regularizado: ElasticNet para regresión y Logistic Regression para clasificación.
- Prophet de Meta.

Todos los modelos entrenados se guardan en `models/` como archivos `.joblib`. Las
métricas individuales se guardan como archivos JSON y el resumen completo está en
`models/training_summary.json`.

## 1.1 Relación con las tres soluciones del sistema

Los modelos se integran con las soluciones de la siguiente manera:

| Solución | Función | Modelos asociados |
|---|---|---|
| S-1 – Módulo de cálculo de rentabilidad | Estimar margen exportador | Modelos 04, 05, 06 y 11 |
| S-2 – Simulador de escenarios de campaña | Estimar riesgo de margen bajo ante cambios y eventualidades | Modelos 07, 08, 09 y 12 |
| S-3 – Modelo predictivo de precios FOB | Predecir el precio FOB a seis semanas | Modelos 01, 02, 03 y 10 |

Aunque se entrenaron por separado, las tres soluciones están conectadas. S-3 genera un
precio FOB futuro que puede utilizarse como entrada en S-1 y S-2.

```text
S-3: precio FOB previsto a seis semanas
                    |
                    v
S-1: margen exportador esperado
                    |
                    v
S-2: riesgo de margen bajo bajo un escenario de campaña
```

El usuario también puede reemplazar la predicción de S-3 por un precio FOB manual. En
ese caso, S-1 y S-2 reciben el precio ingresado sin volver a ejecutar el modelo FOB.

```text
Precio FOB predicho por S-3 ─┐
                             ├─> S-1 y S-2
Precio FOB ingresado manual ─┘
```

### S-1 – Módulo de cálculo de rentabilidad

S-1 utiliza los modelos:

- `04_margen_random_forest`.
- `05_margen_hist_gradient_boosting`.
- `06_margen_elasticnet`.
- `11_margen_prophet`.

Entradas del módulo:

```text
precio_fob_usd_kg
region
provincia
rendimiento_kg_ha
porcentaje_vendido
tipo_conduccion_cultivo
```

Salida:

```text
margen_exportador_soles_kg
```

El precio FOB puede proceder del modelo ganador de S-3 o ser proporcionado manualmente
por el usuario. Los cuatro modelos de S-1 producen estimaciones comparables; el sistema
puede mostrar el resultado de todos o seleccionar automáticamente el modelo con mejor
desempeño validado.

El modelo recomendado actualmente es Random Forest, con R² de 0.777 y MAE aproximado
de S/ 0.343 por kilogramo.

S-1 no debe presentarse únicamente como una resta en la interfaz. La predicción incorpora
el comportamiento aprendido según rendimiento, porcentaje vendido, región, provincia y
tipo de conducción. La fórmula económica fue utilizada para construir el objetivo
histórico, pero la respuesta operativa se obtiene del modelo entrenado.

### S-2 – Simulador de escenarios de campaña

S-2 utiliza los modelos:

- `07_escenario_random_forest`.
- `08_escenario_hist_gradient_boosting`.
- `09_escenario_logistic_regression`.
- `12_escenario_prophet`.

Variables que el usuario puede modificar:

```text
precio_fob_usd_kg
region
rendimiento_kg_ha
porcentaje_vendido
sequia
plagas_enfermedades
```

Salida principal:

```text
probabilidad de riesgo de margen bajo
```

El simulador vuelve a ejecutar los modelos cada vez que el usuario modifica una
condición. Por ejemplo, puede comparar:

- Precio FOB previsto frente a una caída manual del precio.
- Rendimiento normal frente a una reducción por campaña adversa.
- Campaña sin afectación frente a sequía.
- Campaña sin afectación frente a plagas y enfermedades.
- Diferentes porcentajes de producción vendida.

S-2 no se limita a aplicar una regla matemática. Los modelos fueron entrenados para
aprender combinaciones entre precio, productividad, comercialización, ubicación y
eventualidades.

El modelo recomendado para priorizar la detección de riesgos es Logistic Regression,
con recall de 0.865 y ROC AUC de 0.965. Random Forest puede utilizarse como contraste no
lineal.

### S-3 – Modelo predictivo de precios FOB

S-3 utiliza los modelos:

- `01_fob_random_forest`.
- `02_fob_hist_gradient_boosting`.
- `03_fob_elasticnet`.
- `10_fob_prophet`.

Entradas:

```text
destino
temporada
semana_iso
log_volumen_exportado
precio_lag_1
precio_prom_movil_4
```

Salida:

```text
precio FOB esperado en USD/kg a seis semanas
```

S-3 compara cuatro algoritmos y devuelve las predicciones junto con sus métricas. El
precio seleccionado se envía a S-1 para estimar margen y a S-2 para evaluar el riesgo
de campaña.

El modelo recomendado actualmente es Random Forest, con MAE aproximado de 0.309 USD/kg.
Prophet se mantiene como referencia temporal, pero obtuvo menor precisión.

### Flujo operativo completo

1. El usuario selecciona país destino y condiciones comerciales.
2. S-3 predice el precio FOB a seis semanas con los cuatro algoritmos.
3. El usuario selecciona una predicción o ingresa un precio manual.
4. S-1 combina ese precio con las condiciones productivas y estima el margen.
5. S-2 permite modificar precio, rendimiento, porcentaje vendido, sequía y plagas.
6. S-2 devuelve la probabilidad de margen bajo para cada escenario.
7. La interfaz compara resultados entre algoritmos y muestra el modelo recomendado.

## 2. Limpieza y preparación

Antes del entrenamiento se eliminaron los modelos generados anteriormente para evitar
mezclar artefactos de diferentes ejecuciones.

Los datos se tomaron de:

- `precio_palta_semanal.csv`: precio FOB semanal, país destino, volumen y operaciones.
- `costos_produccion/<anio>/Produccion.csv`: producción, venta, ubicación y condiciones productivas.
- `costos_produccion/<anio>/Costos_Produccion.csv`: costos de semillas, abono, fertilizantes y plaguicidas.

Para margen y escenarios se filtró exclusivamente palto con código `1162` y venta
declarada para mercado exterior:

```text
P223_3 = 1
```

El año 2021 no participa en estos dos grupos porque no contiene la variable `P223_3`.

Se normalizaron los cambios de nombres entre encuestas:

- `FACTOR` y `FACTOR_PRODUCTOR`.
- `CONGLOMERADO/NSELUA` y `NSEGM/ID_PROD`.
- Nombres de región y provincia con problemas de tildes o codificación.

Las preguntas originales se transformaron a nombres comprensibles, por ejemplo:

- `P208` -> `tipo_conduccion_cultivo`.
- `P211_8` -> `considero_fertilidad_suelo`.
- `P212` -> `fuente_agua_riego`.
- `P213` -> `sistema_riego`.
- `P223B_1` -> `sequia`.
- `P223B_7` -> `plagas_enfermedades`.

## 3. Datasets finales mínimos

### 3.1 Predicción FOB

Archivo: `outputs/dataset_prediccion_fob_minimo.csv`

Contiene 2,673 registros y estas variables:

- `fecha`: fecha semanal de observación.
- `destino`: país destino.
- `temporada`: estación del año.
- `semana_iso`: semana del año.
- `log_volumen_exportado`: logaritmo de `1 + volumen exportado`.
- `precio_lag_1`: precio FOB anterior del mismo destino.
- `precio_prom_movil_4`: media de los cuatro precios anteriores.
- `target_fob_6_semanas`: precio FOB que se desea predecir.

El objetivo se construyó desplazando seis observaciones hacia adelante el precio de
cada país destino.

### 3.2 Predicción de margen

Archivo: `outputs/dataset_prediccion_margen_minimo.csv`

Contiene 585 registros de mercado exterior:

- `fecha`: mes aproximado de venta, usando el final de cosecha.
- `precio_fob_usd_kg`.
- `region`.
- `provincia`.
- `rendimiento_kg_ha`.
- `porcentaje_vendido`.
- `tipo_conduccion_cultivo`.
- `margen_exportador_soles_kg`: objetivo.

El precio FOB se agregó por año y mes, ponderado por volumen exportado. Para construir
el objetivo se utilizó un tipo de cambio constante de S/ 3.75:

```text
margen_exportador_soles_kg =
    precio_fob_usd_kg * 3.75 - costo_soles_kg
```

El margen es el objetivo histórico usado para entrenar los modelos. Durante una
simulación, el precio FOB podrá proceder del modelo FOB o ser ingresado manualmente.

### 3.3 Predicción de escenarios

Archivo: `outputs/dataset_prediccion_escenarios_minimo.csv`

Contiene 585 registros:

- `fecha`.
- `precio_fob_usd_kg`.
- `region`.
- `rendimiento_kg_ha`.
- `porcentaje_vendido`.
- `sequia`.
- `plagas_enfermedades`.
- `riesgo_margen_bajo`: objetivo binario.

Se definió como margen bajo todo registro ubicado en el 25% inferior de la distribución
del margen exportador:

```text
riesgo_margen_bajo = 1 si margen <= percentil 25
riesgo_margen_bajo = 0 en otro caso
```

La distribución resultante fue:

- Clase 0, margen normal: 438 registros.
- Clase 1, margen bajo: 147 registros.

## 4. Procesamiento compartido

Los modelos de Scikit-learn usan un `Pipeline` con:

- Imputación por mediana para variables numéricas.
- Imputación por moda para variables categóricas.
- One-hot encoding para categorías.
- Categorías desconocidas ignoradas durante la predicción.

ElasticNet y Logistic Regression también estandarizan las variables numéricas.

Para FOB se utiliza una división temporal aproximada:

- Primer 80% de los registros para entrenamiento.
- Último 20% para prueba.
- Sin mezclar aleatoriamente las filas.

Los modelos de margen usan una separación aleatoria reproducible 75/25 con semilla 42.
Los modelos de escenarios usan una separación estratificada 75/25 para conservar la
proporción de las clases.

## 5. Modelos FOB

### 5.1 Random Forest

Script: `entrenamiento/01_fob_random_forest.py`

Configuración:

```text
RandomForestRegressor
n_estimators = 400
min_samples_leaf = 3
random_state = 42
n_jobs = -1
```

Random Forest combina muchos árboles entrenados sobre subconjuntos de los datos. Puede
capturar relaciones no lineales entre precio histórico, volumen, destino y temporada.

Resultado de la última ejecución:

- Entrenamiento: 2,138 registros.
- Prueba: 535 registros.
- MAE: 0.309 USD/kg.
- R²: 0.249.

### 5.2 HistGradientBoosting

Script: `entrenamiento/02_fob_hist_gradient_boosting.py`

Configuración:

```text
HistGradientBoostingRegressor
max_iter = 350
learning_rate = 0.05
l2_regularization = 0.05
random_state = 42
```

Construye árboles secuencialmente, haciendo que cada árbol corrija parte del error del
anterior.

Resultado:

- Entrenamiento: 2,138 registros.
- Prueba: 535 registros.
- MAE: 0.319 USD/kg.
- R²: 0.203.

### 5.3 ElasticNet

Script: `entrenamiento/03_fob_elasticnet.py`

Configuración:

```text
ElasticNet
alpha = 0.01
l1_ratio = 0.25
max_iter = 20000
random_state = 42
```

Es el modelo lineal de referencia. Combina penalizaciones L1 y L2 para reducir
sobreajuste y controlar variables correlacionadas.

Resultado:

- Entrenamiento: 2,138 registros.
- Prueba: 535 registros.
- MAE: 0.324 USD/kg.
- R²: 0.255.

### 5.4 Prophet

Script: `entrenamiento/10_fob_prophet.py`

Prophet usa la fecha como eje temporal, estacionalidad anual y regresores adicionales:

- Volumen exportado transformado.
- Precio anterior.
- Promedio móvil de cuatro observaciones.
- País destino codificado.
- Temporada codificada.

La primera prueba con un Prophet independiente por país fue inestable. Se reemplazó por
un Prophet global con los países y temporadas como regresores categóricos.

Resultado:

- Entrenamiento: 2,138 registros.
- Prueba: 535 registros.
- MAE: 0.365 USD/kg.
- R²: 0.135.

Prophet quedó por debajo de Random Forest y ElasticNet para este dataset.

## 6. Modelos de margen

### 6.1 Random Forest

Script: `entrenamiento/04_margen_random_forest.py`

Entradas:

- Precio FOB.
- Rendimiento.
- Porcentaje vendido.
- Región.
- Provincia.
- Tipo de conducción.

Configuración:

```text
RandomForestRegressor
n_estimators = 400
min_samples_leaf = 3
random_state = 42
n_jobs = -1
```

Resultado:

- Entrenamiento: 438 registros.
- Prueba: 147 registros.
- MAE: S/ 0.343 por kg.
- R²: 0.777.

Fue el mejor modelo de margen en la última ejecución.

### 6.2 HistGradientBoosting

Script: `entrenamiento/05_margen_hist_gradient_boosting.py`

Utiliza las mismas seis variables de entrada.

Resultado:

- Entrenamiento: 438 registros.
- Prueba: 147 registros.
- MAE: S/ 0.382 por kg.
- R²: 0.725.

### 6.3 ElasticNet

Script: `entrenamiento/06_margen_elasticnet.py`

Las variables numéricas son estandarizadas antes del entrenamiento.

Resultado:

- Entrenamiento: 438 registros.
- Prueba: 147 registros.
- MAE: S/ 0.392 por kg.
- R²: 0.724.

El resultado relativamente cercano al boosting indica que el precio FOB explica una
parte importante y aproximadamente lineal del margen.

### 6.4 Prophet

Script: `entrenamiento/11_margen_prophet.py`

Prophet usa:

- Fecha mensual aproximada de venta.
- Precio FOB.
- Rendimiento.
- Porcentaje vendido.
- Región, provincia y conducción codificadas como regresores.

No se activó estacionalidad anual porque solo existen unos pocos años de datos y muchas
observaciones comparten el mismo mes.

Resultado:

- Entrenamiento: 468 registros.
- Prueba temporal: 117 registros.
- MAE: S/ 0.360 por kg.
- R²: 0.266.

El MAE es razonable, pero el R² es mucho menor que el de Random Forest. Prophet no es
el enfoque principal recomendado para margen porque estos datos son principalmente
tabulares y no una serie temporal continua por productor.

## 7. Modelos de escenarios

Estos modelos estiman la probabilidad de pertenecer al grupo de margen bajo. Las
variables ajustables por el simulador son:

- Precio FOB.
- Rendimiento.
- Porcentaje vendido.
- Región.
- Sequía.
- Plagas y enfermedades.

### 7.1 Random Forest Classifier

Script: `entrenamiento/07_escenario_random_forest.py`

Configuración:

```text
RandomForestClassifier
n_estimators = 400
min_samples_leaf = 3
class_weight = balanced
random_state = 42
n_jobs = -1
```

Resultado:

- Accuracy: 0.878.
- Precision: 0.757.
- Recall: 0.757.
- ROC AUC: 0.961.

### 7.2 HistGradientBoosting Classifier

Script: `entrenamiento/08_escenario_hist_gradient_boosting.py`

Resultado:

- Accuracy: 0.878.
- Precision: 0.852.
- Recall: 0.622.
- ROC AUC: 0.956.

Es más preciso cuando declara riesgo, pero deja sin detectar más casos de margen bajo.

### 7.3 Logistic Regression

Script: `entrenamiento/09_escenario_logistic_regression.py`

Configuración:

```text
LogisticRegression
class_weight = balanced
max_iter = 20000
solver = lbfgs
```

Resultado:

- Accuracy: 0.878.
- Precision: 0.711.
- Recall: 0.865.
- ROC AUC: 0.965.

Fue el modelo con mejor capacidad para detectar casos de riesgo y el mejor ROC AUC.

### 7.4 Prophet

Script: `entrenamiento/12_escenario_prophet.py`

Prophet es un modelo de regresión temporal, no un clasificador nativo. Para compararlo
se entrenó con el objetivo binario como valor continuo:

1. Prophet produce un valor estimado.
2. El valor se restringe al intervalo 0–1.
3. Se clasifica como riesgo cuando el resultado es igual o superior a 0.5.

La separación se hizo de forma estratificada. Una primera evaluación temporal produjo
una accuracy engañosa porque casi todos los casos de riesgo quedaron en entrenamiento;
esa evaluación fue descartada y corregida.

Resultado corregido:

- Accuracy: 0.857.
- Precision: 0.833.
- Recall: 0.541.
- ROC AUC: 0.926.

Prophet puede aportar una referencia temporal, pero no reemplaza a los clasificadores
nativos. Logistic Regression y Random Forest son opciones más adecuadas.

## 8. Comparación general

| Solución | Mejor modelo actual | Métrica principal |
|---|---|---|
| FOB a seis semanas | Random Forest | MAE 0.309 USD/kg |
| Margen exportador | Random Forest | R² 0.777 |
| Riesgo de margen bajo | Logistic Regression | ROC AUC 0.965 |

Prophet se conserva como cuarto modelo comparativo para cada solución, pero no fue el
ganador en la evaluación actual.

## 9. Archivos principales

Scripts:

```text
entrenamiento/01_fob_random_forest.py
entrenamiento/02_fob_hist_gradient_boosting.py
entrenamiento/03_fob_elasticnet.py
entrenamiento/04_margen_random_forest.py
entrenamiento/05_margen_hist_gradient_boosting.py
entrenamiento/06_margen_elasticnet.py
entrenamiento/07_escenario_random_forest.py
entrenamiento/08_escenario_hist_gradient_boosting.py
entrenamiento/09_escenario_logistic_regression.py
entrenamiento/10_fob_prophet.py
entrenamiento/11_margen_prophet.py
entrenamiento/12_escenario_prophet.py
```

Soporte:

```text
entrenamiento/common.py
entrenamiento/prophet_common.py
entrenamiento/crear_datasets_minimos.py
entrenamiento/run_all.py
```

Para regenerar los datasets:

```powershell
python .\entrenamiento\crear_datasets_minimos.py
```

Para entrenar todos los modelos:

```powershell
python .\entrenamiento\run_all.py
```

## 10. Limitaciones

- La encuesta no identifica el país al que exportó cada productor. El precio FOB usado
  en margen es un precio mensual nacional ponderado por volumen.
- El tipo de cambio se fijó en S/ 3.75. Para producción debe sustituirse por un dato
  histórico o un valor configurable.
- La fecha de venta se aproxima usando el mes final de cosecha.
- Solo existen 585 registros exportadores utilizables para margen y escenarios.
- El objetivo de margen fue construido a partir del precio FOB y el costo productivo;
  las métricas deben interpretarse teniendo presente esa relación.
- `riesgo_margen_bajo` representa el cuartil inferior de margen, no una pérdida
  necesariamente negativa.
- Prophet funciona mejor con series temporales largas y regulares; margen y escenarios
  tienen estructura principalmente tabular.
- Antes de producción conviene usar validación cruzada temporal, calibración de
  probabilidades y análisis de estabilidad por región y campaña.
