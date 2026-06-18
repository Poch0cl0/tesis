from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "entrenamiento_modelos_palta.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def build_notebook():
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb.cells = [
        md(
            "# Entrenamiento de modelos para rentabilidad y escenarios de campaña de palta\n\n"
            "Este notebook construye datasets enriquecidos desde los CSV fuente, entrena modelos para apoyar "
            "S-1 y S-2, guarda artefactos en `models/` y exporta tablas listas para el dashboard en `outputs/`."
        ),
        code(
            "from pathlib import Path\n"
            "import json\n"
            "import sys\n"
            "import pandas as pd\n"
            "ROOT = Path.cwd()\n"
            "sys.path.insert(0, str(ROOT / 'src'))\n"
            "from modeling_pipeline import build_enriched_cost_dataset, build_price_features, train_models\n"
            "pd.set_option('display.max_columns', 120)\n"
            "pd.set_option('display.width', 180)\n"
            "ROOT"
        ),
        md("## 1. Construcción de datasets enriquecidos"),
        code(
            "cost_df = build_enriched_cost_dataset(ROOT)\n"
            "price_df = build_price_features(ROOT)\n"
            "print('Filas costo enriquecido:', len(cost_df))\n"
            "print('Filas precio FOB:', len(price_df))\n"
            "display(cost_df.head())\n"
            "display(price_df.head())"
        ),
        md("## 2. Variables principales disponibles"),
        code(
            "cost_summary = cost_df.groupby('ANIO').agg(\n"
            "    registros=('costo_por_kg', 'count'),\n"
            "    costo_mediano_kg=('costo_por_kg', 'median'),\n"
            "    costo_promedio_kg=('costo_por_kg', 'mean'),\n"
            "    produccion_total_kg=('produccion_kg', 'sum'),\n"
            "    rendimiento_mediano_kg_ha=('rendimiento_kg_ha', 'median'),\n"
            "    pct_outliers=('es_outlier', 'mean'),\n"
            ").round(3)\n"
            "display(cost_summary)\n"
            "price_summary = price_df.groupby('destino').agg(\n"
            "    registros=('precio_fob_por_kilogramo', 'count'),\n"
            "    volumen_total=('volumen_exportado', 'sum'),\n"
            "    precio_promedio=('precio_fob_por_kilogramo', 'mean'),\n"
            ").sort_values('volumen_total', ascending=False).head(12).round(3)\n"
            "display(price_summary)"
        ),
        md("## 3. Entrenamiento y evaluación"),
        code(
            "result = train_models(ROOT, ROOT / 'models')\n"
            "metrics = result['metrics']\n"
            "metadata = result['metadata']\n"
            "print(json.dumps(metrics, indent=2))\n"
            "print('\\nModelos guardados en:', ROOT / 'models')"
        ),
        md(
            "## 4. Lectura de métricas\n\n"
            "- `cost_model`: estima costo productivo por kg a partir de variables productivas, agronómicas y comerciales. "
            "Se entrena sin outliers IQR.\n"
            "- `price_model`: estima precio FOB por kg según destino, semana, temporada, volumen, operaciones y señales recientes.\n"
            "- `risk_model`: estima probabilidad de producción afectada cuando la variable `P223A` está disponible."
        ),
        code(
            "for name, values in metrics.items():\n"
            "    print(f'\\n{name}')\n"
            "    print(values)\n"
            "print('\\nArchivos generados:')\n"
            "for path in sorted((ROOT / 'models').glob('*')):\n"
            "    print('-', path.name)\n"
            "for path in sorted((ROOT / 'outputs').glob('*.csv')):\n"
            "    print('-', path.name)"
        ),
    ]
    return nb


if __name__ == "__main__":
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    nbf.write(nb, NOTEBOOK_PATH)
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Notebook ejecutado: {NOTEBOOK_PATH}")
