"""Build the complete LaTeX report as a PDF.

Usage:
    python build_pdf.py
    python build_pdf.py --engine xelatex
    python build_pdf.py --output dist/informe-palta.pdf

The script compiles main.tex and writes the final PDF to dist/informe.pdf by
default. It uses an installed LaTeX engine: latexmk, tectonic, xelatex or
pdflatex.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAIN_TEX = ROOT / "main.tex"
BUILD_DIR = ROOT / "build"
DEFAULT_OUTPUT = ROOT / "dist" / "informe.pdf"


ENGINES = ("latexmk", "tectonic", "xelatex", "pdflatex")


def find_engine(preferred: str) -> str | None:
    """Return the selected LaTeX engine executable name, if available."""
    if preferred != "auto":
        return preferred if shutil.which(preferred) else None

    for engine in ENGINES:
        if shutil.which(engine):
            return engine
    return None


def run_command(command: list[str]) -> None:
    """Run a command in the project root and stream output to the terminal."""
    print(f"\n> {' '.join(command)}\n")
    subprocess.run(command, cwd=ROOT, check=True)


def compile_with_latexmk() -> Path:
    BUILD_DIR.mkdir(exist_ok=True)
    run_command(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={BUILD_DIR}",
            str(MAIN_TEX.name),
        ]
    )
    return BUILD_DIR / "main.pdf"


def compile_with_tectonic() -> Path:
    BUILD_DIR.mkdir(exist_ok=True)
    run_command(
        [
            "tectonic",
            "--keep-logs",
            "--keep-intermediates",
            "--outdir",
            str(BUILD_DIR),
            str(MAIN_TEX.name),
        ]
    )
    return BUILD_DIR / "main.pdf"


def compile_with_classic_engine(engine: str, passes: int) -> Path:
    BUILD_DIR.mkdir(exist_ok=True)
    command = [
        engine,
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={BUILD_DIR}",
        str(MAIN_TEX.name),
    ]

    for _ in range(passes):
        run_command(command)

    return BUILD_DIR / "main.pdf"


def build_pdf(engine: str, output: Path, passes: int) -> Path:
    if not MAIN_TEX.exists():
        raise FileNotFoundError(f"No se encontró {MAIN_TEX}")

    selected_engine = find_engine(engine)
    if not selected_engine:
        available = ", ".join(ENGINES)
        raise RuntimeError(
            "No se encontró un motor LaTeX instalado.\n"
            f"Instala uno de estos motores y vuelve a ejecutar el script: {available}.\n"
            "Recomendación en Windows: instala MiKTeX o TeX Live, o instala Tectonic "
            "y asegúrate de que el comando quede disponible en PATH."
        )

    print(f"Motor LaTeX seleccionado: {selected_engine}")

    if selected_engine == "latexmk":
        built_pdf = compile_with_latexmk()
    elif selected_engine == "tectonic":
        built_pdf = compile_with_tectonic()
    else:
        built_pdf = compile_with_classic_engine(selected_engine, passes)

    if not built_pdf.exists():
        raise FileNotFoundError(f"La compilación terminó, pero no se generó {built_pdf}")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built_pdf, output)
    return output


def clean() -> None:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    print("Carpeta build eliminada.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compila el informe LaTeX completo y genera un PDF."
    )
    parser.add_argument(
        "--engine",
        choices=("auto", *ENGINES),
        default="auto",
        help="Motor LaTeX a usar. Por defecto detecta automáticamente.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Ruta del PDF final. Por defecto: dist/informe.pdf",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=2,
        help="Número de pasadas para xelatex/pdflatex. Por defecto: 2.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Elimina la carpeta build antes de compilar.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if args.clean:
            clean()

        output = build_pdf(args.engine, args.output, args.passes)
    except subprocess.CalledProcessError as exc:
        print("\nLa compilación falló.", file=sys.stderr)
        print(f"Comando: {' '.join(exc.cmd)}", file=sys.stderr)
        print("Revisa el log generado en la carpeta build.", file=sys.stderr)
        return exc.returncode or 1
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1

    print(f"\nPDF generado correctamente: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
