"""Interface de linha de comando do relatório.

Uso:
    python -m dengue_tl.report --csv "data/Dados 2007-2024.csv" \
        --results resultados_cnn_lstm.json
"""

from __future__ import annotations

import argparse

from dengue_tl.report.artifacts import save_all_report_artifacts
from dengue_tl.report.data_io import DEFAULT_DPI, DEFAULT_OUTPUT_DIR


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera tabelas e graficos para o relatorio do projeto."
    )
    parser.add_argument("--csv", required=True, help="CSV bruto do projeto.")
    parser.add_argument(
        "--results",
        required=True,
        help="JSON de resultados gerado pelo train_runner.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Pasta base de saida; os artefatos vao para uma subpasta "
            "com o nome da arquitetura do JSON de resultados."
        ),
    )
    parser.add_argument(
        "--date-column",
        default="Data",
        help="Nome da coluna de data, se existir no CSV bruto.",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parser.parse_args()


def main() -> None:
    """Entrada de linha de comando."""
    args = _parse_args()
    artefatos = save_all_report_artifacts(
        csv_path=args.csv,
        results_path=args.results,
        output_dir=args.output_dir,
        date_column=args.date_column,
        dpi=args.dpi,
    )
    print("Artefatos gerados:")
    for nome, caminho in artefatos.items():
        print(f"- {nome}: {caminho}")


if __name__ == "__main__":
    main()
