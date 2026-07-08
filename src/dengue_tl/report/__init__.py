"""Geração de tabelas e gráficos para o relatório do projeto.

O pacote lê o CSV bruto e o JSON de resultados do treino, produzindo artefatos
em CSV/PNG sem alterar a lógica de treino. Módulos:

- `data_io`: carregamento do CSV bruto (com reparo numérico) e do JSON de resultados
- `tables`: construção das tabelas do relatório
- `plots`: construção dos gráficos do relatório
- `artifacts`: orquestrador que gera todos os artefatos de uma vez
- `cli`: interface de linha de comando (`python -m dengue_tl.report`)
"""

from dengue_tl.report.artifacts import save_all_report_artifacts
from dengue_tl.report.data_io import (
    ReportConfig,
    extract_test_predictions,
    load_raw_dataset,
    load_training_results,
)
from dengue_tl.report.tables import (
    build_data_structure_table,
    build_error_by_range_table,
    build_metrics_table,
    build_pipeline_config_table,
    build_split_table,
)

__all__ = [
    "ReportConfig",
    "build_data_structure_table",
    "build_error_by_range_table",
    "build_metrics_table",
    "build_pipeline_config_table",
    "build_split_table",
    "extract_test_predictions",
    "load_raw_dataset",
    "load_training_results",
    "save_all_report_artifacts",
]
