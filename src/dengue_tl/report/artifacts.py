"""Orquestrador: gera todas as tabelas e gráficos do relatório de uma vez."""

from __future__ import annotations

from pathlib import Path

from dengue_tl.report.data_io import (
    DEFAULT_DPI,
    DEFAULT_OUTPUT_DIR,
    ensure_output_dir,
    load_raw_dataset,
    load_training_results,
)
from dengue_tl.report.plots import (
    plot_case_series,
    plot_dispersao_real_previsto,
    plot_error_by_range,
    plot_metrics_comparison,
    plot_model_vs_baselines,
    plot_original_variables,
    plot_real_vs_pred,
    plot_residuals,
)
from dengue_tl.report.tables import (
    build_data_structure_table,
    build_error_by_range_table,
    build_metrics_table,
    build_pipeline_config_table,
    build_split_table,
    save_table,
)


def save_all_report_artifacts(
    csv_path: str | Path,
    results_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    date_column: str = "Data",
    dpi: int = DEFAULT_DPI,
) -> dict[str, Path]:
    """Gera todas as tabelas e gráficos do relatório.

    Os artefatos são separados por modelo: ficam em
    `output_dir/<arquitetura>/`, com a arquitetura lida do JSON de
    resultados (`modelo` quando o JSON não a informa). Assim os
    resultados de cnn_lstm, cnn2d e efficientnet não se sobrescrevem.
    """
    raw_df = load_raw_dataset(csv_path, date_column=date_column)
    results = load_training_results(results_path)
    arquitetura = str(results.get("config", {}).get("arquitetura") or "modelo")
    output = ensure_output_dir(Path(output_dir) / arquitetura)

    metrics_df = build_metrics_table(results)
    error_df = build_error_by_range_table(results)
    tabelas = {
        "tabela_estrutura_dados": build_data_structure_table(
            raw_df, date_column=date_column
        ),
        "tabela_configuracao_pipeline": build_pipeline_config_table(results),
        "tabela_split_temporal": build_split_table(results),
        "tabela_metricas_modelos": metrics_df,
        "tabela_erro_por_faixa": error_df,
    }

    artefatos: dict[str, Path] = {
        nome: save_table(df, output / f"{nome}.csv") for nome, df in tabelas.items()
    }

    artefatos["grafico_serie_casos"] = plot_case_series(
        raw_df, output / "grafico_serie_casos.png", date_column=date_column, dpi=dpi
    )
    artefatos["grafico_variaveis_originais"] = plot_original_variables(
        raw_df,
        output / "grafico_variaveis_originais.png",
        date_column=date_column,
        dpi=dpi,
    )
    artefatos["grafico_real_vs_previsto"] = plot_real_vs_pred(
        results, output / "grafico_real_vs_previsto.png", dpi=dpi
    )
    artefatos["grafico_modelo_vs_baselines"] = plot_model_vs_baselines(
        results, output / "grafico_modelo_vs_baselines.png", dpi=dpi
    )
    artefatos["grafico_metricas_comparativas"] = plot_metrics_comparison(
        metrics_df, output / "grafico_metricas_comparativas.png", dpi=dpi
    )
    artefatos["grafico_dispersao_real_previsto"] = plot_dispersao_real_previsto(
        results, output / "grafico_dispersao_real_previsto.png", dpi=dpi
    )
    artefatos["grafico_residuos"] = plot_residuals(
        results, output / "grafico_residuos.png", dpi=dpi
    )
    artefatos["grafico_erro_por_faixa"] = plot_error_by_range(
        error_df, output / "grafico_erro_por_faixa.png", dpi=dpi
    )

    return artefatos
