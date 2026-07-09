"""Orquestrador: gera todas as tabelas e gráficos do relatório de uma vez."""

from __future__ import annotations

from pathlib import Path

from dengue_tl.report.data_io import (
    DEFAULT_DPI,
    DEFAULT_OUTPUT_DIR,
    ensure_output_dir,
    load_raw_dataset,
    load_training_results,
    test_date_axis,
)
from dengue_tl.report.plots import (
    plot_architectures_comparison,
    plot_error_by_range,
    plot_learning_curves,
    plot_metrics_comparison,
    plot_model_vs_baselines,
    plot_original_variables,
    plot_residuals,
    plot_residuals_over_time,
    plot_train_test_distribution,
)
from dengue_tl.report.tables import (
    build_comparison_table,
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
    """Gera todas as tabelas e gráficos do relatório para uma arquitetura.

    Os artefatos ficam em `output_dir/<arquitetura>/relatorio/`, de modo que os
    resultados de cnn_lstm, cnn2d e mlp não se sobrescrevem e o relatório fica
    junto dos demais artefatos daquele modelo (ver dengue_tl.paths).
    """
    raw_df = load_raw_dataset(csv_path, date_column=date_column)
    results = load_training_results(results_path)
    arquitetura = str(results.get("config", {}).get("arquitetura") or "modelo")
    output = ensure_output_dir(Path(output_dir) / arquitetura / "relatorio")

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

    daxis = test_date_axis(results)

    artefatos["grafico_variaveis_originais"] = plot_original_variables(
        raw_df,
        output / "grafico_variaveis_originais.png",
        date_column=date_column,
        dpi=dpi,
    )
    artefatos["grafico_modelo_vs_baselines"] = plot_model_vs_baselines(
        results,
        output / "grafico_modelo_vs_baselines.png",
        date_axis=daxis,
        dpi=dpi,
    )
    artefatos["grafico_metricas_comparativas"] = plot_metrics_comparison(
        metrics_df, output / "grafico_metricas_comparativas.png", dpi=dpi
    )
    artefatos["grafico_residuos"] = plot_residuals(
        results, output / "grafico_residuos.png", dpi=dpi
    )
    artefatos["grafico_erro_por_faixa"] = plot_error_by_range(
        error_df, output / "grafico_erro_por_faixa.png", dpi=dpi
    )
    artefatos["grafico_curvas_aprendizado"] = plot_learning_curves(
        results, output / "grafico_curvas_aprendizado.png", dpi=dpi
    )
    artefatos["grafico_residuos_no_tempo"] = plot_residuals_over_time(
        results,
        output / "grafico_residuos_no_tempo.png",
        date_axis=daxis,
        dpi=dpi,
    )
    artefatos["grafico_distribuicao_treino_teste"] = plot_train_test_distribution(
        raw_df, results, output / "grafico_distribuicao_treino_teste.png", dpi=dpi
    )

    return artefatos


def save_comparison_artifacts(
    csv_path: str | Path,
    results_paths: dict[str, str | Path],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    date_column: str = "Data",
    dpi: int = DEFAULT_DPI,
) -> dict[str, Path]:
    """Gera tabela e gráfico comparativos entre arquiteturas.

    `results_paths` deve ser um dict `{label: path}`, ex.:
    `{"CNN-LSTM": "resultados_cnn_lstm.json", "CNN2D": "resultados_cnn2d.json"}`.
    Os artefatos ficam em `output_dir/comparacao/`.
    """
    raw_df = load_raw_dataset(csv_path, date_column=date_column)
    output = ensure_output_dir(Path(output_dir) / "comparacao")

    def _desembrulha(r: dict) -> dict:
        """Extrai resultado_final de JSONs de otimizacao, passando treino direto."""
        return r.get("resultado_final", r)

    entries = [
        (label, _desembrulha(load_training_results(path)))
        for label, path in results_paths.items()
    ]

    artefatos: dict[str, Path] = {}

    tabela = build_comparison_table(entries)
    artefatos["tabela_comparacao_arquiteturas"] = save_table(
        tabela, output / "tabela_comparacao_arquiteturas.csv"
    )

    if len(entries) >= 2:
        # Converte para o formato (results, label) esperado pela funcao de plot.
        artefatos["grafico_comparacao_arquiteturas"] = plot_architectures_comparison(
            [(results, label) for label, results in entries],
            output / "grafico_comparacao_arquiteturas.png",
            dpi=dpi,
        )

    return artefatos
