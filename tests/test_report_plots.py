import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

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
from dengue_tl.report.tables import build_error_by_range_table, build_metrics_table

DPI_TESTE = 60  # baixa resolução: os testes só verificam que o PNG é gerado


def _dados_exemplo(n=40):
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "Precipitacao": rng.uniform(0, 100, n),
            "Temp_media": rng.uniform(15, 30, n),
            "Umidade_rel": rng.uniform(60, 100, n),
            "Qtde_Casos": rng.integers(0, 50, n).astype(float),
        }
    )


def _resultados_exemplo(n=40):
    rng = np.random.default_rng(7)
    y_true = rng.uniform(0, 100, n).tolist()
    return {
        "metricas": {
            "modelo": {"mae": 38.1, "rmse": 107.4, "cc": 0.63},
            "baseline_media": {"mae": 40.2, "rmse": 110.7, "cc": 0.0},
            "baseline_historico": {"mae": 40.3, "rmse": 98.9, "cc": 0.55},
        },
        "predicoes_teste": {
            "y_true": y_true,
            "y_pred_modelo": [v + 1.0 for v in y_true],
            "y_pred_baseline_media": [50.0] * n,
            "y_pred_baseline_historico": [v - 2.0 for v in y_true],
        },
    }


def test_plot_case_series_gera_png(tmp_path):
    caminho = plot_case_series(
        _dados_exemplo(), tmp_path / "serie.png", dpi=DPI_TESTE
    )

    assert caminho.exists()
    assert caminho.stat().st_size > 0


def test_plot_original_variables_gera_png(tmp_path):
    caminho = plot_original_variables(
        _dados_exemplo(), tmp_path / "variaveis.png", dpi=DPI_TESTE
    )

    assert caminho.exists()


def test_plot_original_variables_exige_colunas_numericas(tmp_path):
    # Regressão do gráfico quebrado: coluna corrompida (string) deve falhar
    # alto aqui, não gerar um PNG com escala absurda.
    dados = _dados_exemplo()
    dados["Temp_media"] = "1.595.629.411.764.700"

    with pytest.raises(ValueError, match="Temp_media"):
        plot_original_variables(dados, tmp_path / "variaveis.png", dpi=DPI_TESTE)


def test_plot_real_vs_pred_gera_png(tmp_path):
    caminho = plot_real_vs_pred(
        _resultados_exemplo(), tmp_path / "real_vs_pred.png", dpi=DPI_TESTE
    )

    assert caminho.exists()


def test_plot_model_vs_baselines_gera_png(tmp_path):
    caminho = plot_model_vs_baselines(
        _resultados_exemplo(), tmp_path / "baselines.png", dpi=DPI_TESTE
    )

    assert caminho.exists()


def test_plot_metrics_comparison_gera_png(tmp_path):
    metrics_df = build_metrics_table(_resultados_exemplo())

    caminho = plot_metrics_comparison(
        metrics_df, tmp_path / "metricas.png", dpi=DPI_TESTE
    )

    assert caminho.exists()


def test_plot_dispersao_real_previsto_gera_png(tmp_path):
    caminho = plot_dispersao_real_previsto(
        _resultados_exemplo(), tmp_path / "dispersao.png", dpi=DPI_TESTE
    )

    assert caminho.exists()


def test_plot_residuals_gera_png(tmp_path):
    caminho = plot_residuals(
        _resultados_exemplo(), tmp_path / "residuos.png", dpi=DPI_TESTE
    )

    assert caminho.exists()


def test_plot_error_by_range_gera_png(tmp_path):
    error_df = build_error_by_range_table(_resultados_exemplo())

    caminho = plot_error_by_range(error_df, tmp_path / "erro_faixa.png", dpi=DPI_TESTE)

    assert caminho.exists()


def test_plot_error_by_range_aceita_tabela_vazia(tmp_path):
    caminho = plot_error_by_range(
        pd.DataFrame(), tmp_path / "erro_vazio.png", dpi=DPI_TESTE
    )

    assert caminho.exists()
