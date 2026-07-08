"""Gráficos do relatório em matplotlib (backend Agg, sem display)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dengue_tl.lagged_table import VARIAVEL_ALVO, VARIAVEIS_CLIMATICAS
from dengue_tl.report.data_io import DEFAULT_DPI, extract_test_predictions, time_axis


def _valida_colunas_numericas(df: pd.DataFrame, colunas: list[str]) -> None:
    """Falha alto se alguma coluna não for numérica.

    O matplotlib plota strings como eixo categórico sem reclamar, gerando
    gráficos com escala absurda quando o CSV tem valores corrompidos que
    escaparam do reparo (`repara_valores_numericos`).
    """
    nao_numericas = [
        coluna for coluna in colunas if not pd.api.types.is_numeric_dtype(df[coluna])
    ]
    if nao_numericas:
        raise ValueError(
            "colunas nao numericas para plotar (CSV corrompido? use "
            "`load_raw_dataset`): " + ", ".join(nao_numericas)
        )


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> Path:
    """Salva uma figura em PNG com qualidade alta."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_case_series(
    raw_df: pd.DataFrame, output_path: Path, date_column: str = "Data", dpi: int = DEFAULT_DPI
) -> Path:
    """Gráfico da série temporal de `Qtde_Casos`."""
    _valida_colunas_numericas(raw_df, [VARIAVEL_ALVO])
    eixo = time_axis(raw_df, date_column=date_column)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(eixo, raw_df[VARIAVEL_ALVO], color="#1f77b4", linewidth=1.5)
    ax.set_title("Serie temporal dos casos de dengue")
    ax.set_xlabel("Tempo")
    ax.set_ylabel(VARIAVEL_ALVO)
    ax.grid(True, alpha=0.25)
    return save_figure(fig, output_path, dpi)


def plot_original_variables(
    raw_df: pd.DataFrame, output_path: Path, date_column: str = "Data", dpi: int = DEFAULT_DPI
) -> Path:
    """Gráfico das variáveis originais em subplots."""
    cols = [*VARIAVEIS_CLIMATICAS, VARIAVEL_ALVO]
    _valida_colunas_numericas(raw_df, cols)
    eixo = time_axis(raw_df, date_column=date_column)
    fig, axes = plt.subplots(len(cols), 1, figsize=(12, 10), sharex=True)
    cores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for ax, coluna, cor in zip(axes, cols, cores):
        ax.plot(eixo, raw_df[coluna], color=cor, linewidth=1.1)
        ax.set_title(coluna)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Tempo")
    fig.supylabel("Valor")
    fig.suptitle("Variaveis originais ao longo do tempo")
    return save_figure(fig, output_path, dpi)


def plot_real_vs_pred(
    results: dict[str, Any], output_path: Path, dpi: int = DEFAULT_DPI
) -> Path:
    """Gráfico principal comparando real e previsto pelo modelo."""
    preds = extract_test_predictions(results)
    x = np.arange(len(preds.y_true))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, preds.y_true, label="Real", color="#1f77b4", linewidth=1.5)
    ax.plot(x, preds.y_model, label="Previsto pelo modelo", color="#d62728", linewidth=1.5)
    ax.set_title("Real x previsto no conjunto de teste")
    ax.set_xlabel("Amostra do teste")
    ax.set_ylabel(VARIAVEL_ALVO)
    ax.legend()
    ax.grid(True, alpha=0.25)
    return save_figure(fig, output_path, dpi)


def plot_model_vs_baselines(
    results: dict[str, Any], output_path: Path, dpi: int = DEFAULT_DPI
) -> Path:
    """Compara modelo e baselines no teste."""
    preds = extract_test_predictions(results)
    x = np.arange(len(preds.y_true))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, preds.y_true, label="Real", color="#1f77b4", linewidth=1.8)
    ax.plot(x, preds.y_model, label="Modelo", color="#d62728", linewidth=1.4)
    ax.plot(x, preds.y_media, label="Baseline média", color="#2ca02c", linestyle="--")
    ax.plot(x, preds.y_hist, label="Baseline histórico", color="#ff7f0e", linestyle=":")
    ax.set_title("Comparação entre modelo e baselines no teste")
    ax.set_xlabel("Amostra do teste")
    ax.set_ylabel(VARIAVEL_ALVO)
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.25)
    return save_figure(fig, output_path, dpi)


def plot_metrics_comparison(
    metrics_df: pd.DataFrame, output_path: Path, dpi: int = DEFAULT_DPI
) -> Path:
    """Gráfico de barras com MAE, RMSE e CC."""
    metricas = ["MAE", "RMSE", "CC"]
    metodos = metrics_df["metodo"].tolist()
    cores = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metrica in zip(axes, metricas):
        valores = metrics_df[metrica].to_numpy(dtype=float)
        barras = ax.bar(metodos, valores, color=cores)
        ax.set_title(metrica)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.25)
        for barra, valor in zip(barras, valores):
            ax.annotate(
                f"{valor:.2f}",
                xy=(barra.get_x() + barra.get_width() / 2, barra.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    fig.suptitle("Comparacao das metricas entre modelo e baselines")
    return save_figure(fig, output_path, dpi)


def plot_dispersao_real_previsto(
    results: dict[str, Any], output_path: Path, dpi: int = DEFAULT_DPI
) -> Path:
    """Scatter de real contra previsto com linha ideal."""
    preds = extract_test_predictions(results)
    minimo = float(min(preds.y_true.min(), preds.y_model.min()))
    maximo = float(max(preds.y_true.max(), preds.y_model.max()))
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(preds.y_true, preds.y_model, alpha=0.75, color="#1f77b4", edgecolor="white")
    ax.plot([minimo, maximo], [minimo, maximo], color="#d62728", linestyle="--")
    ax.set_title("Dispersao: real x previsto")
    ax.set_xlabel("Real")
    ax.set_ylabel("Previsto pelo modelo")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(minimo, maximo)
    ax.set_ylim(minimo, maximo)
    return save_figure(fig, output_path, dpi)


def plot_residuals(
    results: dict[str, Any], output_path: Path, dpi: int = DEFAULT_DPI
) -> Path:
    """Histograma e boxplot dos resíduos do modelo."""
    preds = extract_test_predictions(results)
    residuos = preds.y_true - preds.y_model
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(residuos, bins=15, color="#9467bd", alpha=0.85)
    axes[0].set_title("Histograma dos residuos")
    axes[0].set_xlabel("Residuo")
    axes[0].set_ylabel("Frequencia")
    axes[0].grid(True, alpha=0.25)

    axes[1].boxplot(residuos, patch_artist=True)
    axes[1].set_title("Boxplot dos residuos")
    axes[1].set_ylabel("Residuo")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle("Analise dos residuos do modelo")
    return save_figure(fig, output_path, dpi)


def plot_error_by_range(
    error_df: pd.DataFrame, output_path: Path, dpi: int = DEFAULT_DPI
) -> Path:
    """Gráfico de erro por faixa de incidência."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if error_df.empty:
        for ax in axes:
            ax.axis("off")
        return save_figure(fig, output_path, dpi)

    for ax, metrica in zip(axes, ["MAE", "RMSE"]):
        ax.bar(error_df["faixa"], error_df[metrica], color="#1f77b4")
        ax.set_title(metrica)
        ax.set_xlabel("Faixa de incidencia")
        ax.set_ylabel(metrica)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Erro do modelo por faixa de incidencia")
    return save_figure(fig, output_path, dpi)
