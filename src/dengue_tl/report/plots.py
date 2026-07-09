"""Gráficos do relatório em matplotlib (backend Agg, sem display)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.dates as mdates

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


def _formata_eixo_datas(ax: plt.Axes, fig: plt.Figure) -> None:
    """Aplica formatação de datas ao eixo X (intervalo trimestral)."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30, ha="right")


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> Path:
    """Salva uma figura em PNG com qualidade alta."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_original_variables(
    raw_df: pd.DataFrame, output_path: Path, date_column: str = "Data", dpi: int = DEFAULT_DPI
) -> Path:
    """Subplots das 4 variáveis originais ao longo do tempo."""
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


def plot_model_vs_baselines(
    results: dict[str, Any],
    output_path: Path,
    date_axis: pd.DatetimeIndex | None = None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Série temporal do teste: real, modelo e baselines.

    Usa eixo de datas quando `date_axis` é fornecido (ver `test_date_axis`),
    ou índice sequencial como fallback.
    """
    preds = extract_test_predictions(results)
    x = date_axis if date_axis is not None else np.arange(len(preds.y_true))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(x, preds.y_true, label="Real", color="#1f77b4", linewidth=1.8)
    ax.plot(x, preds.y_model, label="Modelo", color="#d62728", linewidth=1.4)
    ax.plot(x, preds.y_media, label="Baseline média", color="#2ca02c", linestyle="--", alpha=0.85)
    ax.plot(x, preds.y_hist, label="Baseline histórico", color="#ff7f0e", linestyle=":", alpha=0.85)
    ax.set_title("Real vs. modelo e baselines no conjunto de teste")
    ax.set_ylabel(VARIAVEL_ALVO)
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.25)
    if date_axis is not None:
        ax.set_xlabel("Data")
        _formata_eixo_datas(ax, fig)
    else:
        ax.set_xlabel("Amostra do teste")
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


# -------------------------------------------------- novos gráficos -----------


def plot_learning_curves(
    results: dict[str, Any], output_path: Path, dpi: int = DEFAULT_DPI
) -> Path:
    """Loss e val_loss por época — mostra convergência e overfitting."""
    historico = results.get("historico", {}).get("treino", {})
    loss = historico.get("loss", [])
    val_loss = historico.get("val_loss", [])

    fig, ax = plt.subplots(figsize=(8, 4))
    if not loss:
        ax.axis("off")
        ax.text(0.5, 0.5, "Historico de treino nao disponivel", ha="center", va="center")
        return save_figure(fig, output_path, dpi)

    epochs = np.arange(1, len(loss) + 1)
    ax.plot(epochs, loss, label="Treino", color="#1f77b4")
    ax.plot(epochs, val_loss, label="Validacao", color="#d62728", linestyle="--")
    ax.set_title("Curva de aprendizado (loss por epoca)")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Loss (log-razao, escala normalizada)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    return save_figure(fig, output_path, dpi)


def plot_residuals_over_time(
    results: dict[str, Any],
    output_path: Path,
    date_axis: pd.DatetimeIndex | None = None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Resíduo (real − previsto) ao longo do tempo, com casos reais acima.

    Deixa visível se o modelo sistematicamente subestima nos picos de surto.
    """
    preds = extract_test_predictions(results)
    residuos = preds.y_true - preds.y_model
    x = date_axis if date_axis is not None else np.arange(len(preds.y_true))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax1.plot(x, preds.y_true, color="#1f77b4", linewidth=1.2, label="Real")
    ax1.set_ylabel(VARIAVEL_ALVO)
    ax1.set_title("Casos reais e residuo do modelo ao longo do tempo")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.25)

    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.fill_between(
        x, residuos, 0,
        where=residuos > 0, color="#d62728", alpha=0.5, label="Subestimou"
    )
    ax2.fill_between(
        x, residuos, 0,
        where=residuos < 0, color="#2ca02c", alpha=0.5, label="Superestimou"
    )
    ax2.set_ylabel("Real − Previsto")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.25)

    if date_axis is not None:
        ax2.set_xlabel("Data")
        _formata_eixo_datas(ax2, fig)
    else:
        ax2.set_xlabel("Amostra do teste")

    return save_figure(fig, output_path, dpi)


def plot_train_test_distribution(
    raw_df: pd.DataFrame,
    results: dict[str, Any],
    output_path: Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Histograma sobreposto de Qtde_Casos no treino vs. teste.

    Evidencia o distribution shift: treino com casos baixos, teste com surtos
    maiores — contexto essencial para interpretar as métricas.
    """
    _valida_colunas_numericas(raw_df, [VARIAVEL_ALVO])
    config = results.get("config", {})
    split = results.get("split", {})
    lag_clima = int(config.get("lag_clima", 45))
    raio = int(config.get("raio", 4))
    offset = lag_clima + raio

    casos = raw_df[VARIAVEL_ALVO].to_numpy(dtype=float)
    t_ini = offset + int(split.get("treino", [0, 0])[0])
    t_fim = offset + int(split.get("treino", [0, 0])[1])
    te_ini = offset + int(split.get("teste", [0, 0])[0])
    te_fim = offset + int(split.get("teste", [0, 0])[1])
    y_treino = casos[t_ini : min(t_fim, len(casos))]
    y_teste = casos[te_ini : min(te_fim, len(casos))]

    if y_treino.size == 0 or y_teste.size == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "Dados insuficientes para distribuicao", ha="center", va="center")
        return save_figure(fig, output_path, dpi)

    maximo = float(max(casos.max(), 1.0))
    p95 = float(np.percentile(casos, 95))
    bins_full = np.linspace(0, maximo, 40)
    bins_zoom = np.linspace(0, p95, 30)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.hist(y_treino, bins=bins_full, color="#1f77b4", alpha=0.7,
             label=f"Treino (max={y_treino.max():.0f})")
    ax1.hist(y_teste, bins=bins_full, color="#d62728", alpha=0.7,
             label=f"Teste (max={y_teste.max():.0f})")
    ax1.set_xlabel("Casos/dia")
    ax1.set_ylabel("Frequência")
    ax1.set_title("Distribuição completa")
    ax1.legend()
    ax1.grid(True, alpha=0.25)

    ax2.hist(y_treino[y_treino <= p95], bins=bins_zoom, color="#1f77b4", alpha=0.7,
             label="Treino")
    ax2.hist(y_teste[y_teste <= p95], bins=bins_zoom, color="#d62728", alpha=0.7,
             label="Teste")
    ax2.set_xlabel("Casos/dia")
    ax2.set_title(f"Zoom: até p95 ({p95:.0f} casos)")
    ax2.legend()
    ax2.grid(True, alpha=0.25)

    fig.suptitle("Deslocamento de distribuição: treino → teste")
    return save_figure(fig, output_path, dpi)


def plot_architectures_comparison(
    entries: list[tuple[dict[str, Any], str]],
    output_path: Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Previsões de N arquiteturas sobrepostas ao real no teste.

    `entries` é uma lista de `(results_dict, label)` — um por arquitetura.
    """
    from dengue_tl.report.data_io import test_date_axis

    if not entries:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.axis("off")
        return save_figure(fig, output_path, dpi)

    cores_modelos = ["#d62728", "#9467bd", "#2ca02c", "#ff7f0e", "#8c564b"]
    estilos = ["-", "--", "-.", ":", "-"]

    results_ref, _ = entries[0]
    date_axis = test_date_axis(results_ref)
    preds_ref = extract_test_predictions(results_ref)
    n = min(len(extract_test_predictions(r).y_true) for r, _ in entries)
    x = date_axis[:n] if date_axis is not None else np.arange(n)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(x, preds_ref.y_true[:n], label="Real", color="#1f77b4", linewidth=1.8)

    for (results, label), cor, estilo in zip(entries, cores_modelos, estilos):
        preds = extract_test_predictions(results)
        ax.plot(x, preds.y_model[:n], label=label, color=cor, linewidth=1.2, linestyle=estilo)

    labels = " vs. ".join(label for _, label in entries)
    ax.set_title(f"Comparacao: {labels}")
    ax.set_ylabel(VARIAVEL_ALVO)
    ax.legend(ncol=len(entries) + 1)
    ax.grid(True, alpha=0.25)
    if date_axis is not None:
        ax.set_xlabel("Data")
        _formata_eixo_datas(ax, fig)
    else:
        ax.set_xlabel("Amostra do teste")
    return save_figure(fig, output_path, dpi)
