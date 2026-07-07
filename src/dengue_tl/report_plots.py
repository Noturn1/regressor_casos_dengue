"""Gera tabelas e graficos para o relatorio do projeto.

O modulo le o CSV bruto e o JSON de resultados do treino, produzindo artefatos
em CSV/PNG sem alterar a logica de treino.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dengue_tl.encoder import TAMANHO_IMAGEM
from dengue_tl.lagged_table import VARIAVEL_ALVO
from dengue_tl.matrix_windower import feature_columns

BACKBONE_NOME = "efficientnetb0"
DEFAULT_OUTPUT_DIR = Path("relatorio_outputs")
DEFAULT_DPI = 220


@dataclass(frozen=True)
class ReportConfig:
    """Configuracao de entrada e saida do gerador de relatorio."""

    csv_path: str
    results_path: str
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    date_column: str = "Data"
    dpi: int = DEFAULT_DPI


def _ensure_output_dir(output_dir: str | Path) -> Path:
    """Cria a pasta de saida se ela nao existir."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    return output


def load_training_results(results_path: str | Path) -> dict[str, Any]:
    """Carrega o JSON produzido pelo runner."""
    with Path(results_path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_raw_dataset(csv_path: str | Path, date_column: str = "Data") -> pd.DataFrame:
    """Le o CSV bruto preservando data quando existir."""
    csv_path = Path(csv_path)
    cabecalho = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if date_column in cabecalho:
        df = pd.read_csv(csv_path, parse_dates=[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)
        return df
    return pd.read_csv(csv_path)


def _time_axis(df: pd.DataFrame, date_column: str = "Data") -> pd.Series:
    """Retorna eixo temporal com data ou indice sequencial."""
    if date_column in df.columns:
        return pd.to_datetime(df[date_column])
    return pd.Series(np.arange(len(df)), name="indice")


def _format_value(value: Any) -> Any:
    """Formata valores para escrita em CSV."""
    if pd.isna(value):
        return ""
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def build_data_structure_table(
    raw_df: pd.DataFrame, date_column: str = "Data"
) -> pd.DataFrame:
    """Resume colunas, papel, tipo e estatisticas basicas dos dados brutos."""
    roles = {
        "Precipitacao": "clima / entrada do modelo",
        "Temp_media": "clima / entrada do modelo",
        "Umidade_rel": "clima / entrada do modelo",
        VARIAVEL_ALVO: "alvo / historico",
        date_column: "indice temporal",
    }

    linhas = []
    for coluna in raw_df.columns:
        serie = raw_df[coluna]
        tipo = str(serie.dtype)
        nulos = int(serie.isna().sum())
        if pd.api.types.is_numeric_dtype(serie):
            minimo = _format_value(serie.min())
            maximo = _format_value(serie.max())
            media = _format_value(serie.mean())
        elif pd.api.types.is_datetime64_any_dtype(serie):
            minimo = _format_value(serie.min())
            maximo = _format_value(serie.max())
            media = ""
        else:
            minimo = ""
            maximo = ""
            media = ""

        linhas.append(
            {
                "coluna": coluna,
                "papel_da_variavel": roles.get(coluna, "entrada do modelo"),
                "tipo_de_dado": tipo,
                "nulos": nulos,
                "minimo": minimo,
                "maximo": maximo,
                "media": media,
            }
        )

    return pd.DataFrame(linhas)


def build_pipeline_config_table(results: dict[str, Any]) -> pd.DataFrame:
    """Resume os principais parametros do pipeline a partir do JSON do treino."""
    config = results.get("config", {})
    raio = int(config.get("raio", 4))
    lag_clima = int(config.get("lag_clima", 45))
    lag_historico = int(config.get("lag_historico", 30))
    janela = 2 * raio + 1

    linhas = [
        ("lag_clima", lag_clima),
        ("lag_historico", lag_historico),
        ("raio", raio),
        ("tamanho_da_janela", janela),
        ("shape_matriz_entrada", f"({janela}, 4)"),
        ("shape_imagem", f"({TAMANHO_IMAGEM}, {TAMANHO_IMAGEM}, 3)"),
        ("backbone", BACKBONE_NOME),
        ("metricas_usadas", "MAE, RMSE, CC"),
        ("baselines_usados", "baseline_media, baseline_historico"),
    ]
    return pd.DataFrame(linhas, columns=["parametro", "valor"])


def build_split_table(results: dict[str, Any]) -> pd.DataFrame:
    """Resume o split temporal armazenado no JSON."""
    split = results.get("split", {})
    linhas = []
    total = int(split["teste"][1])
    for conjunto in ("treino", "validacao", "teste"):
        inicio, fim = split[conjunto]
        quantidade = int(fim - inicio)
        linhas.append(
            {
                "conjunto": conjunto,
                "indice_inicial": int(inicio),
                "indice_final": int(fim),
                "quantidade_amostras": quantidade,
                "porcentagem_do_total": round((quantidade / total) * 100, 2),
            }
        )
    return pd.DataFrame(linhas)


def build_metrics_table(results: dict[str, Any]) -> pd.DataFrame:
    """Monta a tabela de metricas finais do modelo e baselines."""
    metricas = results.get("metricas", {})
    ordem = [
        ("modelo", metricas.get("modelo", {})),
        ("baseline_media", metricas.get("baseline_media", {})),
        (
            "baseline_historico",
            metricas.get("baseline_historico")
            or metricas.get("baseline_ultimo_vizinho", {}),
        ),
    ]
    linhas = []
    for metodo, valores in ordem:
        linhas.append(
            {
                "metodo": metodo,
                "MAE": valores.get("mae", np.nan),
                "RMSE": valores.get("rmse", np.nan),
                "CC": valores.get("cc", np.nan),
            }
        )
    return pd.DataFrame(linhas)


def _save_table(df: pd.DataFrame, path: Path) -> Path:
    """Salva um DataFrame em CSV."""
    df.to_csv(path, index=False)
    return path


def _save_figure(fig: plt.Figure, path: Path, dpi: int) -> Path:
    """Salva uma figura em PNG com qualidade alta."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_case_series(
    raw_df: pd.DataFrame, output_path: Path, date_column: str = "Data", dpi: int = DEFAULT_DPI
) -> Path:
    """Grafico da serie temporal de `Qtde_Casos`."""
    eixo = _time_axis(raw_df, date_column=date_column)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(eixo, raw_df[VARIAVEL_ALVO], color="#1f77b4", linewidth=1.5)
    ax.set_title("Serie temporal dos casos de dengue")
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Qtde_Casos")
    ax.grid(True, alpha=0.25)
    return _save_figure(fig, output_path, dpi)


def plot_original_variables(
    raw_df: pd.DataFrame, output_path: Path, date_column: str = "Data", dpi: int = DEFAULT_DPI
) -> Path:
    """Grafico das variaveis originais em subplots."""
    eixo = _time_axis(raw_df, date_column=date_column)
    cols = ["Precipitacao", "Temp_media", "Umidade_rel", VARIAVEL_ALVO]
    titulos = ["Precipitacao", "Temp_media", "Umidade_rel", "Qtde_Casos"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    cores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for ax, coluna, titulo, cor in zip(axes, cols, titulos, cores):
        ax.plot(eixo, raw_df[coluna], color=cor, linewidth=1.1)
        ax.set_title(titulo)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Tempo")
    fig.supylabel("Valor")
    fig.suptitle("Variaveis originais ao longo do tempo")
    return _save_figure(fig, output_path, dpi)


def _test_series_xy(results: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds = results.get("predicoes_teste", {})
    y_true = np.asarray(preds.get("y_true", []), dtype=float)
    y_model = np.asarray(preds.get("y_pred_modelo", []), dtype=float)
    y_media = np.asarray(preds.get("y_pred_baseline_media", []), dtype=float)
    y_hist = np.asarray(
        preds.get("y_pred_baseline_historico")
        or preds.get("y_pred_baseline_ultimo_vizinho", []),
        dtype=float,
    )
    return y_true, y_model, y_media, y_hist


def plot_real_vs_pred(
    results: dict[str, Any], output_path: Path, dpi: int = DEFAULT_DPI
) -> Path:
    """Grafico principal comparando real e previsto pelo modelo."""
    y_true, y_model, *_ = _test_series_xy(results)
    x = np.arange(len(y_true))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, y_true, label="Real", color="#1f77b4", linewidth=1.5)
    ax.plot(x, y_model, label="Previsto pelo modelo", color="#d62728", linewidth=1.5)
    ax.set_title("Real x previsto no conjunto de teste")
    ax.set_xlabel("Amostra do teste")
    ax.set_ylabel("Qtde_Casos")
    ax.legend()
    ax.grid(True, alpha=0.25)
    return _save_figure(fig, output_path, dpi)


def plot_model_vs_baselines(
    results: dict[str, Any], output_path: Path, dpi: int = DEFAULT_DPI
) -> Path:
    """Compara modelo e baselines no teste."""
    y_true, y_model, y_media, y_hist = _test_series_xy(results)
    x = np.arange(len(y_true))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, y_true, label="Real", color="#1f77b4", linewidth=1.8)
    ax.plot(x, y_model, label="Modelo", color="#d62728", linewidth=1.4)
    ax.plot(x, y_media, label="Baseline média", color="#2ca02c", linestyle="--")
    ax.plot(
        x,
        y_hist,
        label="Baseline histórico",
        color="#ff7f0e",
        linestyle=":",
    )
    ax.set_title("Comparação entre modelo e baselines no teste")
    ax.set_xlabel("Amostra do teste")
    ax.set_ylabel("Qtde_Casos")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.25)
    return _save_figure(fig, output_path, dpi)


def plot_metrics_comparison(
    metrics_df: pd.DataFrame, output_path: Path, dpi: int = DEFAULT_DPI
) -> Path:
    """Grafico de barras com MAE, RMSE e CC."""
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
    return _save_figure(fig, output_path, dpi)


def plot_dispersao_real_previsto(
    results: dict[str, Any], output_path: Path, dpi: int = DEFAULT_DPI
) -> Path:
    """Scatter de real contra previsto com linha ideal."""
    y_true, y_model, *_ = _test_series_xy(results)
    minimo = float(min(y_true.min(), y_model.min()))
    maximo = float(max(y_true.max(), y_model.max()))
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_model, alpha=0.75, color="#1f77b4", edgecolor="white")
    ax.plot([minimo, maximo], [minimo, maximo], color="#d62728", linestyle="--")
    ax.set_title("Dispersao: real x previsto")
    ax.set_xlabel("Real")
    ax.set_ylabel("Previsto pelo modelo")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(minimo, maximo)
    ax.set_ylim(minimo, maximo)
    return _save_figure(fig, output_path, dpi)


def plot_residuals(
    results: dict[str, Any], output_path: Path, dpi: int = DEFAULT_DPI
) -> Path:
    """Histograma e boxplot dos residuos do modelo."""
    y_true, y_model, *_ = _test_series_xy(results)
    residuos = y_true - y_model
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(residuos, bins=15, color="#9467bd", alpha=0.85)
    axes[0].set_title("Histograma dos residuos")
    axes[0].set_xlabel("Residuo")
    axes[0].set_ylabel("Frequencia")
    axes[0].grid(True, alpha=0.25)

    axes[1].boxplot(residuos, vert=True, patch_artist=True)
    axes[1].set_title("Boxplot dos residuos")
    axes[1].set_ylabel("Residuo")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle("Analise dos residuos do modelo")
    return _save_figure(fig, output_path, dpi)


def _classificar_faixas(series: pd.Series) -> tuple[pd.Series, list[str]]:
    """Cria faixas ordenadas de incidencia com fallback robusto."""
    valores = pd.Series(series, dtype=float)
    labels_base = ["baixa", "media", "alta"]

    if valores.nunique(dropna=True) <= 1:
        faixa = pd.Series(["unica"] * len(valores), index=valores.index)
        return faixa, ["unica"]

    try:
        cat = pd.qcut(valores, q=3, duplicates="drop")
    except ValueError:
        cat = None

    if cat is not None and len(cat.cat.categories) >= 1:
        categorias = list(cat.cat.categories)
        labels = labels_base[: len(categorias)]
        if len(categorias) == len(labels):
            mapeamento = {cat_: label for cat_, label in zip(categorias, labels)}
            faixa = cat.map(mapeamento).astype(str)
            faixa = pd.Categorical(faixa, categories=labels, ordered=True)
            return pd.Series(faixa, index=valores.index), labels

    bins = np.linspace(valores.min(), valores.max(), 4)
    bins = np.unique(bins)
    if len(bins) <= 2:
        faixa = pd.Series(["unica"] * len(valores), index=valores.index)
        return faixa, ["unica"]

    cat = pd.cut(valores, bins=bins, include_lowest=True)
    categorias = list(cat.cat.categories)
    labels = labels_base[: len(categorias)]
    mapeamento = {cat_: label for cat_, label in zip(categorias, labels)}
    faixa = cat.map(mapeamento).astype(str)
    faixa = pd.Categorical(faixa, categories=labels, ordered=True)
    return pd.Series(faixa, index=valores.index), labels


def build_error_by_range_table(results: dict[str, Any]) -> pd.DataFrame:
    """Calcula erros por faixa de incidencia do `y_true`."""
    y_true, y_model, *_ = _test_series_xy(results)
    faixa, ordem = _classificar_faixas(pd.Series(y_true))
    df = pd.DataFrame({"faixa": faixa, "y_true": y_true, "y_pred": y_model})

    linhas = []
    for nome_faixa in ordem:
        subset = df[df["faixa"] == nome_faixa]
        if subset.empty:
            continue
        erros = subset["y_true"].to_numpy() - subset["y_pred"].to_numpy()
        linhas.append(
            {
                "faixa": nome_faixa,
                "quantidade_amostras": int(len(subset)),
                "MAE": float(np.mean(np.abs(erros))),
                "RMSE": float(np.sqrt(np.mean(erros**2))),
            }
        )
    return pd.DataFrame(linhas)


def plot_error_by_range(
    error_df: pd.DataFrame, output_path: Path, dpi: int = DEFAULT_DPI
) -> Path:
    """Grafico de erro por faixa de incidencia."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if error_df.empty:
        for ax in axes:
            ax.axis("off")
        return _save_figure(fig, output_path, dpi)

    for ax, metrica in zip(axes, ["MAE", "RMSE"]):
        ax.bar(error_df["faixa"], error_df[metrica], color="#1f77b4")
        ax.set_title(metrica)
        ax.set_xlabel("Faixa de incidencia")
        ax.set_ylabel(metrica)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Erro do modelo por faixa de incidencia")
    return _save_figure(fig, output_path, dpi)


def save_all_report_artifacts(
    csv_path: str | Path,
    results_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    date_column: str = "Data",
    dpi: int = DEFAULT_DPI,
) -> dict[str, Path]:
    """Gera todas as tabelas e graficos do relatorio."""
    output = _ensure_output_dir(output_dir)
    raw_df = _load_raw_dataset(csv_path, date_column=date_column)
    results = load_training_results(results_path)

    artefatos: dict[str, Path] = {}

    estrutura_df = build_data_structure_table(raw_df, date_column=date_column)
    artefatos["tabela_estrutura_dados"] = _save_table(
        estrutura_df, output / "tabela_estrutura_dados.csv"
    )

    config_df = build_pipeline_config_table(results)
    artefatos["tabela_configuracao_pipeline"] = _save_table(
        config_df, output / "tabela_configuracao_pipeline.csv"
    )

    split_df = build_split_table(results)
    artefatos["tabela_split_temporal"] = _save_table(
        split_df, output / "tabela_split_temporal.csv"
    )

    metrics_df = build_metrics_table(results)
    artefatos["tabela_metricas_modelos"] = _save_table(
        metrics_df, output / "tabela_metricas_modelos.csv"
    )

    error_df = build_error_by_range_table(results)
    artefatos["tabela_erro_por_faixa"] = _save_table(
        error_df, output / "tabela_erro_por_faixa.csv"
    )

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
        help="Pasta de saida dos artefatos.",
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
