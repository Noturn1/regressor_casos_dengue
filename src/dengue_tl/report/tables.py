"""Tabelas do relatório construídas a partir do CSV bruto e do JSON do treino."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dengue_tl.lagged_table import VARIAVEL_ALVO, VARIAVEIS_CLIMATICAS
from dengue_tl.report.data_io import extract_test_predictions


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
    """Resume colunas, papel, tipo e estatísticas básicas dos dados brutos."""
    roles = {
        **{coluna: "clima / entrada do modelo" for coluna in VARIAVEIS_CLIMATICAS},
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
    """Resume os principais parâmetros do pipeline a partir do JSON do treino."""
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
    """Monta a tabela de métricas finais do modelo e baselines."""
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


def classificar_faixas(series: pd.Series) -> tuple[pd.Series, list[str]]:
    """Cria faixas ordenadas de incidência com fallback robusto."""
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
    """Calcula erros por faixa de incidência do `y_true`."""
    preds = extract_test_predictions(results)
    faixa, ordem = classificar_faixas(pd.Series(preds.y_true))
    df = pd.DataFrame({"faixa": faixa, "y_true": preds.y_true, "y_pred": preds.y_model})

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


def save_table(df: pd.DataFrame, path: Path) -> Path:
    """Salva um DataFrame em CSV."""
    df.to_csv(path, index=False)
    return path
