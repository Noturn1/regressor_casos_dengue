"""Construcao de tabela supervisionada com lags por variavel.

Este modulo transforma a serie diaria original em uma tabela tabular em que:

- clima entra com atraso de 45 dias
- historico de casos entra com atraso de 30 dias
- o alvo permanece como `Qtde_Casos` do dia atual

A ideia e materializar, em colunas explicitas, a hipotese epidemiologica de
que o clima atua mais tarde e os casos anteriores ajudam a explicar a
transmissao subsequente.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dengue_tl.series_loader import SeriesGapError, load

VARIAVEIS_CLIMATICAS = ("Precipitacao", "Temp_media", "Umidade_rel")
VARIAVEL_ALVO = "Qtde_Casos"


@dataclass(frozen=True)
class LaggedTableConfig:
    lag_clima: int = 45
    lag_historico: int = 30
    date_column: str = "Data"


def _valida_contiguidade_diaria(indice: pd.Index) -> None:
    """Reaplica a regra de serie diaria contigua usada pelo loader."""
    if not isinstance(indice, pd.DatetimeIndex):
        return

    diffs = indice.to_series().diff().dropna()
    vaos = diffs[diffs != pd.Timedelta(days=1)]
    if not vaos.empty:
        primeiro = vaos.index[0]
        raise SeriesGapError(
            f"Vao temporal na serie: {primeiro.date()} nao esta a 1 dia do dia "
            f"anterior (diferenca de {vaos.iloc[0].days} dias)."
        )


def _normaliza_entrada(dados, date_column: str) -> pd.DataFrame:
    """Padroniza a entrada como DataFrame ordenado com index temporal, quando houver."""
    if isinstance(dados, (str, Path)):
        caminho = Path(dados)
        cabecalho = pd.read_csv(caminho, nrows=0).columns.tolist()
        if date_column in cabecalho:
            return load(caminho, date_column=date_column)
        return pd.read_csv(caminho)

    if isinstance(dados, pd.DataFrame):
        base = dados.copy()
        if date_column in base.columns:
            base[date_column] = pd.to_datetime(base[date_column])
            base = base.sort_values(date_column).set_index(date_column)
            _valida_contiguidade_diaria(base.index)
            return base
        if isinstance(base.index, pd.DatetimeIndex):
            base = base.sort_index()
            _valida_contiguidade_diaria(base.index)
            return base
        return base

    raise TypeError(
        "`dados` deve ser um caminho de CSV ou um pandas.DataFrame."
    )


def build_lagged_table(
    dados,
    config: LaggedTableConfig = LaggedTableConfig(),
) -> pd.DataFrame:
    """Gera a tabela supervisionada com lags separados por tipo de variavel.

    A saida contem:
      - `Precipitacao_lag{lag_clima}`
      - `Temp_media_lag{lag_clima}`
      - `Umidade_rel_lag{lag_clima}`
      - `Historico_lag{lag_historico}`
      - `Qtde_Casos`

    As linhas iniciais sem contexto suficiente sao removidas com `dropna()`.
    """
    base = _normaliza_entrada(dados, date_column=config.date_column)

    faltantes = [
        coluna
        for coluna in (*VARIAVEIS_CLIMATICAS, VARIAVEL_ALVO)
        if coluna not in base.columns
    ]
    if faltantes:
        raise ValueError(
            "colunas obrigatorias ausentes para construir a tabela com lags: "
            + ", ".join(faltantes)
        )

    base = base.loc[:, [*VARIAVEIS_CLIMATICAS, VARIAVEL_ALVO]]

    clima = (
        base[list(VARIAVEIS_CLIMATICAS)]
        .shift(config.lag_clima)
        .add_suffix(f"_lag{config.lag_clima}")
    )
    historico = base[[VARIAVEL_ALVO]].shift(config.lag_historico).rename(
        columns={VARIAVEL_ALVO: f"Historico_lag{config.lag_historico}"}
    )
    alvo = base[[VARIAVEL_ALVO]]

    tabela = pd.concat([clima, historico, alvo], axis=1).dropna().copy()
    tabela.index.name = base.index.name
    return tabela
