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

import numpy as np
import pandas as pd

from dengue_tl.series_loader import SeriesGapError, load, repara_valores_numericos

VARIAVEIS_CLIMATICAS = ("Precipitacao", "Temp_media", "Umidade_rel")
VARIAVEL_ALVO = "Qtde_Casos"


@dataclass(frozen=True)
class LaggedTableConfig:
    lag_clima: int = 45
    lag_historico: int = 30
    date_column: str = "Data"
    # Sazonalidade: adiciona `sin_ano`/`cos_ano` do dia t (o dia-alvo) como
    # features. NAO e vazamento — o calendario e conhecido de antemao para
    # qualquer data (feature exogena de futuro conhecido), diferente das
    # variaveis de caso/clima, que precisam de lag. Desligado por padrao aqui
    # (mantem o contrato basico da tabela); o experimento liga via TreinoConfig.
    sazonalidade: bool = False
    # Data do primeiro registro, usada so quando a serie e dateless (sem coluna
    # 'Data'): o dataset completo comeca em 2007-01-01. Com indice datetime, as
    # datas reais mandam e este campo e ignorado.
    data_inicial: str = "2007-01-01"


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
        return repara_valores_numericos(pd.read_csv(caminho))

    if isinstance(dados, pd.DataFrame):
        base = repara_valores_numericos(dados)
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


def _features_sazonais(base: pd.DataFrame, data_inicial: str) -> pd.DataFrame:
    """Codifica o dia-do-ano de cada linha como par seno/cosseno.

    O par (periodo 365.25) evita a descontinuidade 31/dez -> 1/jan que uma
    codificacao linear teria. Usa o indice datetime quando existe; caso
    contrario, reconstroi as datas posicionalmente a partir de `data_inicial`
    (a serie completa e diaria e contigua).
    """
    if isinstance(base.index, pd.DatetimeIndex):
        datas = base.index
    else:
        datas = pd.date_range(data_inicial, periods=len(base), freq="D")

    angulo = 2.0 * np.pi * datas.dayofyear.to_numpy(dtype=float) / 365.25
    return pd.DataFrame(
        {"sin_ano": np.sin(angulo), "cos_ano": np.cos(angulo)},
        index=base.index,
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

    # As colunas sazonais entram entre as features e o alvo: sem lag (calendario
    # do proprio dia) e sem NaN, entao o dropna abaixo so corta o aquecimento
    # dos lags, preservando o seno/cosseno correto de cada linha remanescente.
    partes = [clima, historico]
    if config.sazonalidade:
        partes.append(_features_sazonais(base, config.data_inicial))
    partes.append(alvo)

    tabela = pd.concat(partes, axis=1).dropna().copy()
    tabela.index.name = base.index.name
    return tabela


def build_or_load_lagged_table(
    dados,
    cache_path,
    config: LaggedTableConfig = LaggedTableConfig(),
) -> pd.DataFrame:
    """Materializa a tabela com lags em disco (cache).

    - Se `cache_path` existe, carrega a tabela pronta (evita recomputar os
      lags a cada execução).
    - Se não existe, constrói via `build_lagged_table`, salva em `cache_path`
      e retorna. Cria a pasta do cache se necessário.

    A tabela é salva sem índice: como a série completa é dateless (índice
    posicional), a ordem das linhas — preservada pelo CSV — é o que importa
    para o janelamento posterior.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_csv(cache_path)

    tabela = build_lagged_table(dados, config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(cache_path, index=False)
    return tabela
