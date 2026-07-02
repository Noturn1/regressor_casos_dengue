"""Carrega a série diária de Cascavel a partir de um CSV.

A base completa é diária, contígua e possui coluna de data. O loader lê essa
coluna e valida a contiguidade diária como defesa em profundidade — um vão falha
alto, não silenciosamente. A amostra de desenvolvimento (`data/AmostraDados.csv`)
não tem coluna de data; nesse caso use o índice sequencial (ver README).

Reaproveitado de uma Iniciação Científica, sem alterações de lógica.
"""

import pandas as pd

VARIAVEIS = ["Precipitacao", "Temp_media", "Umidade_rel", "Qtde_Casos"]


class SeriesGapError(Exception):
    """Levantada quando dias consecutivos não diferem por exatamente 1 dia."""


def load(csv_path, date_column="Data"):
    df = pd.read_csv(csv_path, parse_dates=[date_column])
    df = df.sort_values(date_column).set_index(date_column)
    _valida_contiguidade_diaria(df.index)
    return df[VARIAVEIS]


def _valida_contiguidade_diaria(indice):
    diffs = indice.to_series().diff().dropna()
    vaos = diffs[diffs != pd.Timedelta(days=1)]
    if not vaos.empty:
        primeiro = vaos.index[0]
        raise SeriesGapError(
            f"Vão temporal na série: {primeiro.date()} não está a 1 dia do dia anterior "
            f"(diferença de {vaos.iloc[0].days} dias)."
        )
