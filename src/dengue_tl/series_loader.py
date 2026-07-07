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


def _reconstroi_numero_corrompido(bruto: str, digitos_inteiros: int) -> float:
    """Reconstrói um float exportado com `.` de milhar e o decimal perdido.

    Ex.: `'1.595.629.411.764.700'` com `digitos_inteiros=2` -> `15.9563`.
    Tira os pontos e reposiciona o decimal para a parte inteira ter
    `digitos_inteiros` casas — o número de casas é calibrado pela magnitude
    dos valores limpos da própria coluna (ver `repara_valores_numericos`).
    """
    negativo = bruto.strip().startswith("-")
    digitos = bruto.replace(".", "").replace("-", "").lstrip("0") or "0"
    valor = int(digitos) / 10 ** (len(digitos) - digitos_inteiros)
    return -valor if negativo else valor


def repara_valores_numericos(df: pd.DataFrame, colunas=VARIAVEIS) -> pd.DataFrame:
    """Repara colunas numéricas sujas por separador de milhar `.` (formato BR).

    Alguns blocos do CSV bruto vêm com floats em precisão total exportados com
    `.` de milhar e sem o separador decimal (ex.: `1.595.629.411.764.700` em vez
    de `15.9563`). `pd.read_csv` lê esses valores como string e o pipeline quebra
    ao converter para float. Aqui, para cada coluna numérica esperada:

    - valores já numéricos passam direto;
    - valores corrompidos (string que não converte) são reconstruídos posicionando
      o decimal pela magnitude dos valores limpos da coluna (defesa em profundidade
      — repara o dado conhecido sem hardcodar a escala de cada variável).

    Opera sobre uma cópia; não muta o DataFrame recebido.
    """
    df = df.copy()
    for coluna in colunas:
        if coluna not in df.columns:
            continue
        numerica = pd.to_numeric(df[coluna], errors="coerce")
        corrompidos = numerica.isna() & df[coluna].notna()
        if not corrompidos.any():
            df[coluna] = numerica
            continue
        validos = numerica.dropna()
        if validos.empty:
            continue  # sem referência limpa: não há como calibrar o decimal
        digitos_inteiros = max(1, len(str(int(abs(validos.median())))))
        numerica.loc[corrompidos] = df.loc[corrompidos, coluna].map(
            lambda bruto: _reconstroi_numero_corrompido(str(bruto), digitos_inteiros)
        )
        df[coluna] = numerica
    return df


def load(csv_path, date_column="Data"):
    df = pd.read_csv(csv_path, parse_dates=[date_column])
    df = df.sort_values(date_column).set_index(date_column)
    _valida_contiguidade_diaria(df.index)
    return repara_valores_numericos(df[VARIAVEIS])


def _valida_contiguidade_diaria(indice):
    diffs = indice.to_series().diff().dropna()
    vaos = diffs[diffs != pd.Timedelta(days=1)]
    if not vaos.empty:
        primeiro = vaos.index[0]
        raise SeriesGapError(
            f"Vão temporal na série: {primeiro.date()} não está a 1 dia do dia anterior "
            f"(diferença de {vaos.iloc[0].days} dias)."
        )
