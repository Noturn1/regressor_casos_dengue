import numpy as np
import pandas as pd
import pytest

from dengue_tl.series_loader import (
    load,
    repara_valores_numericos,
    SeriesGapError,
    VARIAVEIS,
)


def _escreve_csv(caminho, datas, precipitacao, temp, umidade, casos):
    pd.DataFrame(
        {
            "Data": datas,
            "Precipitacao": precipitacao,
            "Temp_media": temp,
            "Umidade_rel": umidade,
            "Qtde_Casos": casos,
        }
    ).to_csv(caminho, index=False)


def test_load_retorna_serie_diaria_com_quatro_variaveis(tmp_path):
    csv = tmp_path / "serie.csv"
    _escreve_csv(
        csv,
        datas=["2007-01-01", "2007-01-02", "2007-01-03"],
        precipitacao=[17, 0, 12],
        temp=[23.72, 23.506, 22.956],
        umidade=[78.016, 78.595, 69.431],
        casos=[2, 8, 4],
    )

    serie = load(csv)

    assert list(serie.columns) == VARIAVEIS
    assert isinstance(serie.index, pd.DatetimeIndex)
    assert len(serie) == 3
    assert serie.loc["2007-01-02", "Qtde_Casos"] == 8


def test_load_levanta_erro_em_vao_temporal(tmp_path):
    csv = tmp_path / "serie_com_vao.csv"
    _escreve_csv(
        csv,
        datas=["2007-01-01", "2007-01-02", "2007-01-05"],  # pula 03 e 04
        precipitacao=[17, 0, 12],
        temp=[23.72, 23.506, 22.956],
        umidade=[78.016, 78.595, 69.431],
        casos=[2, 8, 4],
    )

    with pytest.raises(SeriesGapError):
        load(csv)


def test_repara_reconstroi_float_com_separador_de_milhar():
    # Bloco real do CSV: temp/umidade em precisão total com '.' de milhar e
    # decimal perdido, misturados com valores limpos.
    df = pd.DataFrame(
        {
            "Temp_media": ["15.51", "1.595.629.411.764.700", "16.008.529.411.764.700"],
            "Umidade_rel": ["85.909", "7.762.223.529.411.760", "77"],
        }
    )

    reparado = repara_valores_numericos(df, colunas=["Temp_media", "Umidade_rel"])

    assert reparado["Temp_media"].tolist() == pytest.approx([15.51, 15.9563, 16.0085], abs=1e-3)
    assert reparado["Umidade_rel"].tolist() == pytest.approx([85.909, 77.6222, 77.0], abs=1e-3)
    assert reparado["Temp_media"].dtype == float


def test_repara_nao_altera_coluna_limpa():
    df = pd.DataFrame({"Temp_media": [23.72, 23.506], "Umidade_rel": [78.016, 69.431]})

    reparado = repara_valores_numericos(df, colunas=["Temp_media", "Umidade_rel"])

    assert reparado["Temp_media"].tolist() == [23.72, 23.506]
    # Não muta o original.
    assert df["Temp_media"].tolist() == [23.72, 23.506]


def test_repara_calibra_decimal_pela_magnitude_da_coluna():
    # Coluna com parte inteira de 3 dígitos (ex.: precipitação alta) calibra o
    # decimal para 3 casas, não 2 — não hardcoda a escala.
    df = pd.DataFrame({"Precipitacao": ["120.5", "150.0", "1.234.567.890"]})

    reparado = repara_valores_numericos(df, colunas=["Precipitacao"])

    assert reparado["Precipitacao"].iloc[2] == pytest.approx(123.4567890, abs=1e-4)
