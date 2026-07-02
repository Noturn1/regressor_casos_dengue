import pandas as pd
import pytest

from dengue_tl.series_loader import load, SeriesGapError, VARIAVEIS


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
