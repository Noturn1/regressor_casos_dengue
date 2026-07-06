import numpy as np
import pandas as pd
import pytest

from dengue_tl.lagged_table import LaggedTableConfig, build_lagged_table


def _dados_exemplo(n=50):
    datas = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "Data": datas,
            "Precipitacao": 1000 + np.arange(n, dtype=float),
            "Temp_media": 2000 + np.arange(n, dtype=float),
            "Umidade_rel": 3000 + np.arange(n, dtype=float),
            "Qtde_Casos": 4000 + np.arange(n, dtype=float),
        }
    )


def test_build_lagged_table_cria_colunas_esperadas():
    tabela = build_lagged_table(_dados_exemplo())

    assert list(tabela.columns) == [
        "Precipitacao_lag45",
        "Temp_media_lag45",
        "Umidade_rel_lag45",
        "Historico_lag30",
        "Qtde_Casos",
    ]


def test_build_lagged_table_descarta_linhas_sem_contexto():
    tabela = build_lagged_table(_dados_exemplo())

    assert len(tabela) == 5
    assert tabela.index[0] == pd.Timestamp("2020-02-15")


def test_build_lagged_table_usa_45_dias_para_clima_e_30_para_historico():
    tabela = build_lagged_table(_dados_exemplo())

    primeira = tabela.iloc[0]
    assert primeira["Precipitacao_lag45"] == 1000.0
    assert primeira["Temp_media_lag45"] == 2000.0
    assert primeira["Umidade_rel_lag45"] == 3000.0
    assert primeira["Historico_lag30"] == 4015.0
    assert primeira["Qtde_Casos"] == 4045.0


def test_build_lagged_table_permite_custom_lags():
    tabela = build_lagged_table(
        _dados_exemplo(),
        LaggedTableConfig(lag_clima=10, lag_historico=5),
    )

    assert list(tabela.columns) == [
        "Precipitacao_lag10",
        "Temp_media_lag10",
        "Umidade_rel_lag10",
        "Historico_lag5",
        "Qtde_Casos",
    ]
    assert len(tabela) == 40


def test_build_lagged_table_erro_com_colunas_ausentes():
    dados = pd.DataFrame(
        {
            "Data": pd.date_range("2020-01-01", periods=50, freq="D"),
            "Precipitacao": np.arange(50, dtype=float),
            "Qtde_Casos": np.arange(50, dtype=float),
        }
    )

    with pytest.raises(ValueError, match="colunas obrigatorias ausentes"):
        build_lagged_table(dados)
