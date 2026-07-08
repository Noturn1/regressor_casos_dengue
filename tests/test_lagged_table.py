import numpy as np
import pandas as pd
import pytest

from dengue_tl.lagged_table import (
    LaggedTableConfig,
    build_lagged_table,
    build_or_load_lagged_table,
)


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


def test_build_lagged_table_funciona_sem_coluna_de_data():
    # Dataset real de Cascavel não tem coluna 'Data': os lags devem operar
    # posicionalmente (shift por linha), preservando a ordem do arquivo.
    sem_data = _dados_exemplo().drop(columns=["Data"])

    tabela = build_lagged_table(sem_data)

    assert list(tabela.columns) == [
        "Precipitacao_lag45",
        "Temp_media_lag45",
        "Umidade_rel_lag45",
        "Historico_lag30",
        "Qtde_Casos",
    ]
    assert len(tabela) == 5
    # Mesmos valores defasados do caso com data, mas com índice posicional.
    primeira = tabela.iloc[0]
    assert primeira["Precipitacao_lag45"] == 1000.0
    assert primeira["Historico_lag30"] == 4015.0
    assert primeira["Qtde_Casos"] == 4045.0


def test_sazonalidade_adiciona_sin_cos_antes_do_alvo():
    tabela = build_lagged_table(
        _dados_exemplo(), LaggedTableConfig(sazonalidade=True)
    )

    assert list(tabela.columns) == [
        "Precipitacao_lag45",
        "Temp_media_lag45",
        "Umidade_rel_lag45",
        "Historico_lag30",
        "sin_ano",
        "cos_ano",
        "Qtde_Casos",
    ]
    # Par unitario: sin^2 + cos^2 == 1 em toda linha.
    np.testing.assert_allclose(
        tabela["sin_ano"] ** 2 + tabela["cos_ano"] ** 2, 1.0
    )


def test_sazonalidade_usa_o_calendario_da_linha():
    # Primeira linha remanescente e 2020-02-15 (dia-do-ano 46).
    tabela = build_lagged_table(
        _dados_exemplo(), LaggedTableConfig(sazonalidade=True)
    )

    angulo = 2.0 * np.pi * 46 / 365.25
    assert tabela.iloc[0]["sin_ano"] == pytest.approx(np.sin(angulo))
    assert tabela.iloc[0]["cos_ano"] == pytest.approx(np.cos(angulo))


def test_sazonalidade_sem_data_reconstroi_a_partir_de_data_inicial():
    # Serie dateless: as datas vem de data_inicial + posicao. Comeca em
    # 2007-01-01, entao a 1a linha remanescente (posicao 45) e 2007-02-15.
    sem_data = _dados_exemplo().drop(columns=["Data"])

    tabela = build_lagged_table(
        sem_data, LaggedTableConfig(sazonalidade=True, data_inicial="2007-01-01")
    )

    dia_do_ano = pd.Timestamp("2007-02-15").dayofyear  # 46
    angulo = 2.0 * np.pi * dia_do_ano / 365.25
    assert tabela.iloc[0]["sin_ano"] == pytest.approx(np.sin(angulo))


def test_build_or_load_cria_cache_quando_inexistente(tmp_path):
    cache = tmp_path / "sub" / "tabela_lagged.csv"
    assert not cache.exists()

    tabela = build_or_load_lagged_table(_dados_exemplo().drop(columns=["Data"]), cache)

    assert cache.exists()  # salvou em disco (criando a pasta)
    recarregada = pd.read_csv(cache)
    pd.testing.assert_frame_equal(
        tabela.reset_index(drop=True), recarregada.reset_index(drop=True)
    )


def test_build_or_load_usa_cache_existente_sem_reconstruir(tmp_path):
    cache = tmp_path / "tabela_lagged.csv"
    # Cache pré-existente com conteúdo arbitrário: a função deve devolvê-lo
    # como está, sem reconstruir a partir de `dados`.
    marcador = pd.DataFrame({"Qtde_Casos": [1, 2, 3]})
    marcador.to_csv(cache, index=False)

    tabela = build_or_load_lagged_table("caminho/que/seria/ignorado.csv", cache)

    pd.testing.assert_frame_equal(tabela, marcador)


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
