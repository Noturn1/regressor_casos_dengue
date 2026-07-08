import numpy as np
import pandas as pd
import pytest

from dengue_tl.train_runner import (
    SplitConfig,
    TreinoConfig,
    baseline_historico,
    calcula_metricas,
    carrega_tabela_lagged,
    treina_e_avalia,
    split_temporal,
)


def _csv_dateless(tmp_path, n):
    csv = tmp_path / "dados.csv"
    pd.DataFrame(
        {
            "Precipitacao": np.arange(n, dtype=float),
            "Temp_media": 20 + np.arange(n, dtype=float),
            "Umidade_rel": 70 + np.arange(n, dtype=float),
            "Qtde_Casos": np.arange(n, dtype=float),
        }
    ).to_csv(csv, index=False)
    return csv


def test_split_temporal_preserva_ordem():
    treino_sl, val_sl, teste_sl = split_temporal(
        n_amostras=100, treino_fracao=0.7, validacao_fracao=0.15
    )

    assert treino_sl == slice(0, 70)
    assert val_sl == slice(70, 85)
    assert teste_sl == slice(85, 100)


def test_split_temporal_erro_quando_inviavel():
    with pytest.raises(ValueError):
        split_temporal(n_amostras=3, treino_fracao=0.9, validacao_fracao=0.09)


def test_baseline_historico_usa_dia_central():
    # X: (n, 9, 4). Coluna 3 = Historico_lag30; linha central (raio=4) e o alvo.
    X = np.zeros((2, 9, 4), dtype=float)
    X[0, 4, 3] = 13.0  # historico do dia central da amostra 0
    X[1, 4, 3] = 27.0

    pred = baseline_historico(X, raio=4, idx_historico=3)

    np.testing.assert_array_equal(pred, [13.0, 27.0])


def test_calcula_metricas():
    y_true = np.array([1.0, 2.0, 3.0], dtype=float)
    y_pred = np.array([1.0, 3.0, 2.0], dtype=float)

    metricas = calcula_metricas(y_true, y_pred)

    assert metricas["mae"] == pytest.approx(2 / 3)
    assert metricas["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert metricas["cc"] == pytest.approx(0.5)


def test_carrega_tabela_lagged_dateless_sem_sazonalidade(tmp_path):
    csv = _csv_dateless(tmp_path, n=60)
    cache = tmp_path / "cache" / "tabela.csv"
    config = TreinoConfig(
        csv_path=str(csv), cache_path=str(cache), sazonalidade=False
    )

    tabela = carrega_tabela_lagged(config)

    assert cache.exists()  # materializou o cache no caminho pedido
    assert list(tabela.columns) == [
        "Precipitacao_lag45",
        "Temp_media_lag45",
        "Umidade_rel_lag45",
        "Historico_lag30",
        "Qtde_Casos",
    ]


def test_carrega_tabela_lagged_sazonal_usa_cache_sufixado(tmp_path):
    # Com sazonalidade (default), o cache ganha sufixo _sazonal para nao reusar
    # silenciosamente uma tabela de 4 features, e as colunas ganham sin/cos.
    csv = _csv_dateless(tmp_path, n=60)
    cache = tmp_path / "cache" / "tabela.csv"
    config = TreinoConfig(csv_path=str(csv), cache_path=str(cache))

    tabela = carrega_tabela_lagged(config)

    assert not cache.exists()
    assert (tmp_path / "cache" / "tabela_sazonal.csv").exists()
    assert list(tabela.columns) == [
        "Precipitacao_lag45",
        "Temp_media_lag45",
        "Umidade_rel_lag45",
        "Historico_lag30",
        "sin_ano",
        "cos_ano",
        "Qtde_Casos",
    ]


def test_treina_e_avalia_erro_quando_amostra_insuficiente(tmp_path):
    # 10 linhas com lag_clima=45 -> tabela lagged vazia -> sem amostras.
    csv = _csv_dateless(tmp_path, n=10)
    cache = tmp_path / "cache.csv"

    with pytest.raises(ValueError, match="Amostras insuficientes"):
        treina_e_avalia(TreinoConfig(csv_path=str(csv), cache_path=str(cache)))


def test_preve_casos_limita_explosao_do_expm1():
    from dengue_tl.train_runner import TransformadorAlvo, preve_casos

    class ModeloExplosivo:
        def predict(self, x, verbose=0):
            # log 40 viraria ~2e17 casos no expm1; -3 viraria contagem negativa.
            return np.array([[40.0], [-3.0], [2.0]])

    transformador = TransformadorAlvo(alvo="nivel", teto_log=np.log1p(100.0))
    pred = preve_casos(
        ModeloExplosivo(), x=None, transformador=transformador, historico=None
    )

    np.testing.assert_allclose(pred, [100.0, 0.0, np.expm1(2.0)], rtol=1e-6)


def test_transformador_razao_ida_e_volta():
    from dengue_tl.train_runner import TransformadorAlvo

    transformador = TransformadorAlvo(alvo="razao", teto_log=np.log1p(1e6))
    casos = np.array([0.0, 30.0, 600.0])
    historico = np.array([5.0, 30.0, 200.0])

    y_modelo = transformador.transforma(casos, historico)
    recuperado = transformador.inverte(y_modelo, historico)

    np.testing.assert_allclose(recuperado, casos, rtol=1e-9)


def test_transformador_razao_zero_e_o_baseline_de_persistencia():
    # Prever 0 na formulacao razao == repetir o historico: qualquer coisa que a
    # rede aprender e ganho sobre o baseline.
    from dengue_tl.train_runner import TransformadorAlvo

    transformador = TransformadorAlvo(alvo="razao", teto_log=np.log1p(1e6))
    historico = np.array([0.0, 7.0, 250.0])

    pred = transformador.inverte(np.zeros(3), historico)

    np.testing.assert_allclose(pred, historico, rtol=1e-9)


def test_media_movel_centrada_suaviza_serrilhado():
    from dengue_tl.train_runner import _media_movel_centrada

    serrilhado = np.array([0.0, 14.0, 0.0, 14.0, 0.0, 14.0, 0.0], dtype=float)

    suave = _media_movel_centrada(serrilhado, janela=7)

    assert suave[3] == pytest.approx(6.0)  # media exata da janela cheia
    assert np.ptp(suave) < np.ptp(serrilhado)  # amplitude reduzida
    # janela <= 1 nao altera nada
    np.testing.assert_array_equal(_media_movel_centrada(serrilhado, 1), serrilhado)
