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


def test_carrega_tabela_lagged_dateless_e_cacheia(tmp_path):
    csv = _csv_dateless(tmp_path, n=60)
    cache = tmp_path / "cache" / "tabela.csv"
    config = TreinoConfig(csv_path=str(csv), cache_path=str(cache))

    tabela = carrega_tabela_lagged(config)

    assert cache.exists()  # materializou o cache
    assert list(tabela.columns) == [
        "Precipitacao_lag45",
        "Temp_media_lag45",
        "Umidade_rel_lag45",
        "Historico_lag30",
        "Qtde_Casos",
    ]


def test_treina_e_avalia_erro_quando_amostra_insuficiente(tmp_path):
    # 10 linhas com lag_clima=45 -> tabela lagged vazia -> sem amostras.
    csv = _csv_dateless(tmp_path, n=10)
    cache = tmp_path / "cache.csv"

    with pytest.raises(ValueError, match="Amostras insuficientes"):
        treina_e_avalia(TreinoConfig(csv_path=str(csv), cache_path=str(cache)))
