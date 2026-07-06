import numpy as np
import pandas as pd
import pytest

from dengue_tl.train_runner import (
    TreinoConfig,
    baseline_ultimo_vizinho,
    calcula_metricas,
    carrega_serie_casos,
    treina_e_avalia,
    split_temporal,
)


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


def test_baseline_ultimo_vizinho_usa_t_menos_1():
    # Com raio=4 e sem dia central, a janela e:
    # [t-4, t-3, t-2, t-1, t+1, t+2, t+3, t+4]
    janelas = np.array([[10, 11, 12, 13, 15, 16, 17, 18]], dtype=float)

    pred = baseline_ultimo_vizinho(janelas, raio=4, incluir_dia_central=False)

    np.testing.assert_array_equal(pred, [13.0])


def test_calcula_metricas():
    y_true = np.array([1.0, 2.0, 3.0], dtype=float)
    y_pred = np.array([1.0, 3.0, 2.0], dtype=float)

    metricas = calcula_metricas(y_true, y_pred)

    assert metricas["mae"] == pytest.approx(2 / 3)
    assert metricas["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert metricas["cc"] == pytest.approx(0.5)


def test_carrega_serie_casos_fallback_sem_data(tmp_path):
    csv = tmp_path / "amostra.csv"
    pd.DataFrame(
        {
            "Precipitacao": [1.0, 2.0, 3.0],
            "Temp_media": [20.0, 21.0, 22.0],
            "Umidade_rel": [70.0, 71.0, 72.0],
            "Qtde_Casos": [5.0, 6.0, 7.0],
        }
    ).to_csv(csv, index=False)

    casos = carrega_serie_casos(str(csv), date_column="Data")

    np.testing.assert_array_equal(casos, [5.0, 6.0, 7.0])


def test_treina_e_avalia_erro_com_sugestao_quando_amostra_insuficiente(tmp_path):
    csv = tmp_path / "amostra.csv"
    pd.DataFrame(
        {
            "Precipitacao": np.arange(10, dtype=float),
            "Temp_media": np.arange(10, dtype=float),
            "Umidade_rel": np.arange(10, dtype=float),
            "Qtde_Casos": np.arange(10, dtype=float),
        }
    ).to_csv(csv, index=False)

    with pytest.raises(ValueError, match=r"--raio <= 3"):
        treina_e_avalia(
            TreinoConfig(
                csv_path=str(csv),
                raio=4,
            )
        )
