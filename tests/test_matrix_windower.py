import numpy as np
import pandas as pd

from dengue_tl.matrix_windower import (
    MatrixWindowConfig,
    build_matrix_windows,
    feature_columns,
)


def _tabela_lagged(n=20):
    # Simula a saída de build_lagged_table: 4 features + alvo.
    return pd.DataFrame(
        {
            "Precipitacao_lag45": np.arange(n, dtype=float),
            "Temp_media_lag45": 100 + np.arange(n, dtype=float),
            "Umidade_rel_lag45": 200 + np.arange(n, dtype=float),
            "Historico_lag30": 300 + np.arange(n, dtype=float),
            "Qtde_Casos": 400 + np.arange(n, dtype=float),
        }
    )


def test_feature_columns_exclui_o_alvo():
    assert feature_columns(_tabela_lagged()) == [
        "Precipitacao_lag45",
        "Temp_media_lag45",
        "Umidade_rel_lag45",
        "Historico_lag30",
    ]


def test_shape_9x4_por_amostra():
    X, y = build_matrix_windows(_tabela_lagged(n=20))

    assert X.shape == (20 - 2 * 4, 9, 4)  # (12, 9, 4)
    assert y.shape == (12,)


def test_numero_de_amostras_descarta_bordas():
    X, y = build_matrix_windows(_tabela_lagged(n=15))

    assert len(X) == 15 - 2 * 4  # 7
    assert len(y) == len(X)


def test_alvo_e_o_dia_central():
    X, y = build_matrix_windows(_tabela_lagged(n=20))

    # Primeiro centro é t=4 -> Qtde_Casos = 400 + 4.
    assert y[0] == 404.0
    np.testing.assert_array_equal(y, 400 + np.arange(4, 16))


def test_janela_contem_9_linhas_das_features_centradas():
    X, _ = build_matrix_windows(_tabela_lagged(n=20))

    # Centro t=4 -> linhas 0..8 da coluna Precipitacao_lag45 (== índice).
    np.testing.assert_array_equal(X[0, :, 0], np.arange(0, 9))
    # Coluna Historico_lag30 começa em 300.
    np.testing.assert_array_equal(X[0, :, 3], 300 + np.arange(0, 9))


def test_raio_customizado():
    X, y = build_matrix_windows(_tabela_lagged(n=20), MatrixWindowConfig(raio=2))

    assert X.shape == (20 - 2 * 2, 5, 4)  # janela de 5 linhas


def test_tabela_curta_nao_gera_amostras():
    X, y = build_matrix_windows(_tabela_lagged(n=8))  # < 2*4+1

    assert len(X) == 0
    assert len(y) == 0
