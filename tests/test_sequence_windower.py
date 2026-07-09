import numpy as np
import pandas as pd
import pytest

from dengue_tl.sequence_windower import (
    SequenceWindowConfig,
    build_sequence_windows,
)


def _serie(n=200):
    # Series distintas por coluna p/ checar posicionamento; casos = indice.
    return pd.DataFrame(
        {
            "Precipitacao": np.arange(n, dtype=float),
            "Temp_media": 1000 + np.arange(n, dtype=float),
            "Umidade_rel": 2000 + np.arange(n, dtype=float),
            "Qtde_Casos": np.arange(n, dtype=float),
        }
    )


def test_shapes_e_numero_de_amostras():
    X, y, hist = build_sequence_windows(
        _serie(200), SequenceWindowConfig(janela_dias=60, gap_dias=30)
    )
    # t valido a partir de gap+janela-1 = 89 -> n = 200 - 89
    assert X.shape == (200 - 89, 60, 4)
    assert y.shape == (111,)
    assert hist.shape == (111,)


def test_alvo_e_o_dia_t():
    X, y, _ = build_sequence_windows(
        _serie(200), SequenceWindowConfig(janela_dias=60, gap_dias=30)
    )
    # Primeiro alvo em t=89 (Qtde_Casos == indice).
    assert y[0] == 89.0
    np.testing.assert_array_equal(y, np.arange(89, 200))


def test_janela_termina_em_t_menos_gap_sem_vazamento():
    N, gap = 60, 30
    X, y, hist = build_sequence_windows(
        _serie(200), SequenceWindowConfig(janela_dias=N, gap_dias=gap)
    )
    # Para a 1a amostra (t=89): janela de casos = [t-gap-N+1 .. t-gap] = [0..59].
    casos_janela = X[0, :, 3]  # coluna Qtde_Casos
    np.testing.assert_array_equal(casos_janela, np.arange(0, 60))
    # Ultimo dia da janela == t-gap == hist; o alvo (t=89) NAO esta na janela.
    assert casos_janela[-1] == 59.0
    assert hist[0] == 59.0
    assert y[0] == 89.0
    assert y[0] not in set(casos_janela)  # sem vazamento do alvo


def test_gap_maior_afasta_a_janela():
    _, _, hist = build_sequence_windows(
        _serie(200), SequenceWindowConfig(janela_dias=10, gap_dias=45)
    )
    # 1a amostra: t = 45+10-1 = 54; hist = casos[t-45] = casos[9] = 9.
    assert hist[0] == 9.0


def test_valida_parametros():
    with pytest.raises(ValueError):
        build_sequence_windows(_serie(50), SequenceWindowConfig(janela_dias=0, gap_dias=30))


def test_erro_com_coluna_ausente():
    ruim = _serie(100).drop(columns=["Umidade_rel"])
    with pytest.raises(ValueError, match="colunas obrigatorias ausentes"):
        build_sequence_windows(ruim)
