import numpy as np
import pytest

from dengue_tl.window_builder import JanelaCentradaConfig, build_centrado


def test_numero_de_amostras_descarta_as_bordas():
    # N=10, raio=4 -> só os dias t=4 e t=5 têm contexto completo dos dois lados.
    casos = np.arange(10)

    janelas, alvos = build_centrado(casos)

    assert len(janelas) == len(casos) - 2 * 4  # == 2
    assert len(alvos) == len(janelas)


def test_excluir_dia_central_gera_janela_dos_8_vizinhos():
    # Default: incluir_dia_central=False. O dia central NÃO entra na janela.
    casos = np.arange(10)  # [0,1,...,9]

    janelas, alvos = build_centrado(casos)

    # Primeiro centro é t=4: vizinhos ±4 sem o próprio 4.
    np.testing.assert_array_equal(janelas[0], [0, 1, 2, 3, 5, 6, 7, 8])
    assert alvos[0] == 4
    # Segundo centro é t=5.
    np.testing.assert_array_equal(janelas[1], [1, 2, 3, 4, 6, 7, 8, 9])
    assert alvos[1] == 5


def test_janela_excluindo_central_tem_comprimento_2_raio():
    casos = np.arange(10)

    janelas, _ = build_centrado(casos)

    assert janelas.shape[1] == 2 * 4  # 8 vizinhos


def test_incluir_dia_central_coloca_o_alvo_na_janela():
    casos = np.arange(10)
    config = JanelaCentradaConfig(incluir_dia_central=True)

    janelas, alvos = build_centrado(casos, config)

    # t=4: janela contígua 0..8, com o próprio 4 no meio.
    np.testing.assert_array_equal(janelas[0], [0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert alvos[0] == 4
    assert janelas.shape[1] == 2 * 4 + 1  # 9 dias


def test_alvos_sao_o_dia_central_correspondente():
    casos = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    _, alvos = build_centrado(casos)

    # Centros t=4 e t=5 -> casos 50 e 60.
    np.testing.assert_array_equal(alvos, [50, 60])


def test_raio_customizado():
    casos = np.array([10, 20, 30, 40, 50])
    config = JanelaCentradaConfig(raio=1)  # 1 dia de cada lado

    janelas, alvos = build_centrado(casos, config)

    assert len(janelas) == len(casos) - 2 * 1  # 3 amostras (t=1,2,3)
    np.testing.assert_array_equal(janelas[0], [10, 30])  # sem o central (20)
    np.testing.assert_array_equal(alvos, [20, 30, 40])


def test_serie_curta_demais_nao_gera_amostras():
    casos = np.arange(8)  # < 2*raio+1 == 9

    janelas, alvos = build_centrado(casos)

    assert len(janelas) == 0
    assert len(alvos) == 0
