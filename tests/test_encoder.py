import numpy as np
import pytest

pytest.importorskip("pyts")  # encoder requer o extra `dl`

from dengue_tl.encoder import TAMANHO_IMAGEM, encode_gasf, encode_matrix


def _janela_exemplo():
    # 8 vizinhos (janela padrão, dia central excluído)
    return np.array([0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0])


def test_shape_100x100x3():
    img = encode_gasf(_janela_exemplo())

    assert img.shape == (TAMANHO_IMAGEM, TAMANHO_IMAGEM, 3)


def test_tres_canais_identicos():
    img = encode_gasf(_janela_exemplo())

    np.testing.assert_array_equal(img[:, :, 0], img[:, :, 1])
    np.testing.assert_array_equal(img[:, :, 0], img[:, :, 2])


def test_matriz_simetrica():
    # Propriedade da GASF; o resize bilinear preserva a simetria.
    img = encode_gasf(_janela_exemplo())

    np.testing.assert_allclose(img[:, :, 0], img[:, :, 0].T, atol=1e-5)


def test_determinismo():
    janela = _janela_exemplo()

    np.testing.assert_array_equal(encode_gasf(janela), encode_gasf(janela.copy()))


def test_range_da_gasf():
    # GASF (summation) vive em [-1, 1]; resize bilinear não extrapola.
    img = encode_gasf(_janela_exemplo())

    assert img.min() >= -1.0 - 1e-6
    assert img.max() <= 1.0 + 1e-6


def test_janela_constante_nao_quebra():
    # pyts reescala por-amostra; uma janela constante é um caso de borda.
    img = encode_gasf(np.full(8, 3.0))

    assert img.shape == (TAMANHO_IMAGEM, TAMANHO_IMAGEM, 3)
    assert np.isfinite(img).all()


# --- encode_matrix (entrada 9x4 lagged) ---


def _matriz_9x4():
    rng = np.random.default_rng(0)
    return rng.random((9, 4))  # já em [0,1]


def test_matrix_shape_100x100x3():
    img = encode_matrix(_matriz_9x4())

    assert img.shape == (TAMANHO_IMAGEM, TAMANHO_IMAGEM, 3)


def test_matrix_tres_canais_identicos():
    img = encode_matrix(_matriz_9x4())

    np.testing.assert_array_equal(img[:, :, 0], img[:, :, 1])
    np.testing.assert_array_equal(img[:, :, 0], img[:, :, 2])


def test_matrix_mapeia_0_1_para_menos1_mais1():
    np.testing.assert_allclose(encode_matrix(np.zeros((9, 4))), -1.0, atol=1e-6)
    np.testing.assert_allclose(encode_matrix(np.ones((9, 4))), 1.0, atol=1e-6)


def test_matrix_determinismo():
    m = _matriz_9x4()

    np.testing.assert_array_equal(encode_matrix(m), encode_matrix(m.copy()))
