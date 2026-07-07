import numpy as np
import pytest

pytest.importorskip("tensorflow")  # requer o extra `dl` (rodar no venv)

from dengue_tl.models.cnn_lstm import build_cnn_lstm, prepara_entrada


def test_smoke_saida_batch_1():
    model = build_cnn_lstm(input_shape=(9, 4))
    lote = np.random.rand(4, 9, 4).astype("float32")

    saida = model.predict(lote, verbose=0)

    assert saida.shape == (4, 1)


def test_saida_e_linear():
    # Regressão: última camada sem ativação (não limitar a saída).
    model = build_cnn_lstm(input_shape=(9, 4))

    assert model.layers[-1].activation.__name__ == "linear"


def test_prepara_entrada_mantem_shape_da_janela():
    # CNN-LSTM come a janela (n, 9, 4) direto — sem codificar em imagem.
    X = np.zeros((5, 9, 4), dtype=float)

    entrada = prepara_entrada(X)

    assert entrada.shape == (5, 9, 4)
    assert entrada.dtype == np.float32
