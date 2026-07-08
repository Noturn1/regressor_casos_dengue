import numpy as np
import pytest

pytest.importorskip("tensorflow")  # requer o extra `dl` (rodar no venv)

from dengue_tl.models.cnn2d import build_cnn2d, prepara_entrada


def test_smoke_saida_batch():
    model = build_cnn2d(input_shape=(4, 9, 1))
    lote = np.random.rand(4, 4, 9, 1).astype("float32")

    saida = model.predict(lote, verbose=0)

    assert saida.shape == (4, 1)


def test_saida_e_linear():
    # Regressão: última camada sem ativação (não limitar a saída).
    model = build_cnn2d(input_shape=(4, 9, 1))

    assert model.layers[-1].activation.__name__ == "linear"


def test_sem_pooling():
    # Entrada 4x9 já é minúscula: a spec não tem pooling nem stride > 1.
    model = build_cnn2d(input_shape=(4, 9, 1))

    nomes = [type(camada).__name__ for camada in model.layers]
    assert not any("Pooling" in nome for nome in nomes)


def test_prepara_entrada_transpoe_para_features_x_dias():
    # Janela (n, 9, 4) [dias x features] -> entrada (n, 4, 9, 1) [features x dias].
    X = np.arange(2 * 9 * 4, dtype=float).reshape(2, 9, 4)

    entrada = prepara_entrada(X)

    assert entrada.shape == (2, 4, 9, 1)
    assert entrada.dtype == np.float32
    # O valor do dia d / feature f precisa ser o mesmo, só reposicionado.
    assert entrada[1, 3, 5, 0] == X[1, 5, 3]
