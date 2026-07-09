import numpy as np
import pytest

pytest.importorskip("tensorflow")  # requer o extra `dl` (rodar no venv)

from dengue_tl.models.mlp import build_mlp, prepara_entrada, treina


def test_smoke_saida_batch():
    model = build_mlp(input_shape=(4,))
    lote = np.random.rand(4, 4).astype("float32")

    saida = model.predict(lote, verbose=0)

    assert saida.shape == (4, 1)


def test_saida_e_linear():
    # Regressão: última camada sem ativação (não limitar a saída).
    model = build_mlp(input_shape=(4,))

    assert model.layers[-1].activation.__name__ == "linear"


def test_prepara_entrada_colapsa_linha_central():
    # Janela (n, 9, 4) -> entrada (n, 4): só a linha central (dia t).
    X = np.arange(2 * 9 * 4, dtype=float).reshape(2, 9, 4)

    entrada = prepara_entrada(X)

    assert entrada.shape == (2, 4)
    assert entrada.dtype == np.float32
    # Linha central é a de índice 9 // 2 == 4.
    assert np.array_equal(entrada[1], X[1, 4, :].astype("float32"))


def test_treina_smoke_retorna_modelo_e_historico():
    from dataclasses import dataclass

    @dataclass
    class ConfigStub:
        seed = 0
        dense_unidades = 8
        dropout = 0.0
        learning_rate = 1e-3
        paciencia_early_stopping = 2
        epocas = 2
        batch_size = 4

    x = np.random.rand(16, 4).astype("float32")
    y = np.random.rand(16).astype("float32")

    model, historico = treina(x, y, x, y, ConfigStub())

    assert model is not None
    assert "treino" in historico
    assert model.predict(x, verbose=0).shape == (16, 1)
