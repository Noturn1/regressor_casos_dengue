import numpy as np
import pytest

pytest.importorskip("tensorflow")  # modelo requer o extra `dl` (rodar no venv)

from keras.layers import BatchNormalization

from dengue_tl.model import BACKBONE_NOME, TAMANHO_IMAGEM, build_model, descongela_backbone


def _modelo_leve():
    # weights=None evita baixar os pesos ImageNet: smoke test rápido e offline.
    return build_model(weights=None)


def test_smoke_saida_batch_1():
    model = _modelo_leve()
    lote = np.random.rand(4, TAMANHO_IMAGEM, TAMANHO_IMAGEM, 3).astype("float32")

    saida = model.predict(lote, verbose=0)

    assert saida.shape == (4, 1)


def test_backbone_congelado_por_padrao():
    model = _modelo_leve()

    # Fase 1: só a cabeça Dense(1) treina -> kernel + bias = 2 variáveis.
    assert len(model.trainable_variables) == 2


def test_saida_e_linear():
    # Regressão: a última camada não pode ter ativação (não limitar a [0,1]).
    model = _modelo_leve()

    assert model.layers[-1].activation.__name__ == "linear"


def test_descongela_backbone_habilita_mais_treino():
    model = _modelo_leve()
    antes = len(model.trainable_variables)

    descongela_backbone(model, n_camadas_finais=20)

    assert len(model.trainable_variables) > antes


def test_descongela_mantem_batchnorm_congelada():
    # Boa prática de fine-tuning: BatchNorm fica congelada mesmo na Fase 2.
    model = _modelo_leve()

    descongela_backbone(model, n_camadas_finais=20)

    backbone = model.get_layer(BACKBONE_NOME)
    bns = [c.trainable for c in backbone.layers if isinstance(c, BatchNormalization)]
    assert not any(bns)
