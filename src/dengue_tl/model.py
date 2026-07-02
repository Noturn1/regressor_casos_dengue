"""Modelo EfficientNet-B0 (transfer learning) para regressão de casos.

Ver `roteiro.md`, Etapa 5. Backbone ImageNet congelado -> GlobalAveragePooling2D
-> Dropout -> Dense(1) linear. Fine-tuning em 2 fases (congela / descongela os
últimos blocos). Cuidado com normalização dupla (roteiro §7).

Requer o extra `dl` (tensorflow). Rodar no venv — o TensorFlow segfalta no
Python 3.13 do anaconda base; num venv isolado importa normalmente.
"""

import keras
from keras import layers
from keras.applications import EfficientNetB0

TAMANHO_IMAGEM = 100
BACKBONE_NOME = "efficientnetb0"


def build_model(input_shape=(TAMANHO_IMAGEM, TAMANHO_IMAGEM, 3), weights="imagenet", dropout=0.3):
    """Retorna um `keras.Model` que mapeia (batch, 100, 100, 3) -> (batch, 1).

    A GASF (encoder) sai em [-1, 1]; a EfficientNet do Keras embute
    Rescaling+Normalization e espera pixels ~[0, 255]. Mapeamos [-1,1] -> [0,255]
    com uma Rescaling e deixamos o backbone normalizar (não normalizar duas
    vezes, roteiro §7). Fase 1: backbone congelado; ver `descongela_backbone`.

    Smoke test: um lote aleatório produz saída de shape (batch, 1).
    """
    backbone = EfficientNetB0(
        include_top=False, weights=weights, input_shape=input_shape, name=BACKBONE_NOME
    )
    backbone.trainable = False  # Fase 1: treina só a cabeça.

    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(scale=127.5, offset=127.5)(inputs)  # [-1,1] -> [0,255]
    x = backbone(x, training=False)  # BN em modo inferência enquanto congelado
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)  # regressão: saída linear, sem ativação
    return keras.Model(inputs, outputs, name="dengue_efficientnet")


def descongela_backbone(model, n_camadas_finais=20):
    """Fase 2 do fine-tuning: descongela os últimos blocos do backbone.

    As camadas iniciais (features genéricas) ficam congeladas; só as
    `n_camadas_finais` treinam. BatchNorm permanece congelada — atualizar suas
    estatísticas com lotes pequenos degrada o transfer learning.
    """
    backbone = model.get_layer(BACKBONE_NOME)
    backbone.trainable = True
    for camada in backbone.layers[:-n_camadas_finais]:
        camada.trainable = False
    for camada in backbone.layers:
        if isinstance(camada, layers.BatchNormalization):
            camada.trainable = False
    return model
