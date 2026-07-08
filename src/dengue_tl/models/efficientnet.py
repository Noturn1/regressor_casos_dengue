"""EfficientNet-B0 (transfer learning) sobre a janela 9x4 codificada em imagem.

Ver `roteiro.md`, Etapa 5. Backbone ImageNet congelado -> GlobalAveragePooling2D
-> Dropout -> Dense(1) linear. Fine-tuning em 2 fases (congela / descongela os
últimos blocos). Cuidado com normalização dupla (roteiro §7).

Esta arquitetura **exige** codificar a janela 9x4 em imagem 100x100x3 (a rede tem
downsampling ÷32 e não roda com entrada minúscula) — `prepara_entrada` faz isso via
`encode_matrix`. Requer o extra `dl` (tensorflow). Rodar no venv — o TensorFlow
segfalta no Python 3.13 do anaconda base; num venv isolado importa normalmente.
"""

from __future__ import annotations

import numpy as np
import keras
from keras import layers
from keras.applications import EfficientNetB0

TAMANHO_IMAGEM = 100
BACKBONE_NOME = "efficientnetb0"


def build_efficientnet(
    input_shape=(TAMANHO_IMAGEM, TAMANHO_IMAGEM, 3), weights="imagenet", dropout=0.3
):
    """Retorna um `keras.Model` que mapeia (batch, 100, 100, 3) -> (batch, 1).

    A imagem (encoder) sai em [-1, 1]; a EfficientNet do Keras embute
    Rescaling+Normalization e espera pixels ~[0, 255]. Mapeamos [-1,1] -> [0,255]
    com uma Rescaling e deixamos o backbone normalizar (não normalizar duas
    vezes, roteiro §7). Fase 1: backbone congelado; ver `descongela_backbone`.
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


def prepara_entrada(X_escalado: np.ndarray) -> np.ndarray:
    """Codifica cada janela 9x4 escalada em imagem 100x100x3 (float32)."""
    from dengue_tl.encoder import encode_matrix

    return np.stack([encode_matrix(m) for m in X_escalado], axis=0).astype("float32")


def treina(x_treino, y_treino, x_val, y_val, config):
    """Fine-tuning em 2 fases: cabeça congelada e depois últimos blocos."""
    keras.utils.set_random_seed(config.seed)

    model = build_efficientnet(weights="imagenet", dropout=config.dropout)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate_fase1),
        loss="mse",
        metrics=["mae"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.paciencia_early_stopping,
            restore_best_weights=True,
        )
    ]

    hist_fase1 = model.fit(
        x_treino,
        y_treino,
        validation_data=(x_val, y_val),
        epochs=config.epocas_fase1,
        batch_size=config.batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    model = descongela_backbone(model, n_camadas_finais=config.n_camadas_finais)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate_fase2),
        loss="mse",
        metrics=["mae"],
    )

    hist_fase2 = model.fit(
        x_treino,
        y_treino,
        validation_data=(x_val, y_val),
        epochs=config.epocas_fase2,
        batch_size=config.batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    historico = {
        "fase1": {k: [float(v) for v in vs] for k, vs in hist_fase1.history.items()},
        "fase2": {k: [float(v) for v in vs] for k, vs in hist_fase2.history.items()},
    }
    return model, historico


def espaco_busca(trial) -> dict:
    """Espaço de busca do Optuna. As chaves são campos de `TreinoConfig`."""
    return {
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "learning_rate_fase1": trial.suggest_float(
            "learning_rate_fase1", 1e-4, 1e-2, log=True
        ),
        "learning_rate_fase2": trial.suggest_float(
            "learning_rate_fase2", 1e-6, 1e-3, log=True
        ),
        "n_camadas_finais": trial.suggest_int("n_camadas_finais", 10, 60, step=10),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
    }
