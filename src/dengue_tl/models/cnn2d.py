"""CNN 2D pura sobre a janela como dado estruturado 4x9x1 (sem imagem, sem RNN).

Arquitetura recomendada pelo professor:

    Entrada (4, 9, 1)  — features x dias, 1 canal
    Conv2D(32, (2,2), padding='same') + ReLU
    Conv2D(64, (2,2), padding='same') + ReLU
    Flatten
    Dense(64) + ReLU
    Dense(1) linear (regressao)

Sem pooling: a entrada ja e minuscula (4x9), downsampling jogaria estrutura
fora. Os kernels 2x2 cruzam features vizinhas E dias vizinhos ao mesmo tempo —
diferente da CNN-LSTM, que convolve so no tempo. A janela `(n, 9, 4)` do
janelador e transposta para `(n, 4, 9, 1)` em `prepara_entrada`.

Rede pequena (dezenas de milhares de parametros), treina em segundos na CPU.
Requer o extra `dl` (tensorflow).
"""

from __future__ import annotations

import keras
import numpy as np
from keras import layers


def build_cnn2d(input_shape=(4, 9, 1), filtros=32, dense_unidades=64, dropout=0.0):
    """Mapeia `(batch, 4, 9, 1)` -> `(batch, 1)`.

    `filtros` e a primeira Conv2D; a segunda dobra (32 -> 64 na spec).
    A spec do professor nao tem dropout; o knob existe porque o config
    compartilhado (`TreinoConfig.dropout`) e a busca de hiperparametros o
    controlam — para reproduzir a spec exata, treinar com `--dropout 0`.
    """
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(filtros, (2, 2), padding="same", activation="relu")(inputs)
    x = layers.Conv2D(filtros * 2, (2, 2), padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(dense_unidades, activation="relu")(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs, name="dengue_cnn2d")


def prepara_entrada(X_escalado: np.ndarray) -> np.ndarray:
    """Janela `(n, 9, 4)` -> `(n, 4, 9, 1)`: features nas linhas, dias nas colunas."""
    X = np.asarray(X_escalado, dtype="float32")
    return np.transpose(X, (0, 2, 1))[..., np.newaxis]


def treina(x_treino, y_treino, x_val, y_val, config):
    """Treino em fase unica (rede pequena), como a CNN-LSTM.

    Usa `epocas_fase1 + epocas_fase2` como orcamento total de epocas e o
    `learning_rate_fase1` do config; `EarlyStopping` restaura os melhores pesos.
    """
    keras.utils.set_random_seed(config.seed)

    model = build_cnn2d(
        input_shape=x_treino.shape[1:],
        filtros=config.filtros,
        dense_unidades=config.dense_unidades,
        dropout=config.dropout,
    )
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

    hist = model.fit(
        x_treino,
        y_treino,
        validation_data=(x_val, y_val),
        epochs=config.epocas_fase1 + config.epocas_fase2,
        batch_size=config.batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    historico = {
        "treino": {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    }
    return model, historico


def espaco_busca(trial) -> dict:
    """Espaço de busca do Optuna. As chaves são campos de `TreinoConfig`."""
    return {
        "filtros": trial.suggest_categorical("filtros", [16, 32, 64]),
        "dense_unidades": trial.suggest_categorical("dense_unidades", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "learning_rate_fase1": trial.suggest_float(
            "learning_rate_fase1", 1e-4, 1e-2, log=True
        ),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }
