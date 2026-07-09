"""MLP denso sobre as 4 features defasadas do dia t — o baseline mais simples.

Diferente da cnn2d (que enxerga a matriz 9×4 inteira) e da cnn_lstm (que convolve
no tempo sobre a janela `(9, 4)`), este baseline ignora completamente a janela de
vizinhos: usa APENAS as 4 features defasadas do dia central `t`
(`Precipitacao_lag45`, `Temp_media_lag45`, `Umidade_rel_lag45`, `Historico_lag30`).

O pipeline continua entregando a janela `(n, 9, 4)`; `prepara_entrada` colapsa
para a linha central `(n, 4)`. Assim o MLP usa exatamente o mesmo conjunto de
amostras e o mesmo split que a cnn2d/cnn_lstm (comparação justa), sem precisar
rodar com `--raio 0`.

É o contraste natural: se a janela ajuda, cnn2d/cnn_lstm devem bater este MLP.
Rede minúscula, treina em segundos na CPU. Requer o extra `dl` (tensorflow).
"""

from __future__ import annotations

import keras
import numpy as np
from keras import layers


def build_mlp(input_shape=(4,), unidades=64, dropout=0.3):
    """Mapeia `(batch, 4)` -> `(batch, 1)`.

    Duas camadas densas ReLU (cada uma seguida de Dropout se `dropout > 0`) e
    uma Dense(1) linear para a regressão.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(2):
        x = layers.Dense(unidades, activation="relu")(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs, name="dengue_mlp")


def prepara_entrada(X_escalado: np.ndarray) -> np.ndarray:
    """Janela `(n, 9, 4)` -> `(n, 4)`: só a linha central (o dia t)."""
    X = np.asarray(X_escalado, dtype="float32")
    centro = X.shape[1] // 2  # == raio; o dia t (defasagens t-45/t-30)
    return X[:, centro, :]  # (n, 4)


def treina(x_treino, y_treino, x_val, y_val, config, sample_weight=None):
    """Treino em fase única (rede pequena), como a cnn2d.

    Usa `epocas` como orçamento total e o `learning_rate` do config;
    `EarlyStopping` restaura os melhores pesos.
    `sample_weight` (opcional) pondera cada dia de treino na loss — ver
    `train_runner.pesos_por_nivel` (peso de pico). A validação fica sem peso.
    """
    keras.utils.set_random_seed(config.seed)

    model = build_mlp(
        input_shape=x_treino.shape[1:],
        unidades=config.dense_unidades,
        dropout=config.dropout,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
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
        epochs=config.epocas,
        batch_size=config.batch_size,
        sample_weight=sample_weight,
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
        "dense_unidades": trial.suggest_categorical("dense_unidades", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-4, 1e-2, log=True
        ),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        # Peso de pico: 0 (sem peso) até forte. A busca decide se ponderar os
        # dias de alto numero de casos ajuda — ver train_runner.pesos_por_nivel.
        "peso_pico": trial.suggest_float("peso_pico", 0.0, 8.0),
    }
