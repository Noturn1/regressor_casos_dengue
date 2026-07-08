"""CNN-LSTM: convoluções 1D no tempo + camada recorrente sobre a janela 9x4.

Abordagem "CNN + recorrente" da IC, aplicada diretamente à janela `(9, 4)`
(9 dias × 4 features lagged) — **sem** codificar em imagem e **sem** resize: o
LSTM já espera `(timesteps, features)`, que é exatamente o formato do janelador.
As `Conv1D` extraem padrões locais ao longo dos 9 dias e o `LSTM` modela a
dependência temporal; a cabeça `Dense(1)` faz a regressão.

Rede minúscula (poucos milhares de parâmetros): treina em segundos na CPU.
Requer o extra `dl` (tensorflow).
"""

from __future__ import annotations

import keras
import numpy as np
from keras import layers


def build_cnn_lstm(input_shape=(9, 4), filtros=32, unidades_lstm=32, dropout=0.3):
    """Mapeia `(batch, 9, 4)` -> `(batch, 1)`.

    - `Conv1D` com `padding="same"` preserva os 9 passos (sem downsampling agressivo);
    - `LSTM` colapsa a sequência num vetor;
    - `Dense(1)` linear: regressão, sem ativação de saída.
    """
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(filtros, kernel_size=2, padding="same", activation="relu")(inputs)
    x = layers.Conv1D(filtros, kernel_size=2, padding="same", activation="relu")(x)
    x = layers.LSTM(unidades_lstm)(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs, name="dengue_cnn_lstm")


def prepara_entrada(X_escalado: np.ndarray) -> np.ndarray:
    """A janela `(n, 9, 4)` já é a entrada da rede — só garante `float32`."""
    return np.asarray(X_escalado, dtype="float32")


def treina(x_treino, y_treino, x_val, y_val, config, sample_weight=None):
    """Treino em fase única (a rede é pequena; sem congelar/descongelar).

    Usa `epocas` como orçamento total e o `learning_rate` do config;
    `EarlyStopping` restaura os melhores pesos.
    `sample_weight` (opcional) pondera cada dia de treino na loss — ver
    `train_runner.pesos_por_nivel` (peso de pico). A validação fica sem peso.
    """
    keras.utils.set_random_seed(config.seed)

    model = build_cnn_lstm(
        input_shape=x_treino.shape[1:],
        filtros=config.filtros,
        unidades_lstm=config.unidades_lstm,
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
        "filtros": trial.suggest_categorical("filtros", [16, 32, 64, 128]),
        "unidades_lstm": trial.suggest_categorical("unidades_lstm", [16, 32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-4, 1e-2, log=True
        ),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        # Peso de pico: 0 (sem peso) até forte. A busca decide se ponderar os
        # dias de alto numero de casos ajuda — ver train_runner.pesos_por_nivel.
        "peso_pico": trial.suggest_float("peso_pico", 0.0, 8.0),
    }
