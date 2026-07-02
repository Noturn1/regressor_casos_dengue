"""Modelo EfficientNet-B0 (transfer learning) para regressão de casos.

Ver `roteiro.md`, Etapa 5. Backbone ImageNet congelado -> GlobalAveragePooling2D
-> Dropout -> Dense(1) linear. Fine-tuning em 2 fases (congela / descongela os
últimos blocos). Cuidado com normalização dupla (roteiro §7).

STUB: implementar na Etapa 5 do roteiro. Requer o extra `dl` (tensorflow).
"""

TAMANHO_IMAGEM = 100


def build_model(input_shape=(TAMANHO_IMAGEM, TAMANHO_IMAGEM, 3)):
    """Retorna um `tf.keras.Model` que mapeia (batch, 100, 100, 3) -> (batch, 1).

    Smoke test esperado: um lote aleatório produz saída de shape (batch, 1).
    """
    raise NotImplementedError("Implementar na Etapa 5 do roteiro.")
