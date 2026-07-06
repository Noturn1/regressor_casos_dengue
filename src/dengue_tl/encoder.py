"""Codifica a entrada da EfficientNet em imagem 100x100x3.

Dois codificadores:
- `encode_gasf(janela_1d)`: GASF (univariado) — abordagem original do roteiro.
- `encode_matrix(matriz_9x4)`: empilha a matriz lagged (9 dias × 4 features) como
  imagem — abordagem atual (ver `matrix_windower`).

Ambos redimensionam para 100×100 e replicam em 3 canais RGB. Requer o extra
`dl` (pyts, scipy).
"""

import numpy as np
from pyts.image import GramianAngularField
from scipy.ndimage import zoom

TAMANHO_IMAGEM = 100

# pyts reescala cada janela internamente (por-amostra) — não é vazamento (roteiro §7).
_gaf = GramianAngularField(method="summation")


def encode_gasf(janela):
    """`janela`: array 1-D. Retorna imagem `np.ndarray` de shape (100, 100, 3).

    Propriedades esperadas (escreva os testes primeiro):
      - shape (100, 100, 3); os 3 canais idênticos
      - matriz simétrica (propriedade da GASF)
      - determinismo: mesma entrada -> mesma imagem
    """
    janela = np.asarray(janela, dtype=float).reshape(1, -1)
    matriz = _gaf.fit_transform(janela)[0]  # (L, L) em [-1, 1]

    # Resize L×L -> 100×100 com interpolação linear (order=1). Preserva a
    # simetria e não extrapola além de [-1, 1] (ao contrário do resize float do PIL).
    fator = TAMANHO_IMAGEM / matriz.shape[0]
    canal = zoom(matriz, fator, order=1).astype(np.float32)

    # Replica em 3 canais RGB (a EfficientNet espera 3 canais).
    return np.stack([canal, canal, canal], axis=-1)


def encode_matrix(matriz):
    """`matriz`: array 2-D (9×4) já escalado em ~[0,1] (por-feature, no treino).

    Retorna imagem `(100, 100, 3)` em ~[-1,1]. Passos: resize 9×4 → 100×100
    (linear) e mapeia [0,1] → [-1,1]; o `Rescaling` do modelo leva a [0,255]
    (§7 — normalização única). Não reescala aqui: manter a escala do `Scaler`
    (fitado só no treino) preserva a validade temporal.

    Propriedades esperadas:
      - shape (100, 100, 3); os 3 canais idênticos
      - matriz de 0s → imagem de -1; matriz de 1s → imagem de +1
      - determinismo: mesma entrada → mesma imagem
    """
    m = np.asarray(matriz, dtype=float)
    fatores = (TAMANHO_IMAGEM / m.shape[0], TAMANHO_IMAGEM / m.shape[1])
    canal = zoom(m, fatores, order=1)  # (100, 100), ainda em ~[0,1]
    canal = (canal * 2.0 - 1.0).astype(np.float32)  # [0,1] -> [-1,1]

    return np.stack([canal, canal, canal], axis=-1)
