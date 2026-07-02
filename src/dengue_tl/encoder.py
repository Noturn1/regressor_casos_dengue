"""Codifica uma janela 1-D em imagem GASF 100x100x3 para a EfficientNet.

Ver `roteiro.md`, Etapa 4. Usa `pyts.image.GramianAngularField(method="summation")`,
redimensiona L×L -> 100×100 e replica em 3 canais RGB. A `pyts` reescala cada
janela internamente (por-amostra, não é vazamento).

STUB: implementar na Etapa 4 do roteiro, via TDD. Requer o extra `dl` (pyts, pillow).
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
