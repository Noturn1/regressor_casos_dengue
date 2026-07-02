"""Codifica uma janela 1-D em imagem GASF 100x100x3 para a EfficientNet.

Ver `roteiro.md`, Etapa 4. Usa `pyts.image.GramianAngularField(method="summation")`,
redimensiona L×L -> 100×100 e replica em 3 canais RGB. A `pyts` reescala cada
janela internamente (por-amostra, não é vazamento).

STUB: implementar na Etapa 4 do roteiro, via TDD. Requer o extra `dl` (pyts, pillow).
"""

TAMANHO_IMAGEM = 100


def encode_gasf(janela):
    """`janela`: array 1-D. Retorna imagem `np.ndarray` de shape (100, 100, 3).

    Propriedades esperadas (escreva os testes primeiro):
      - shape (100, 100, 3); os 3 canais idênticos
      - matriz simétrica (propriedade da GASF)
      - determinismo: mesma entrada -> mesma imagem
    """
    raise NotImplementedError("Implementar na Etapa 4 do roteiro (TDD).")
