"""Modelos do pipeline, um por arquivo.

Cada módulo de arquitetura expõe a mesma interface, o que mantém o
`train_runner` agnóstico e torna trivial adicionar novas redes (basta um arquivo
novo que implemente):

- ``prepara_entrada(X_escalado) -> np.ndarray``: adapta a janela 9x4 ao formato
  que a rede espera (imagem, sequência, tabela...);
- ``treina(x_treino, y_treino, x_val, y_val, config) -> (model, historico)``:
  constrói, treina e devolve o modelo Keras e o histórico de treino
  (dict ``{fase: {metrica: [valores por época]}}``).

`seleciona_arquitetura` faz import preguiçoso do módulo — importar este pacote
não puxa TensorFlow; só a arquitetura escolhida carrega Keras.
"""

from __future__ import annotations

ARQUITETURAS = ("cnn_lstm", "efficientnet")


def seleciona_arquitetura(nome: str):
    """Retorna o módulo da arquitetura pedida (import preguiçoso)."""
    if nome == "cnn_lstm":
        from dengue_tl.models import cnn_lstm

        return cnn_lstm
    if nome == "efficientnet":
        from dengue_tl.models import efficientnet

        return efficientnet
    raise ValueError(
        f"Arquitetura desconhecida: {nome!r}. Opções: {', '.join(ARQUITETURAS)}."
    )
