"""Janela centrada de 9 dias para nowcasting do dia central.

Ver `roteiro.md`, Etapa 2. Para cada dia `t` com `raio` dias de contexto de cada
lado, gera (janela, alvo): `alvo = casos[t]`; `janela` = os vizinhos ±raio.

Se `incluir_dia_central=False` (default), o dia central NÃO entra na janela —
evita o vazamento do valor-alvo para a imagem (ver `roteiro.md` §2 e CONTEXT.md).

STUB: implementar na Etapa 2 do roteiro, via TDD.
"""

from dataclasses import dataclass

import numpy as np  # noqa: F401  (usado na implementação)


@dataclass(frozen=True)
class JanelaCentradaConfig:
    raio: int = 4
    incluir_dia_central: bool = False


def build_centrado(casos, config=JanelaCentradaConfig()):
    """`casos`: array 1-D de Qtde_Casos. Retorna `(janelas, alvos)`.

    Propriedades esperadas (escreva os testes primeiro):
      - nº de amostras == len(casos) - 2*raio
      - comprimento da janela == 2*raio+1 (incluir central) ou 2*raio (excluir)
      - alvos[i] == casos do dia central correspondente
      - dias de borda (sem contexto completo) são descartados
    """
    casos = np.asarray(casos)
    raio = config.raio
    largura = 2 * raio + 1

    janelas = []
    alvos = []
    # Só os dias com `raio` de contexto completo dos dois lados viram amostra;
    # as bordas (primeiros/últimos `raio` dias) são descartadas.
    for t in range(raio, len(casos) - raio):
        vizinhanca = casos[t - raio : t + raio + 1]  # 2*raio+1 dias, centro em raio
        if config.incluir_dia_central:
            janela = vizinhanca
        else:
            # Remove o dia central para não vazar o alvo na imagem (CONTEXT.md §Vazamento).
            janela = np.concatenate([vizinhanca[:raio], vizinhanca[raio + 1 :]])
        janelas.append(janela)
        alvos.append(casos[t])

    comprimento = largura if config.incluir_dia_central else largura - 1
    janelas = np.array(janelas).reshape(-1, comprimento)
    alvos = np.array(alvos)
    return janelas, alvos
