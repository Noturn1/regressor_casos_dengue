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
    raise NotImplementedError("Implementar na Etapa 2 do roteiro (TDD).")
