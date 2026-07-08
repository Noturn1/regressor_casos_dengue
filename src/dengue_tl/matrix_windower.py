"""Janelas 9x4 a partir da tabela lagged (entrada das redes cnn_lstm / cnn2d).

Ver `roteiro.md`. A tabela de `lagged_table` já alinha, em cada linha `t`:
clima em `t-45`, histórico de casos em `t-30` e o alvo `Qtde_Casos[t]`. Aqui
empilhamos **9 linhas consecutivas** dessas features numa matriz `9x4` (janela
centrada no dia `t`), cujo alvo é `Qtde_Casos` do dia central.

Sem vazamento: as 4 features já são defasadas (≥30 dias), então nenhuma linha
da janela `±4` carrega `Qtde_Casos[t]` — o alvo nunca entra na entrada.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dengue_tl.lagged_table import VARIAVEL_ALVO


@dataclass(frozen=True)
class MatrixWindowConfig:
    raio: int = 4  # janela de 2*raio+1 == 9 linhas


def feature_columns(tabela: pd.DataFrame) -> list[str]:
    """Colunas de feature da tabela lagged (tudo menos o alvo)."""
    return [c for c in tabela.columns if c != VARIAVEL_ALVO]


def build_matrix_windows(tabela: pd.DataFrame, config: MatrixWindowConfig = MatrixWindowConfig()):
    """Da tabela lagged para `(X, y)`.

    - `X`: array `(n, 2*raio+1, n_features)` — uma matriz por dia central.
    - `y`: array `(n,)` — `Qtde_Casos` do dia central.

    Dias de borda (sem contexto completo dos dois lados) são descartados:
    `n == len(tabela) - 2*raio`.
    """
    colunas = feature_columns(tabela)
    features = tabela[colunas].to_numpy(dtype=float)  # (N, F)
    alvo = tabela[VARIAVEL_ALVO].to_numpy(dtype=float)  # (N,)
    raio = config.raio
    n_total = len(tabela)

    janelas = []
    alvos = []
    for t in range(raio, n_total - raio):
        janelas.append(features[t - raio : t + raio + 1])  # (2*raio+1, F)
        alvos.append(alvo[t])

    largura = 2 * raio + 1
    X = np.array(janelas, dtype=float).reshape(-1, largura, len(colunas))
    y = np.array(alvos, dtype=float)
    return X, y
