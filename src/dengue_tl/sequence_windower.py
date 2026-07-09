"""Janela causal crua (sem defasagem embutida) para o LSTM descobrir a dependencia temporal.

Contraste com `matrix_windower` (abordagem lagged): la as features ja chegam
defasadas (clima@t-45, casos@t-30), o que **crava na mao** a dependencia temporal.
Aqui empilhamos os `janela_dias` dias consecutivos das 4 variaveis **brutas**
(Precipitacao, Temp_media, Umidade_rel, Qtde_Casos) terminando em `t - gap_dias`,
e deixamos o LSTM aprender qual defasagem importa (recomendacao do professor).

- Alvo: `Qtde_Casos[t]`.
- Entrada: `[t - gap - janela + 1 : t - gap + 1]` -> matriz `(janela_dias, 4)`.
- `gap_dias >= 1` garante que `Qtde_Casos[t]` **nunca** entra na janela (sem
  vazamento) e preserva a premissa de atraso do projeto. O caso mais recente
  disponivel, `Qtde_Casos[t - gap]` (ultimo dia da coluna de casos na janela), e
  o denominador natural da log-razao e o baseline de persistencia.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_tl.series_loader import VARIAVEIS, repara_valores_numericos

VARIAVEL_ALVO = "Qtde_Casos"


@dataclass(frozen=True)
class SequenceWindowConfig:
    janela_dias: int = 60
    gap_dias: int = 30
    date_column: str = "Data"


def _carrega_serie(dados, date_column: str) -> pd.DataFrame:
    """Serie diaria bruta, ordenada, com as 4 VARIAVEIS (path de CSV ou DataFrame)."""
    if isinstance(dados, (str, Path)):
        base = pd.read_csv(dados)
    elif isinstance(dados, pd.DataFrame):
        base = dados.copy()
    else:
        raise TypeError("`dados` deve ser um caminho de CSV ou um pandas.DataFrame.")

    if date_column in base.columns:
        base = base.sort_values(date_column).reset_index(drop=True)

    faltantes = [c for c in VARIAVEIS if c not in base.columns]
    if faltantes:
        raise ValueError(
            "colunas obrigatorias ausentes para a janela crua: " + ", ".join(faltantes)
        )
    return repara_valores_numericos(base)


def build_sequence_windows(dados, config: SequenceWindowConfig = SequenceWindowConfig()):
    """Da serie bruta para `(X, y, historico)`.

    - `X`: `(n, janela_dias, 4)` — os N dias brutos terminando em `t - gap`.
    - `y`: `(n,)` — `Qtde_Casos[t]`.
    - `historico`: `(n,)` — `Qtde_Casos[t - gap]`, o caso mais recente disponivel
      (denominador da log-razao / baseline de persistencia).

    So viram amostra os `t` com contexto completo: `t >= gap + janela - 1`.
    """
    base = _carrega_serie(dados, config.date_column)
    feats = base[VARIAVEIS].to_numpy(dtype=float)  # (T, 4)
    alvo = base[VARIAVEL_ALVO].to_numpy(dtype=float)  # (T,)

    N, gap = config.janela_dias, config.gap_dias
    if N < 1 or gap < 1:
        raise ValueError("`janela_dias` e `gap_dias` devem ser >= 1.")

    janelas, alvos, historicos = [], [], []
    for t in range(gap + N - 1, len(feats)):
        janelas.append(feats[t - gap - N + 1 : t - gap + 1])  # (N, 4)
        alvos.append(alvo[t])
        historicos.append(alvo[t - gap])  # ultimo dia da janela (coluna de casos)

    X = np.array(janelas, dtype=float).reshape(-1, N, len(VARIAVEIS))
    y = np.array(alvos, dtype=float)
    historico = np.array(historicos, dtype=float)
    return X, y, historico
