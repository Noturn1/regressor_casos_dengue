"""Escalonamento das variáveis e transformação do alvo.

As variáveis de entrada são escalonadas para [0,1] com min/max **derivados só
do treino** (evita vazamento temporal). O alvo é treinado em `log1p` e invertido
com `expm1` na avaliação.

Reaproveitado de uma Iniciação Científica, sem alterações.
"""

import numpy as np


class Scaler:
    def fit(self, treino):
        treino = np.asarray(treino, dtype=float)
        eixos = tuple(range(treino.ndim - 1))
        self.min_ = treino.min(axis=eixos)
        self.max_ = treino.max(axis=eixos)
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        return (x - self.min_) / (self.max_ - self.min_)

    def transform_target(self, y):
        return np.log1p(np.asarray(y, dtype=float))

    def inverse_target(self, y):
        return np.expm1(np.asarray(y, dtype=float))
