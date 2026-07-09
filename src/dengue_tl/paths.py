"""Convencao unica de caminhos de saida: tudo agrupado por arquitetura.

Em vez de espalhar `resultados_*.json`, `best_hiperparametros_*.json` e a pasta
de relatorio na raiz do projeto, todos os artefatos gerados vivem sob
`outputs/<arquitetura>/`:

    outputs/<arch>/resultado.json     -> experiment.py / train_runner.py
    outputs/<arch>/otimizacao.json    -> tune_runner.py (busca Optuna)
    outputs/<arch>/best_config.json   -> menu (melhor config persistida)
    outputs/<arch>/relatorio/         -> tabelas + graficos do report
    outputs/comparacao/               -> report comparativo entre arquiteturas

`outputs/` inteira e ignorada pelo git (artefatos regeneraveis). O cache da
tabela lagged continua em `cache/` (entrada, nao saida).
"""

from __future__ import annotations

from pathlib import Path

OUTPUTS_DIR = Path("outputs")


def dir_arquitetura(arquitetura: str) -> Path:
    return OUTPUTS_DIR / arquitetura


def caminho_resultado(arquitetura: str) -> Path:
    return dir_arquitetura(arquitetura) / "resultado.json"


def caminho_otimizacao(arquitetura: str) -> Path:
    return dir_arquitetura(arquitetura) / "otimizacao.json"


def caminho_best_config(arquitetura: str) -> Path:
    return dir_arquitetura(arquitetura) / "best_config.json"


def dir_relatorio(arquitetura: str) -> Path:
    return dir_arquitetura(arquitetura) / "relatorio"


def garante_pai(caminho: str | Path) -> Path:
    """Cria a pasta-pai do arquivo (parents=True) e devolve o caminho como Path."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    return caminho
