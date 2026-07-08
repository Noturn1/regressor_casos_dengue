"""Entrada de dados do relatório: CSV bruto e JSON de resultados do treino."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from dengue_tl.series_loader import repara_valores_numericos

DEFAULT_OUTPUT_DIR = Path("relatorio_outputs")
DEFAULT_DPI = 220


@dataclass(frozen=True)
class ReportConfig:
    """Configuração de entrada e saída do gerador de relatório."""

    csv_path: str
    results_path: str
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    date_column: str = "Data"
    dpi: int = DEFAULT_DPI


class TestPredictions(NamedTuple):
    """Séries do conjunto de teste extraídas do JSON de resultados."""

    y_true: np.ndarray
    y_model: np.ndarray
    y_media: np.ndarray
    y_hist: np.ndarray


def load_training_results(results_path: str | Path) -> dict[str, Any]:
    """Carrega o JSON produzido pelo runner."""
    with Path(results_path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_raw_dataset(csv_path: str | Path, date_column: str = "Data") -> pd.DataFrame:
    """Lê o CSV bruto reparando valores corrompidos, preservando data quando existir.

    O CSV bruto tem blocos com floats exportados com `.` de milhar e sem o
    separador decimal; sem `repara_valores_numericos` essas colunas viram
    strings e os gráficos saem com escala absurda.
    """
    csv_path = Path(csv_path)
    cabecalho = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if date_column in cabecalho:
        df = pd.read_csv(csv_path, parse_dates=[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)
    else:
        df = pd.read_csv(csv_path)
    return repara_valores_numericos(df)


def time_axis(df: pd.DataFrame, date_column: str = "Data") -> pd.Series:
    """Retorna eixo temporal com data ou índice sequencial."""
    if date_column in df.columns:
        return pd.to_datetime(df[date_column])
    return pd.Series(np.arange(len(df)), name="indice")


def extract_test_predictions(results: dict[str, Any]) -> TestPredictions:
    """Extrai as séries de teste do JSON, falhando alto se estiverem ausentes."""
    preds = results.get("predicoes_teste", {})
    y_true = np.asarray(preds.get("y_true", []), dtype=float)
    if y_true.size == 0:
        raise ValueError(
            "JSON de resultados sem `predicoes_teste.y_true`: gere o arquivo com o "
            "train_runner antes de montar o relatório."
        )
    y_model = np.asarray(preds.get("y_pred_modelo", []), dtype=float)
    y_media = np.asarray(preds.get("y_pred_baseline_media", []), dtype=float)
    y_hist = np.asarray(
        preds.get("y_pred_baseline_historico")
        or preds.get("y_pred_baseline_ultimo_vizinho", []),
        dtype=float,
    )
    return TestPredictions(y_true, y_model, y_media, y_hist)


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Cria a pasta de saída se ela não existir."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    return output
