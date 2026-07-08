import json

import numpy as np
import pandas as pd
import pytest

from dengue_tl.report.data_io import (
    extract_test_predictions,
    load_raw_dataset,
    load_training_results,
    time_axis,
)


def _csv_sem_data(tmp_path, temp_media):
    caminho = tmp_path / "dados.csv"
    df = pd.DataFrame(
        {
            "Precipitacao": [0.0, 35.4, 9.0],
            "Temp_media": temp_media,
            "Umidade_rel": [88.4, 99.3, 99.5],
            "Qtde_Casos": [0, 1, 2],
        }
    )
    df.to_csv(caminho, index=False)
    return caminho


def test_load_raw_dataset_repara_valores_corrompidos(tmp_path):
    # Bloco do CSV bruto exportado com `.` de milhar e sem separador decimal:
    # sem reparo, a coluna vira string e o gráfico de variáveis originais quebra.
    caminho = _csv_sem_data(
        tmp_path, temp_media=["24.05", "1.595.629.411.764.700", "22.47"]
    )

    df = load_raw_dataset(caminho)

    assert pd.api.types.is_float_dtype(df["Temp_media"])
    assert df["Temp_media"].iloc[1] == pytest.approx(15.956294, abs=1e-4)
    assert df["Temp_media"].max() < 100  # escala plausível de temperatura


def test_load_raw_dataset_sem_coluna_de_data_mantem_ordem(tmp_path):
    caminho = _csv_sem_data(tmp_path, temp_media=[24.0, 23.6, 22.4])

    df = load_raw_dataset(caminho)

    assert "Data" not in df.columns
    assert df["Qtde_Casos"].tolist() == [0, 1, 2]


def test_load_raw_dataset_ordena_por_data(tmp_path):
    caminho = tmp_path / "dados.csv"
    pd.DataFrame(
        {
            "Data": ["2020-01-03", "2020-01-01", "2020-01-02"],
            "Precipitacao": [3.0, 1.0, 2.0],
            "Temp_media": [24.0, 23.0, 22.0],
            "Umidade_rel": [90.0, 91.0, 92.0],
            "Qtde_Casos": [3, 1, 2],
        }
    ).to_csv(caminho, index=False)

    df = load_raw_dataset(caminho)

    assert df["Qtde_Casos"].tolist() == [1, 2, 3]
    assert pd.api.types.is_datetime64_any_dtype(df["Data"])


def test_time_axis_usa_data_quando_existe():
    df = pd.DataFrame({"Data": pd.date_range("2020-01-01", periods=3), "x": [1, 2, 3]})

    eixo = time_axis(df)

    assert eixo.iloc[0] == pd.Timestamp("2020-01-01")


def test_time_axis_cai_para_indice_sequencial():
    df = pd.DataFrame({"x": [1, 2, 3]})

    eixo = time_axis(df)

    assert eixo.tolist() == [0, 1, 2]


def test_load_training_results_le_json(tmp_path):
    caminho = tmp_path / "resultados.json"
    caminho.write_text(json.dumps({"metricas": {"modelo": {"mae": 1.0}}}))

    results = load_training_results(caminho)

    assert results["metricas"]["modelo"]["mae"] == 1.0


def test_extract_test_predictions_retorna_series_alinhadas():
    results = {
        "predicoes_teste": {
            "y_true": [1.0, 2.0],
            "y_pred_modelo": [1.1, 2.1],
            "y_pred_baseline_media": [1.5, 1.5],
            "y_pred_baseline_historico": [0.9, 1.9],
        }
    }

    preds = extract_test_predictions(results)

    assert isinstance(preds.y_true, np.ndarray)
    assert preds.y_model.tolist() == [1.1, 2.1]
    assert preds.y_hist.tolist() == [0.9, 1.9]


def test_extract_test_predictions_aceita_nome_legado_do_baseline():
    results = {
        "predicoes_teste": {
            "y_true": [1.0],
            "y_pred_modelo": [1.1],
            "y_pred_baseline_media": [1.5],
            "y_pred_baseline_ultimo_vizinho": [0.9],
        }
    }

    preds = extract_test_predictions(results)

    assert preds.y_hist.tolist() == [0.9]


def test_extract_test_predictions_falha_alto_sem_predicoes():
    with pytest.raises(ValueError, match="predicoes_teste"):
        extract_test_predictions({})
