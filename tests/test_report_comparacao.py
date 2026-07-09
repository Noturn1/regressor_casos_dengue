import json

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from dengue_tl.report.comparacao import gera_comparacao_completa

ARTEFATOS = {
    "tabela_comparacao_final",
    "predicoes_comparacao",
    "cmp_real_vs_previsto",
    "cmp_dispersao",
    "cmp_mae_por_faixa",
    "cmp_metricas",
    "cmp_amplitude_pico",
}


def _resultado(arq, n=60):
    rng = np.random.default_rng(abs(hash(arq)) % 1000)
    yt = rng.integers(0, 120, n).astype(float).tolist()
    return {
        "config": {"arquitetura": arq, "modo": "lagged"},
        "metricas": {
            "modelo": {"mae": 29.0, "rmse": 78.0, "cc": 0.66},
            "baseline_media": {"mae": 40.2, "rmse": 110.7, "cc": 0.0},
            "baseline_historico": {"mae": 40.3, "rmse": 98.9, "cc": 0.55},
        },
        "predicoes_teste": {
            "y_true": yt,
            "y_pred_modelo": [v * 0.8 + 3 for v in yt],
        },
    }


def _monta_outputs(tmp_path, arqs=("cnn2d", "mlp")):
    for arq in arqs:
        d = tmp_path / arq
        d.mkdir()
        (d / "resultado.json").write_text(json.dumps(_resultado(arq)))
    return tmp_path


def test_gera_comparacao_completa(tmp_path):
    base = _monta_outputs(tmp_path)

    artefatos = gera_comparacao_completa(output_dir=base, dpi=50)

    assert set(artefatos) == ARTEFATOS
    for caminho in artefatos.values():
        assert caminho.exists() and caminho.stat().st_size > 0
        assert caminho.parent == base / "comparacao"

    # A tabela tem uma linha por metodo + 2 baselines, com as colunas ricas.
    import csv
    linhas = list(csv.DictReader(open(artefatos["tabela_comparacao_final"])))
    metodos = {l["metodo"] for l in linhas}
    assert {"cnn2d", "mlp", "baseline_media", "baseline_historico"} <= metodos
    modelo = next(l for l in linhas if l["metodo"] == "cnn2d")
    assert modelo["MAE_pico_ge10"] and modelo["pred_max"] and modelo["ganho_mae_pct_vs_baseline"]


def test_comparacao_exige_duas_arquiteturas(tmp_path):
    _monta_outputs(tmp_path, arqs=("cnn2d",))
    with pytest.raises(SystemExit):
        gera_comparacao_completa(output_dir=tmp_path)
