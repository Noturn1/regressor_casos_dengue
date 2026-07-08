import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")  # requer o extra `dl` (rodar no venv)
pytest.importorskip("optuna")  # requer o extra `opt`

from dengue_tl.train_runner import TreinoConfig
from dengue_tl.tune_runner import otimiza


def _csv_dateless(tmp_path, n):
    csv = tmp_path / "dados.csv"
    rng = np.random.default_rng(0)
    pd.DataFrame(
        {
            "Precipitacao": rng.random(n),
            "Temp_media": 20 + rng.random(n),
            "Umidade_rel": 70 + rng.random(n),
            "Qtde_Casos": rng.integers(0, 20, n).astype(float),
        }
    ).to_csv(csv, index=False)
    return csv


def test_espaco_busca_sugere_apenas_campos_do_config():
    # Garante que replace(config, **espaco_busca(trial)) nunca quebra.
    import optuna

    from dengue_tl.models import ARQUITETURAS, seleciona_arquitetura

    campos = set(TreinoConfig.__dataclass_fields__)
    for nome in ARQUITETURAS:
        study = optuna.create_study()
        trial = study.ask()

        params = seleciona_arquitetura(nome).espaco_busca(trial)

        assert params, nome
        assert set(params) <= campos, nome


def test_otimiza_cnn_lstm_fim_a_fim(tmp_path):
    csv = _csv_dateless(tmp_path, n=90)
    config = TreinoConfig(
        csv_path=str(csv),
        cache_path=str(tmp_path / "cache.csv"),
        arquitetura="cnn_lstm",
        epocas_fase1=2,
        epocas_fase2=0,
        paciencia_early_stopping=2,
    )

    resultado = otimiza(config, n_trials=2)

    assert len(resultado["trials"]) == 2
    assert resultado["melhor_val_mae"] == pytest.approx(
        min(t["val_mae"] for t in resultado["trials"])
    )
    # A melhor config sugerida chega inteira ao retreino final.
    config_final = resultado["resultado_final"]["config"]
    for chave, valor in resultado["melhores_hiperparametros"].items():
        assert config_final[chave] == valor
    assert "modelo" in resultado["resultado_final"]["metricas"]
    json.dumps(resultado)  # saida precisa ser serializavel
