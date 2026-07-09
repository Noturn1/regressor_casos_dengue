import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from dengue_tl.report.artifacts import save_all_report_artifacts

ARTEFATOS_ESPERADOS = {
    "tabela_estrutura_dados",
    "tabela_configuracao_pipeline",
    "tabela_split_temporal",
    "tabela_metricas_modelos",
    "tabela_erro_por_faixa",
    "grafico_variaveis_originais",
    "grafico_modelo_vs_baselines",
    "grafico_metricas_comparativas",
    "grafico_residuos",
    "grafico_erro_por_faixa",
    "grafico_curvas_aprendizado",
    "grafico_residuos_no_tempo",
    "grafico_distribuicao_treino_teste",
}


def _prepara_entradas(tmp_path, n=40):
    rng = np.random.default_rng(7)
    csv_path = tmp_path / "dados.csv"
    dados = pd.DataFrame(
        {
            "Precipitacao": rng.uniform(0, 100, n).round(1),
            "Temp_media": rng.uniform(15, 30, n).round(3),
            "Umidade_rel": rng.uniform(60, 100, n).round(3),
            "Qtde_Casos": rng.integers(0, 50, n),
        }
    )
    # Injeta um valor corrompido como no CSV bruto real (`.` de milhar,
    # decimal perdido): o pipeline do relatório precisa repará-lo.
    dados = dados.astype({"Temp_media": object})
    dados.loc[5, "Temp_media"] = "1.595.629.411.764.700"
    dados.to_csv(csv_path, index=False)

    y_true = rng.uniform(0, 100, n).tolist()
    resultados = {
        "config": {
            "arquitetura": "cnn_lstm",
            "raio": 4,
            "lag_clima": 45,
            "lag_historico": 30,
        },
        "split": {"treino": [0, 28], "validacao": [28, 34], "teste": [34, 40]},
        "metricas": {
            "modelo": {"mae": 38.1, "rmse": 107.4, "cc": 0.63},
            "baseline_media": {"mae": 40.2, "rmse": 110.7, "cc": 0.0},
            "baseline_historico": {"mae": 40.3, "rmse": 98.9, "cc": 0.55},
        },
        "predicoes_teste": {
            "y_true": y_true,
            "y_pred_modelo": [v + 1.0 for v in y_true],
            "y_pred_baseline_media": [50.0] * n,
            "y_pred_baseline_historico": [v - 2.0 for v in y_true],
        },
    }
    results_path = tmp_path / "resultados.json"
    results_path.write_text(json.dumps(resultados))
    return csv_path, results_path


def test_save_all_gera_todos_os_artefatos(tmp_path):
    csv_path, results_path = _prepara_entradas(tmp_path)
    saida = tmp_path / "saida"

    artefatos = save_all_report_artifacts(
        csv_path, results_path, output_dir=saida, dpi=60
    )

    assert set(artefatos) == ARTEFATOS_ESPERADOS
    for caminho in artefatos.values():
        assert caminho.exists(), f"artefato ausente: {caminho}"
        assert caminho.stat().st_size > 0


def test_save_all_separa_saida_por_arquitetura(tmp_path):
    # Artefatos de arquiteturas diferentes nao devem se sobrescrever:
    # cada uma ganha uma subpasta com seu nome dentro da pasta base.
    csv_path, results_path = _prepara_entradas(tmp_path)
    saida = tmp_path / "saida"

    artefatos = save_all_report_artifacts(
        csv_path, results_path, output_dir=saida, dpi=60
    )

    for caminho in artefatos.values():
        assert caminho.parent == saida / "cnn_lstm" / "relatorio"


def test_save_all_usa_rotulo_quando_presente(tmp_path):
    # Rotulo custom separa variantes da mesma arquitetura (ex.: cnn_lstm seq).
    csv_path, results_path = _prepara_entradas(tmp_path)
    resultados = json.loads(results_path.read_text())
    resultados["config"]["rotulo"] = "cnn_lstm_sequencia"
    results_path.write_text(json.dumps(resultados))
    saida = tmp_path / "saida"

    artefatos = save_all_report_artifacts(
        csv_path, results_path, output_dir=saida, dpi=60
    )

    for caminho in artefatos.values():
        assert caminho.parent == saida / "cnn_lstm_sequencia" / "relatorio"


def test_save_all_usa_pasta_modelo_sem_arquitetura_no_json(tmp_path):
    csv_path, results_path = _prepara_entradas(tmp_path)
    resultados = json.loads(results_path.read_text())
    del resultados["config"]["arquitetura"]
    results_path.write_text(json.dumps(resultados))
    saida = tmp_path / "saida"

    artefatos = save_all_report_artifacts(
        csv_path, results_path, output_dir=saida, dpi=60
    )

    for caminho in artefatos.values():
        assert caminho.parent == saida / "modelo" / "relatorio"


def test_save_all_repara_valores_corrompidos_na_estrutura(tmp_path):
    # Regressão do gráfico `variaveis_originais` quebrado: o valor corrompido
    # deve entrar reparado (escala de temperatura), não como string gigante.
    csv_path, results_path = _prepara_entradas(tmp_path)
    saida = tmp_path / "saida"

    artefatos = save_all_report_artifacts(
        csv_path, results_path, output_dir=saida, dpi=60
    )

    estrutura = pd.read_csv(artefatos["tabela_estrutura_dados"])
    temp = estrutura.set_index("coluna").loc["Temp_media"]
    assert temp["tipo_de_dado"] == "float64"
    assert float(temp["maximo"]) < 100
