import json

from dengue_tl.menu import _resultado_para_relatorio


def test_resultado_para_relatorio_passa_json_de_treino_direto(tmp_path):
    caminho = tmp_path / "resultados_cnn_lstm.json"
    caminho.write_text(json.dumps({"metricas": {}}), encoding="utf-8")

    assert _resultado_para_relatorio(caminho) == caminho


def test_resultado_para_relatorio_extrai_resultado_final_do_tune(tmp_path):
    # Saida do tune_runner: o bloco que o report entende esta em `resultado_final`.
    caminho = tmp_path / "resultados_otimizacao.json"
    caminho.write_text(
        json.dumps({"melhores_hiperparametros": {}, "resultado_final": {"metricas": {}}}),
        encoding="utf-8",
    )

    derivado = _resultado_para_relatorio(caminho)

    assert derivado == tmp_path / "resultados_otimizacao_final.json"
    assert json.loads(derivado.read_text(encoding="utf-8")) == {"metricas": {}}
