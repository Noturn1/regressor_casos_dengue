"""Testes da camada central de experimento (lógica de resumo, sem TensorFlow)."""

from dengue_tl.experiment import (
    formata_resumo,
    gera_figuras,
    resumo_pertinente,
)


def _resultado_fake():
    """Dict no formato de `treina_e_avalia`, com o modelo batendo os baselines."""
    return {
        "split": {
            "treino": [0, 70],
            "validacao": [70, 85],
            "teste": [85, 100],
        },
        "historico": {
            "fase1": {"val_loss": [0.9, 0.5, 0.7]},  # melhor: época 1
            "fase2": {"val_loss": [0.4, 0.45]},  # melhor global: fase2 época 0
        },
        "metricas": {
            "modelo": {"mae": 2.0, "rmse": 3.0, "cc": 0.9},
            "baseline_media": {"mae": 5.0, "rmse": 6.0, "cc": 0.0},
            "baseline_historico": {"mae": 4.0, "rmse": 5.0, "cc": 0.8},
        },
    }


def test_resumo_identifica_tamanhos_e_melhor_estimador():
    resumo = resumo_pertinente(_resultado_fake())

    assert resumo["amostras"] == {
        "treino": 70,
        "validacao": 15,
        "teste": 15,
        "total": 100,
    }
    # Modelo é o melhor em todas as métricas neste cenário.
    assert resumo["melhor_por_metrica"] == {"mae": "modelo", "rmse": "modelo", "cc": "modelo"}


def test_resumo_calcula_ganho_sobre_melhor_baseline():
    resumo = resumo_pertinente(_resultado_fake())

    # Melhor baseline em MAE é o histórico (4.0); modelo tem 2.0 -> 50% de ganho.
    assert resumo["melhor_baseline"] == "baseline_historico"
    assert resumo["modelo_bate_baseline"] is True
    assert resumo["ganho_mae_pct_vs_melhor_baseline"] == 50.0


def test_resumo_encontra_melhor_epoca_de_validacao():
    resumo = resumo_pertinente(_resultado_fake())

    assert resumo["melhor_epoca_validacao"] == {
        "fase": "fase2",
        "epoca": 0,
        "val_loss": 0.4,
    }


def test_resumo_detecta_modelo_pior_que_baseline():
    resultado = _resultado_fake()
    resultado["metricas"]["modelo"]["mae"] = 9.0  # pior que os baselines
    resumo = resumo_pertinente(resultado)

    assert resumo["modelo_bate_baseline"] is False
    assert resumo["ganho_mae_pct_vs_melhor_baseline"] < 0


def test_formata_resumo_gera_texto_legivel():
    texto = formata_resumo(resumo_pertinente(_resultado_fake()))

    assert "RESULTADOS" in texto
    assert "modelo" in texto
    assert "Veredito" in texto


def test_gera_figuras_e_noop_sem_modulo_de_visualizacoes(tmp_path):
    # Enquanto `dengue_tl/visualizations.py` não existir, é no-op seguro.
    assert gera_figuras(_resultado_fake(), tmp_path) == []
