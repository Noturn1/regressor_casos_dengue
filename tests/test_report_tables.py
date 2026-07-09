import numpy as np
import pandas as pd

from dengue_tl.report.tables import (
    build_data_structure_table,
    build_error_by_range_table,
    build_metrics_table,
    build_pipeline_config_table,
    build_split_table,
    classificar_faixas,
    save_table,
)


def _resultados_exemplo():
    rng = np.random.default_rng(42)
    y_true = rng.uniform(0, 100, size=30).tolist()
    return {
        "config": {"raio": 4, "lag_clima": 45, "lag_historico": 30},
        "split": {"treino": [0, 70], "validacao": [70, 85], "teste": [85, 100]},
        "metricas": {
            "modelo": {"mae": 38.1, "rmse": 107.4, "cc": 0.63},
            "baseline_media": {"mae": 40.2, "rmse": 110.7, "cc": 0.0},
            "baseline_historico": {"mae": 40.3, "rmse": 98.9, "cc": 0.55},
        },
        "predicoes_teste": {
            "y_true": y_true,
            "y_pred_modelo": [v + 1.0 for v in y_true],
            "y_pred_baseline_media": [50.0] * 30,
            "y_pred_baseline_historico": [v - 2.0 for v in y_true],
        },
    }


def test_build_data_structure_table_classifica_papeis():
    df = pd.DataFrame(
        {
            "Data": pd.date_range("2020-01-01", periods=3),
            "Precipitacao": [0.0, 1.0, 2.0],
            "Temp_media": [24.0, 23.0, 22.0],
            "Umidade_rel": [90.0, 91.0, 92.0],
            "Qtde_Casos": [0, 1, 2],
        }
    )

    tabela = build_data_structure_table(df)
    papeis = dict(zip(tabela["coluna"], tabela["papel_da_variavel"]))

    assert papeis["Data"] == "indice temporal"
    assert papeis["Precipitacao"] == "clima / entrada do modelo"
    assert papeis["Qtde_Casos"] == "alvo / historico"


def test_build_data_structure_table_calcula_estatisticas_numericas():
    df = pd.DataFrame({"Qtde_Casos": [0.0, 10.0, 20.0]})

    tabela = build_data_structure_table(df)
    linha = tabela.iloc[0]

    assert linha["minimo"] == 0.0
    assert linha["maximo"] == 20.0
    assert linha["media"] == 10.0
    assert linha["nulos"] == 0


def test_build_pipeline_config_table_deriva_janela_do_raio():
    tabela = build_pipeline_config_table(_resultados_exemplo())
    valores = dict(zip(tabela["parametro"], tabela["valor"]))

    assert valores["raio"] == 4
    assert valores["tamanho_da_janela"] == 9
    assert valores["shape_matriz_entrada"] == "(9, 4)"


def test_build_split_table_calcula_quantidades_e_porcentagens():
    tabela = build_split_table(_resultados_exemplo())

    assert tabela["conjunto"].tolist() == ["treino", "validacao", "teste"]
    assert tabela["quantidade_amostras"].tolist() == [70, 15, 15]
    assert tabela["porcentagem_do_total"].tolist() == [70.0, 15.0, 15.0]


def test_build_metrics_table_ordena_modelo_e_baselines():
    tabela = build_metrics_table(_resultados_exemplo())

    assert tabela["metodo"].tolist() == [
        "modelo",
        "baseline_media",
        "baseline_historico",
    ]
    assert tabela.loc[0, "MAE"] == 38.1


def test_build_metrics_table_aceita_nome_legado_do_baseline():
    resultados = _resultados_exemplo()
    resultados["metricas"]["baseline_ultimo_vizinho"] = resultados["metricas"].pop(
        "baseline_historico"
    )

    tabela = build_metrics_table(resultados)

    assert tabela.loc[2, "RMSE"] == 98.9


def test_classificar_faixas_gera_tres_faixas_ordenadas():
    valores = pd.Series(np.arange(30, dtype=float))

    faixa, ordem = classificar_faixas(valores)

    assert ordem == ["baixa", "media", "alta"]
    assert faixa.iloc[0] == "baixa"
    assert faixa.iloc[-1] == "alta"


def test_classificar_faixas_com_valores_constantes_vira_faixa_unica():
    faixa, ordem = classificar_faixas(pd.Series([5.0] * 10))

    assert ordem == ["unica"]
    assert set(faixa) == {"unica"}


def test_build_error_by_range_table_calcula_mae_e_rmse_por_faixa():
    # Predição deslocada em +1 do real: MAE e RMSE devem ser 1 em toda faixa.
    tabela = build_error_by_range_table(_resultados_exemplo())

    assert set(tabela["faixa"]) == {"baixa", "media", "alta"}
    assert tabela["quantidade_amostras"].sum() == 30
    assert np.allclose(tabela["MAE"], 1.0)
    assert np.allclose(tabela["RMSE"], 1.0)


def test_save_table_escreve_csv_sem_indice(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    caminho = tmp_path / "tabela.csv"

    save_table(df, caminho)

    recarregada = pd.read_csv(caminho)
    pd.testing.assert_frame_equal(df, recarregada)
