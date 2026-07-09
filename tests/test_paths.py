from dengue_tl.paths import (
    OUTPUTS_DIR,
    caminho_otimizacao,
    caminho_resultado,
    rotulo_de,
)


def test_rotulo_de_usa_arquitetura_por_padrao():
    assert rotulo_de("cnn_lstm") == "cnn_lstm"
    assert rotulo_de("cnn_lstm", "") == "cnn_lstm"


def test_rotulo_de_custom_sobrescreve():
    assert rotulo_de("cnn_lstm", "cnn_lstm_sequencia") == "cnn_lstm_sequencia"


def test_caminhos_usam_o_rotulo():
    rot = "cnn_lstm_sequencia"
    assert caminho_resultado(rot) == OUTPUTS_DIR / rot / "resultado.json"
    assert caminho_otimizacao(rot) == OUTPUTS_DIR / rot / "otimizacao.json"
