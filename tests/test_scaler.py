import numpy as np

from dengue_tl.scaler import Scaler


def test_transform_mapeia_treino_para_0_1():
    treino = np.array(
        [
            [0.0, 10.0, 100.0, 1.0],
            [5.0, 20.0, 200.0, 3.0],
            [10.0, 30.0, 300.0, 5.0],
        ]
    )

    escalado = Scaler().fit(treino).transform(treino)

    np.testing.assert_allclose(escalado.min(axis=0), [0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(escalado.max(axis=0), [1.0, 1.0, 1.0, 1.0])


def test_feature_fora_do_range_do_treino_mapeia_fora_de_0_1():
    treino = np.array([[0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0]])
    scaler = Scaler().fit(treino)

    # var0 acima do máximo do treino; var1 abaixo do mínimo do treino
    escalado = scaler.transform(np.array([[15.0, -5.0, 5.0, 10.0]]))

    assert escalado[0, 0] > 1.0  # prova que o máximo veio só do treino (sem clip/refit)
    assert escalado[0, 1] < 0.0  # prova que o mínimo veio só do treino


def test_inverse_target_desfaz_transform_target():
    scaler = Scaler()
    casos = np.array([0.0, 1.0, 7.0, 42.0, 300.0])

    recuperado = scaler.inverse_target(scaler.transform_target(casos))

    np.testing.assert_allclose(recuperado, casos)
