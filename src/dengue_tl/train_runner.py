"""Runner de treino/validacao do pipeline de regressao de dengue (entrada 9x4).

Amarra as etapas do pipeline atual:
1) constroi (ou carrega do cache) a tabela lagged: clima em t-45, historico em t-30, alvo em t
2) janela as features em matrizes 9x4 (dia central = alvo)
3) split temporal treino/validacao/teste (sem embaralhar)
4) escala as features por-coluna (fit so no treino) e codifica cada 9x4 em imagem
5) treina EfficientNet em 2 fases
6) avalia modelo e baselines (media do treino / historico t-30) na escala original
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from dengue_tl.lagged_table import (
    LaggedTableConfig,
    build_or_load_lagged_table,
)
from dengue_tl.matrix_windower import (
    MatrixWindowConfig,
    build_matrix_windows,
    feature_columns,
)
from dengue_tl.scaler import Scaler


@dataclass(frozen=True)
class SplitConfig:
    treino_fracao: float = 0.7
    validacao_fracao: float = 0.15


@dataclass(frozen=True)
class TreinoConfig:
    csv_path: str
    cache_path: str = "cache/tabela_lagged.csv"
    lag_clima: int = 45
    lag_historico: int = 30
    raio: int = 4  # janela de 2*raio+1 == 9 linhas
    batch_size: int = 32
    epocas_fase1: int = 20
    epocas_fase2: int = 10
    paciencia_early_stopping: int = 5
    learning_rate_fase1: float = 1e-3
    learning_rate_fase2: float = 1e-4
    n_camadas_finais: int = 20
    seed: int = 42
    split: SplitConfig = field(default_factory=SplitConfig)


def _min_amostras_para_split(
    treino_fracao: float, validacao_fracao: float, limite_busca: int = 10_000
) -> int:
    """Menor n para o qual o split temporal gera treino/val/teste validos."""
    for n in range(1, limite_busca + 1):
        try:
            split_temporal(
                n_amostras=n,
                treino_fracao=treino_fracao,
                validacao_fracao=validacao_fracao,
            )
            return n
        except ValueError:
            continue
    raise ValueError(
        "Nao foi possivel encontrar tamanho minimo para o split com as fracoes fornecidas."
    )


def carrega_tabela_lagged(config: TreinoConfig):
    """Constroi (ou carrega do cache) a tabela lagged a partir do CSV.

    Dateless: o dataset completo nao tem coluna de data; `build_lagged_table`
    aplica os lags posicionalmente. O cache evita recomputar os lags.
    """
    return build_or_load_lagged_table(
        config.csv_path,
        config.cache_path,
        LaggedTableConfig(
            lag_clima=config.lag_clima, lag_historico=config.lag_historico
        ),
    )


def split_temporal(
    n_amostras: int, treino_fracao: float, validacao_fracao: float
) -> tuple[slice, slice, slice]:
    """Retorna slices (treino, validacao, teste) preservando ordem temporal."""
    if not (0 < treino_fracao < 1):
        raise ValueError("`treino_fracao` deve estar entre 0 e 1.")
    if not (0 <= validacao_fracao < 1):
        raise ValueError("`validacao_fracao` deve estar entre 0 e 1.")
    if treino_fracao + validacao_fracao >= 1:
        raise ValueError("`treino_fracao + validacao_fracao` deve ser < 1.")

    fim_treino = int(n_amostras * treino_fracao)
    fim_validacao = int(n_amostras * (treino_fracao + validacao_fracao))

    if fim_treino <= 0 or fim_validacao <= fim_treino or fim_validacao >= n_amostras:
        raise ValueError(
            "Split invalido para o tamanho da serie: ajuste fracoes ou use mais dados."
        )

    return (
        slice(0, fim_treino),
        slice(fim_treino, fim_validacao),
        slice(fim_validacao, n_amostras),
    )


def baseline_media(y_treino: np.ndarray, tamanho: int) -> np.ndarray:
    return np.full(shape=(tamanho,), fill_value=float(np.mean(y_treino)), dtype=float)


def baseline_historico(X: np.ndarray, raio: int, idx_historico: int) -> np.ndarray:
    """Baseline de persistencia: usa o historico de casos (t-30) do dia central.

    E o valor de casos mais recente disponivel como feature; um estimador
    ingenuo honesto para `Qtde_Casos[t]`.
    """
    return X[:, raio, idx_historico].astype(float)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def cc_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def calcula_metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "cc": cc_pearson(y_true, y_pred),
    }


def _set_semente(seed: int) -> None:
    np.random.seed(seed)


def treina_e_avalia(config: TreinoConfig) -> dict[str, object]:
    """Executa treino em 2 fases e avalia modelo + baselines."""
    _set_semente(config.seed)

    tabela = carrega_tabela_lagged(config)
    X, y = build_matrix_windows(tabela, MatrixWindowConfig(raio=config.raio))

    minimo_split = _min_amostras_para_split(
        treino_fracao=config.split.treino_fracao,
        validacao_fracao=config.split.validacao_fracao,
    )
    if len(y) < minimo_split:
        raise ValueError(
            "Amostras insuficientes apos aplicar lags e janela 9x4. "
            f"Linhas na tabela lagged: {len(tabela)} (os primeiros "
            f"{max(config.lag_clima, config.lag_historico)} dias somem no dropna dos lags); "
            f"raio: {config.raio}; amostras 9x4 geradas: {len(y)}; minimo exigido: {minimo_split}. "
            "Sugestao: usar uma serie maior ou reduzir --lag-clima/--lag-historico/--raio."
        )

    treino_sl, val_sl, teste_sl = split_temporal(
        len(y),
        treino_fracao=config.split.treino_fracao,
        validacao_fracao=config.split.validacao_fracao,
    )

    # Baselines na escala original de casos (usam X/y nao escalados).
    y_treino_raw = y[treino_sl].astype(float)
    y_teste_raw = y[teste_sl].astype(float)
    idx_historico = feature_columns(tabela).index(
        f"Historico_lag{config.lag_historico}"
    )
    baseline_media_pred = baseline_media(y_treino_raw, tamanho=len(y_teste_raw))
    baseline_historico_pred = baseline_historico(
        X[teste_sl], raio=config.raio, idx_historico=idx_historico
    )

    # Escala das features por-coluna: fit SO no treino (sem vazamento temporal).
    scaler_x = Scaler().fit(X[treino_sl])
    X_escalado = scaler_x.transform(X)

    # Alvo do modelo em log1p.
    scaler_y = Scaler()
    y_treino = scaler_y.transform_target(y[treino_sl])
    y_val = scaler_y.transform_target(y[val_sl])

    # Import lazy: permite testar a logica (split/baselines) sem dependencias de DL.
    from dengue_tl.encoder import encode_matrix
    from dengue_tl.model import build_model, descongela_backbone, keras

    imagens = np.stack([encode_matrix(m) for m in X_escalado], axis=0).astype("float32")
    x_treino = imagens[treino_sl]
    x_val = imagens[val_sl]
    x_teste = imagens[teste_sl]

    model = build_model(weights="imagenet")
    optimizer_fase1 = keras.optimizers.Adam(learning_rate=config.learning_rate_fase1)
    model.compile(optimizer=optimizer_fase1, loss="mse", metrics=["mae"])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.paciencia_early_stopping,
            restore_best_weights=True,
        )
    ]

    hist_fase1 = model.fit(
        x_treino,
        y_treino,
        validation_data=(x_val, y_val),
        epochs=config.epocas_fase1,
        batch_size=config.batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    model = descongela_backbone(model, n_camadas_finais=config.n_camadas_finais)
    optimizer_fase2 = keras.optimizers.Adam(learning_rate=config.learning_rate_fase2)
    model.compile(optimizer=optimizer_fase2, loss="mse", metrics=["mae"])

    hist_fase2 = model.fit(
        x_treino,
        y_treino,
        validation_data=(x_val, y_val),
        epochs=config.epocas_fase2,
        batch_size=config.batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    y_pred_log = model.predict(x_teste, verbose=0).reshape(-1)
    y_pred = scaler_y.inverse_target(y_pred_log)

    return {
        "config": asdict(config),
        "split": {
            "treino": [treino_sl.start, treino_sl.stop],
            "validacao": [val_sl.start, val_sl.stop],
            "teste": [teste_sl.start, teste_sl.stop],
        },
        "historico": {
            "fase1": {k: [float(vv) for vv in v] for k, v in hist_fase1.history.items()},
            "fase2": {k: [float(vv) for vv in v] for k, v in hist_fase2.history.items()},
        },
        "metricas": {
            "modelo": calcula_metricas(y_teste_raw, y_pred),
            "baseline_media": calcula_metricas(y_teste_raw, baseline_media_pred),
            "baseline_historico": calcula_metricas(y_teste_raw, baseline_historico_pred),
        },
        "predicoes_teste": {
            "y_true": [float(v) for v in y_teste_raw],
            "y_pred_modelo": [float(v) for v in y_pred],
            "y_pred_baseline_media": [float(v) for v in baseline_media_pred],
            "y_pred_baseline_historico": [float(v) for v in baseline_historico_pred],
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina e avalia o pipeline de regressao de dengue (entrada 9x4)."
    )
    parser.add_argument("--csv", required=True, help="Caminho para o CSV de entrada.")
    parser.add_argument(
        "--cache-path",
        default="cache/tabela_lagged.csv",
        help="Onde salvar/ler a tabela lagged (cache).",
    )
    parser.add_argument("--lag-clima", type=int, default=45)
    parser.add_argument("--lag-historico", type=int, default=30)
    parser.add_argument(
        "--output-json",
        default="resultados_treino.json",
        help="Arquivo JSON de saida com metricas e predicoes.",
    )
    parser.add_argument("--raio", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epocas-fase1", type=int, default=20)
    parser.add_argument("--epocas-fase2", type=int, default=10)
    parser.add_argument("--paciencia", type=int, default=5)
    parser.add_argument("--lr-fase1", type=float, default=1e-3)
    parser.add_argument("--lr-fase2", type=float, default=1e-4)
    parser.add_argument("--n-camadas-finais", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--treino-fracao", type=float, default=0.7)
    parser.add_argument("--validacao-fracao", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = TreinoConfig(
        csv_path=args.csv,
        cache_path=args.cache_path,
        lag_clima=args.lag_clima,
        lag_historico=args.lag_historico,
        raio=args.raio,
        batch_size=args.batch_size,
        epocas_fase1=args.epocas_fase1,
        epocas_fase2=args.epocas_fase2,
        paciencia_early_stopping=args.paciencia,
        learning_rate_fase1=args.lr_fase1,
        learning_rate_fase2=args.lr_fase2,
        n_camadas_finais=args.n_camadas_finais,
        seed=args.seed,
        split=SplitConfig(
            treino_fracao=args.treino_fracao,
            validacao_fracao=args.validacao_fracao,
        ),
    )
    resultado = treina_e_avalia(config)
    output = Path(args.output_json)
    output.write_text(json.dumps(resultado, indent=2), encoding="utf-8")
    print(f"Resultado salvo em: {output}")


if __name__ == "__main__":
    main()
