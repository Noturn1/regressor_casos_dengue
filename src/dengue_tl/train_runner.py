"""Runner de treino/validacao para o pipeline de regressao de dengue.

Este modulo "amarra" as etapas ja implementadas:
1) carrega serie
2) constroi janelas/alvos
3) codifica em GASF
4) faz split temporal treino/validacao/teste
5) treina EfficientNet em 2 fases
6) avalia modelo e baselines
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_tl.scaler import Scaler
from dengue_tl.series_loader import load
from dengue_tl.window_builder import JanelaCentradaConfig, build_centrado


@dataclass(frozen=True)
class SplitConfig:
    treino_fracao: float = 0.7
    validacao_fracao: float = 0.15


@dataclass(frozen=True)
class TreinoConfig:
    csv_path: str
    date_column: str = "Data"
    raio: int = 4
    incluir_dia_central: bool = False
    batch_size: int = 32
    epocas_fase1: int = 20
    epocas_fase2: int = 10
    paciencia_early_stopping: int = 5
    learning_rate_fase1: float = 1e-3
    learning_rate_fase2: float = 1e-4
    n_camadas_finais: int = 20
    seed: int = 42
    split: SplitConfig = SplitConfig()


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


def carrega_serie_casos(csv_path: str, date_column: str = "Data") -> np.ndarray:
    """Carrega `Qtde_Casos` em ordem temporal.

    - Se houver `date_column`, usa o loader oficial com validacao de contiguidade.
    - Se nao houver `date_column` (caso da amostra), usa ordem sequencial do CSV.
    """
    colunas = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if date_column in colunas:
        serie = load(csv_path, date_column=date_column)
        return serie["Qtde_Casos"].to_numpy(dtype=float)

    if "Qtde_Casos" not in colunas:
        raise ValueError(
            f"CSV sem coluna '{date_column}' e sem coluna obrigatoria 'Qtde_Casos'."
        )
    df = pd.read_csv(csv_path)
    return df["Qtde_Casos"].to_numpy(dtype=float)


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

    return slice(0, fim_treino), slice(fim_treino, fim_validacao), slice(
        fim_validacao, n_amostras
    )


def baseline_media(y_treino: np.ndarray, tamanho: int) -> np.ndarray:
    return np.full(shape=(tamanho,), fill_value=float(np.mean(y_treino)), dtype=float)


def baseline_ultimo_vizinho(
    janelas: np.ndarray, raio: int, incluir_dia_central: bool
) -> np.ndarray:
    """Baseline persistente: usa o valor de t-1 disponivel na janela."""
    if incluir_dia_central:
        indice_t_menos_1 = raio - 1
    else:
        indice_t_menos_1 = raio - 1
    return janelas[:, indice_t_menos_1].astype(float)


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

    casos = carrega_serie_casos(config.csv_path, date_column=config.date_column)
    janelas, alvos = build_centrado(
        casos,
        JanelaCentradaConfig(
            raio=config.raio, incluir_dia_central=config.incluir_dia_central
        ),
    )
    minimo_split = _min_amostras_para_split(
        treino_fracao=config.split.treino_fracao,
        validacao_fracao=config.split.validacao_fracao,
    )
    if len(alvos) < minimo_split:
        raio_maximo = (len(casos) - minimo_split) // 2
        sugestao_raio = (
            f"--raio <= {raio_maximo}" if raio_maximo >= 1 else "usar serie maior"
        )
        raise ValueError(
            "Poucas amostras apos janela centrada para o split configurado. "
            f"Linhas no CSV: {len(casos)}; raio atual: {config.raio}; "
            f"amostras geradas: {len(alvos)}; minimo exigido: {minimo_split}. "
            f"Sugestao: {sugestao_raio} (ou ajustar --treino-fracao/--validacao-fracao)."
        )

    treino_sl, val_sl, teste_sl = split_temporal(
        len(alvos),
        treino_fracao=config.split.treino_fracao,
        validacao_fracao=config.split.validacao_fracao,
    )

    # Baselines operam na escala original de casos.
    y_treino_raw = alvos[treino_sl].astype(float)
    y_teste_raw = alvos[teste_sl].astype(float)
    baseline_media_pred = baseline_media(y_treino_raw, tamanho=len(y_teste_raw))
    baseline_ultimo_pred = baseline_ultimo_vizinho(
        janelas[teste_sl], raio=config.raio, incluir_dia_central=config.incluir_dia_central
    )

    # Alvo do modelo em log1p, sempre fitado apenas no treino.
    scaler = Scaler()
    y_treino = scaler.transform_target(alvos[treino_sl])
    y_val = scaler.transform_target(alvos[val_sl])

    # Import lazy para permitir testes deste modulo sem dependencias de DL.
    from dengue_tl.encoder import encode_gasf
    from dengue_tl.model import build_model, descongela_backbone, keras

    imagens = np.stack([encode_gasf(j) for j in janelas], axis=0).astype("float32")
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
    y_pred = scaler.inverse_target(y_pred_log)

    metricas_modelo = calcula_metricas(y_teste_raw, y_pred)
    metricas_baseline_media = calcula_metricas(y_teste_raw, baseline_media_pred)
    metricas_baseline_ultimo = calcula_metricas(y_teste_raw, baseline_ultimo_pred)

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
            "modelo": metricas_modelo,
            "baseline_media": metricas_baseline_media,
            "baseline_ultimo_vizinho": metricas_baseline_ultimo,
        },
        "predicoes_teste": {
            "y_true": [float(v) for v in y_teste_raw],
            "y_pred_modelo": [float(v) for v in y_pred],
            "y_pred_baseline_media": [float(v) for v in baseline_media_pred],
            "y_pred_baseline_ultimo_vizinho": [float(v) for v in baseline_ultimo_pred],
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina e avalia o pipeline de regressao de dengue."
    )
    parser.add_argument("--csv", required=True, help="Caminho para o CSV de entrada.")
    parser.add_argument(
        "--date-column",
        default="Data",
        help="Coluna de data do CSV (se ausente, usa ordem sequencial).",
    )
    parser.add_argument(
        "--output-json",
        default="resultados_treino.json",
        help="Arquivo JSON de saida com metricas e predicoes.",
    )
    parser.add_argument("--raio", type=int, default=4)
    parser.add_argument("--incluir-dia-central", action="store_true")
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
        date_column=args.date_column,
        raio=args.raio,
        incluir_dia_central=args.incluir_dia_central,
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
