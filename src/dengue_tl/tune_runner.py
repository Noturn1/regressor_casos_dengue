"""Busca de hiperparametros com Optuna sobre o pipeline 9x4.

Para cada trial, o Optuna (sampler TPE) sugere uma configuracao a partir do
`espaco_busca` da arquitetura escolhida (ver dengue_tl.models), treina o modelo
e mede o MAE **na validacao, em escala original de casos**. O conjunto de teste
nunca entra na busca: ele so e usado uma vez, no retreino final com a melhor
configuracao (via `treina_e_avalia`), para que a metrica reportada continue
honesta.

Os dados sao preparados uma unica vez (`prepara_dados`) e reutilizados por
todos os trials — importante para a EfficientNet, cuja codificacao das janelas
em imagens 100x100x3 e cara. Requer os extras `dl` (tensorflow) e `optuna`.

Uso:
    python -m dengue_tl.tune_runner --csv data/AmostraDados.csv \
        --arquitetura cnn_lstm --n-trials 50
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from dengue_tl.train_runner import (
    DadosPreparados,
    SplitConfig,
    TreinoConfig,
    mae,
    prepara_dados,
    treina_e_avalia,
)


def _objetivo(trial, dados: DadosPreparados, config_base: TreinoConfig) -> float:
    """Treina uma configuracao sugerida e retorna o MAE de validacao (escala original)."""
    import keras

    from dengue_tl.models import seleciona_arquitetura

    keras.backend.clear_session()  # evita acumular grafos entre trials

    modulo = seleciona_arquitetura(config_base.arquitetura)
    config = replace(config_base, **modulo.espaco_busca(trial))

    model, _ = modulo.treina(
        dados.x_treino, dados.y_treino, dados.x_val, dados.y_val, config
    )

    y_pred_log = model.predict(dados.x_val, verbose=0).reshape(-1)
    y_pred = dados.scaler_y.inverse_target(y_pred_log)
    return mae(dados.y_val_raw, y_pred)


def otimiza(config_base: TreinoConfig, n_trials: int) -> dict[str, object]:
    """Roda o estudo Optuna e retreina/avalia a melhor configuracao no teste."""
    import optuna

    dados = prepara_dados(config_base)

    # Sampler com semente: a busca e reproduzivel dado o mesmo config/seed.
    sampler = optuna.samplers.TPESampler(seed=config_base.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda trial: _objetivo(trial, dados, config_base), n_trials=n_trials
    )

    melhor_config = replace(config_base, **study.best_params)
    resultado_final = treina_e_avalia(melhor_config)

    return {
        "arquitetura": config_base.arquitetura,
        "n_trials": n_trials,
        "melhores_hiperparametros": study.best_params,
        "melhor_val_mae": float(study.best_value),
        "trials": [
            {
                "numero": t.number,
                "params": t.params,
                "val_mae": None if t.value is None else float(t.value),
                "estado": t.state.name,
            }
            for t in study.trials
        ],
        "resultado_final": resultado_final,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Busca de hiperparametros (Optuna) para o pipeline de dengue 9x4."
    )
    parser.add_argument("--csv", required=True, help="Caminho para o CSV de entrada.")
    parser.add_argument(
        "--arquitetura",
        default="cnn_lstm",
        help="Arquitetura do modelo (cnn_lstm | cnn2d | efficientnet).",
    )
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument(
        "--cache-path",
        default="cache/tabela_lagged.csv",
        help="Onde salvar/ler a tabela lagged (cache).",
    )
    parser.add_argument("--lag-clima", type=int, default=45)
    parser.add_argument("--lag-historico", type=int, default=30)
    parser.add_argument(
        "--output-json",
        default="resultados_otimizacao.json",
        help="Arquivo JSON de saida com trials, melhor config e avaliacao final.",
    )
    parser.add_argument("--raio", type=int, default=4)
    parser.add_argument("--epocas-fase1", type=int, default=20)
    parser.add_argument("--epocas-fase2", type=int, default=10)
    parser.add_argument("--paciencia", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--treino-fracao", type=float, default=0.7)
    parser.add_argument("--validacao-fracao", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_base = TreinoConfig(
        csv_path=args.csv,
        cache_path=args.cache_path,
        arquitetura=args.arquitetura,
        lag_clima=args.lag_clima,
        lag_historico=args.lag_historico,
        raio=args.raio,
        epocas_fase1=args.epocas_fase1,
        epocas_fase2=args.epocas_fase2,
        paciencia_early_stopping=args.paciencia,
        seed=args.seed,
        split=SplitConfig(
            treino_fracao=args.treino_fracao,
            validacao_fracao=args.validacao_fracao,
        ),
    )
    resultado = otimiza(config_base, n_trials=args.n_trials)
    output = Path(args.output_json)
    output.write_text(json.dumps(resultado, indent=2), encoding="utf-8")
    print(f"Melhores hiperparametros: {resultado['melhores_hiperparametros']}")
    print(f"MAE validacao do melhor trial: {resultado['melhor_val_mae']:.4f}")
    print(f"Resultado salvo em: {output}")


if __name__ == "__main__":
    main()
