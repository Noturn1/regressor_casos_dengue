"""Busca de hiperparametros com Optuna sobre o pipeline 9x4.

Para cada trial, o Optuna (sampler TPE) sugere uma configuracao a partir do
`espaco_busca` da arquitetura escolhida (ver dengue_tl.models), treina o modelo
e mede o MAE **na validacao, em escala original de casos**. O conjunto de teste
nunca entra na busca: ele so e usado uma vez, no retreino final com a melhor
configuracao (via `treina_e_avalia`), para que a metrica reportada continue
honesta.

Os dados sao preparados uma unica vez (`prepara_dados`) e reutilizados por
todos os trials. O `peso_pico` (peso de amostra) entra no `espaco_busca`, entao
varia por trial: o `sample_weight` e recalculado a cada treino a partir do
`nivel_treino` guardado — barato. Requer os extras `dl` (tensorflow) e `optuna`.

Uso:
    python -m dengue_tl.tune_runner --csv data/AmostraDados.csv \
        --arquitetura cnn_lstm --n-trials 50
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from dengue_tl.paths import caminho_otimizacao, garante_pai
from dengue_tl.train_runner import (
    DadosPreparados,
    SplitConfig,
    TreinoConfig,
    mae,
    pesos_por_nivel,
    prepara_dados,
    preve_casos,
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
        dados.x_treino,
        dados.y_treino,
        dados.x_val,
        dados.y_val,
        config,
        sample_weight=pesos_por_nivel(dados.nivel_treino, config.peso_pico),
    )

    y_pred = preve_casos(model, dados.x_val, dados.transformador, dados.hist_val)
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
        help="Arquitetura do modelo (cnn_lstm | cnn2d | mlp).",
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
        "--sazonalidade",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ablation: adiciona sin/cos do dia-do-ano como features (desligado por padrao).",
    )
    parser.add_argument(
        "--data-inicial",
        default="2007-01-01",
        help="Data do 1o registro (serie dateless), base do calendario sazonal.",
    )
    parser.add_argument(
        "--peso-pico",
        type=float,
        default=0.0,
        help="Peso extra dos dias de alto numero de casos na loss (0 desativa).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="JSON com trials/melhor config. Padrao: outputs/<arquitetura>/otimizacao.json.",
    )
    parser.add_argument("--raio", type=int, default=4)
    parser.add_argument(
        "--modo",
        default="lagged",
        choices=("lagged", "sequencia"),
        help="Preparacao: lagged (matriz 9x4 defasada) ou sequencia (janela crua p/ o LSTM).",
    )
    parser.add_argument("--janela-dias", type=int, default=60)
    parser.add_argument("--gap-dias", type=int, default=30)
    parser.add_argument("--epocas", type=int, default=30)
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
        sazonalidade=args.sazonalidade,
        data_inicial=args.data_inicial,
        peso_pico=args.peso_pico,
        raio=args.raio,
        modo=args.modo,
        janela_dias=args.janela_dias,
        gap_dias=args.gap_dias,
        epocas=args.epocas,
        paciencia_early_stopping=args.paciencia,
        seed=args.seed,
        split=SplitConfig(
            treino_fracao=args.treino_fracao,
            validacao_fracao=args.validacao_fracao,
        ),
    )
    resultado = otimiza(config_base, n_trials=args.n_trials)
    output = garante_pai(args.output_json or caminho_otimizacao(config_base.arquitetura))
    output.write_text(json.dumps(resultado, indent=2), encoding="utf-8")
    print(f"Melhores hiperparametros: {resultado['melhores_hiperparametros']}")
    print(f"MAE validacao do melhor trial: {resultado['melhor_val_mae']:.4f}")
    print(f"Resultado salvo em: {output}")


if __name__ == "__main__":
    main()
