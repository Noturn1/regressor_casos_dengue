"""Ponto de entrada central: roda o pipeline no dataset e resume o essencial.

Amarra tudo numa chamada só:
1) treina e avalia via `train_runner.treina_e_avalia` (pipeline 9x4 completo);
2) destila o dict grande de resultado nos **resultados mais pertinentes**
   (`resumo_pertinente`): tamanhos do split, tabela de métricas modelo × baselines,
   melhor estimador por métrica, ganho do modelo e melhor época de validação;
3) imprime o resumo legível e persiste o resultado completo em JSON.

O resultado completo (retornado por `roda_experimento`) é o artefato que o **futuro
módulo de visualizações** vai consumir — ele já traz os três blocos prontos pra plot:
`historico` (curvas de treino), `predicoes_teste` (y_true × y_pred) e `metricas`
(barras comparativas). A função `gera_figuras` é a costura: quando existir um
`dengue_tl/visualizations.py`, ela o chama; enquanto não existir, é no-op.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dengue_tl.train_runner import (
    SplitConfig,
    TreinoConfig,
    treina_e_avalia,
)

# Métricas em que "menor é melhor" (erro) vs. "maior é melhor" (correlação).
_MENOR_MELHOR = ("mae", "rmse")
_MAIOR_MELHOR = ("cc",)


def _tamanho(par: list[int]) -> int:
    inicio, fim = par
    return fim - inicio


def _melhor_epoca_validacao(historico: dict) -> dict:
    """Melhor época pelo menor `val_loss`, varrendo todas as fases do histórico.

    Agnóstico ao modelo: funciona tanto para treino em fase única (`{"treino": ...}`,
    ex. CNN-LSTM) quanto em 2 fases (`{"fase1": ..., "fase2": ...}`, ex. EfficientNet).
    """
    melhor = {"fase": None, "epoca": None, "val_loss": float("inf")}
    for fase, curvas in historico.items():
        for i, v in enumerate(curvas.get("val_loss", [])):
            if v < melhor["val_loss"]:
                melhor = {"fase": fase, "epoca": i, "val_loss": float(v)}
    return melhor


def resumo_pertinente(resultado: dict) -> dict:
    """Destila o dict completo nos resultados que interessam para decidir.

    Não depende de TensorFlow — opera só sobre o dict serializável.
    """
    split = resultado["split"]
    metricas = resultado["metricas"]
    estimadores = list(metricas)  # modelo, baseline_media, baseline_historico

    melhor_por_metrica: dict[str, str] = {}
    for metrica in (*_MENOR_MELHOR, *_MAIOR_MELHOR):
        chave = (
            max if metrica in _MAIOR_MELHOR else min
        )
        melhor_por_metrica[metrica] = chave(
            estimadores, key=lambda e: metricas[e][metrica]
        )

    # Ganho do modelo em MAE frente ao melhor baseline (o alvo a bater).
    baselines = [e for e in estimadores if e != "modelo"]
    melhor_baseline = min(baselines, key=lambda e: metricas[e]["mae"])
    mae_modelo = metricas["modelo"]["mae"]
    mae_baseline = metricas[melhor_baseline]["mae"]
    ganho_mae_pct = (
        (mae_baseline - mae_modelo) / mae_baseline * 100.0
        if mae_baseline
        else float("nan")
    )

    return {
        "arquitetura": resultado.get("config", {}).get("arquitetura", "?"),
        "amostras": {
            "treino": _tamanho(split["treino"]),
            "validacao": _tamanho(split["validacao"]),
            "teste": _tamanho(split["teste"]),
            "total": split["teste"][1],
        },
        "metricas": metricas,
        "melhor_por_metrica": melhor_por_metrica,
        "modelo_bate_baseline": mae_modelo < mae_baseline,
        "ganho_mae_pct_vs_melhor_baseline": ganho_mae_pct,
        "melhor_baseline": melhor_baseline,
        "melhor_epoca_validacao": _melhor_epoca_validacao(resultado["historico"]),
    }


def formata_resumo(resumo: dict) -> str:
    """Resumo pertinente como texto de terminal legível."""
    a = resumo["amostras"]
    linhas = [
        "=" * 56,
        f"RESULTADOS — pipeline 9x4 ({resumo['arquitetura']})",
        "=" * 56,
        f"Amostras: {a['total']} (treino {a['treino']} / "
        f"val {a['validacao']} / teste {a['teste']})",
        "",
        f"{'estimador':<20}{'MAE':>10}{'RMSE':>10}{'CC':>10}",
        "-" * 50,
    ]
    for estimador, m in resumo["metricas"].items():
        linhas.append(
            f"{estimador:<20}{m['mae']:>10.3f}{m['rmse']:>10.3f}{m['cc']:>10.3f}"
        )

    mb = resumo["melhor_por_metrica"]
    ganho = resumo["ganho_mae_pct_vs_melhor_baseline"]
    veredito = (
        f"supera '{resumo['melhor_baseline']}' em {ganho:.1f}% de MAE"
        if resumo["modelo_bate_baseline"]
        else f"NÃO supera '{resumo['melhor_baseline']}' ({ganho:.1f}% de MAE)"
    )
    ep = resumo["melhor_epoca_validacao"]
    linhas += [
        "-" * 50,
        f"Melhor por métrica: MAE→{mb['mae']}  RMSE→{mb['rmse']}  CC→{mb['cc']}",
        f"Veredito: modelo {veredito}.",
        f"Melhor validação: {ep['fase']} época {ep['epoca']} "
        f"(val_loss {ep['val_loss']:.4f}).",
        "=" * 56,
    ]
    return "\n".join(linhas)


def gera_figuras(resultado: dict, outdir: str | Path) -> list[Path]:
    """Costura para o futuro módulo de visualizações.

    Quando existir `dengue_tl/visualizations.py` expondo `gera_todas(resultado,
    outdir) -> list[Path]`, esta função o invoca. Enquanto não existir, retorna
    lista vazia (no-op) — assim o pipeline central não depende do módulo de plots.
    """
    try:
        from dengue_tl import visualizations
    except ImportError:
        return []
    Path(outdir).mkdir(parents=True, exist_ok=True)
    return list(visualizations.gera_todas(resultado, outdir))


def roda_experimento(config: TreinoConfig) -> dict:
    """Executa o pipeline completo e devolve o dict de resultado (com `resumo`)."""
    resultado = treina_e_avalia(config)
    resultado["resumo"] = resumo_pertinente(resultado)
    return resultado


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roda o pipeline de dengue no dataset e resume os resultados pertinentes."
    )
    parser.add_argument("--csv", required=True, help="Caminho para o CSV de entrada.")
    parser.add_argument(
        "--arquitetura",
        default="cnn_lstm",
        help="Arquitetura do modelo (cnn_lstm | efficientnet).",
    )
    parser.add_argument("--cache-path", default="cache/tabela_lagged.csv")
    parser.add_argument("--lag-clima", type=int, default=45)
    parser.add_argument("--lag-historico", type=int, default=30)
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
    parser.add_argument(
        "--output-json",
        default="resultados_experimento.json",
        help="Arquivo JSON com o resultado completo (artefato para as visualizações).",
    )
    parser.add_argument(
        "--figuras-dir",
        default=None,
        help="Se informado, chama o módulo de visualizações (quando existir) nessa pasta.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = TreinoConfig(
        csv_path=args.csv,
        cache_path=args.cache_path,
        arquitetura=args.arquitetura,
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

    resultado = roda_experimento(config)

    print(formata_resumo(resultado["resumo"]))

    saida = Path(args.output_json)
    saida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultado completo salvo em: {saida}")

    if args.figuras_dir:
        figuras = gera_figuras(resultado, args.figuras_dir)
        if figuras:
            print(f"Figuras geradas: {', '.join(str(f) for f in figuras)}")
        else:
            print("Módulo de visualizações ainda não disponível (nenhuma figura gerada).")


if __name__ == "__main__":
    main()
