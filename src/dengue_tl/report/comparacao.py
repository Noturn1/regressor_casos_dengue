"""Comparativo completo entre arquiteturas: tabela rica + graficos.

Reproduz, como parte do report, a "varredura final": descobre os resultados
tunados de cada arquitetura em `outputs/` (ver `descobre_resultados_por_arquitetura`),
monta uma tabela rica (MAE/RMSE/CC bruto e suavizado, erro por faixa, amplitude
de pico, ganho vs baseline) e gera os graficos comparativos. Nao re-treina nada:
consome os JSON de resultado ja persistidos.

Saidas em `outputs/comparacao/`:
  - tabela_comparacao_final.csv    (uma linha por metodo + baselines)
  - predicoes_comparacao.json      ({metodo: {y_true, y_pred}}, p/ novas viz)
  - cmp_real_vs_previsto.png / cmp_dispersao.png / cmp_mae_por_faixa.png /
    cmp_metricas.png / cmp_amplitude_pico.png
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from dengue_tl.report.artifacts import descobre_resultados_por_arquitetura
from dengue_tl.report.data_io import (
    DEFAULT_DPI,
    DEFAULT_OUTPUT_DIR,
    ensure_output_dir,
    load_training_results,
)

_COLUNAS = [
    "metodo", "arquitetura", "modo", "MAE", "RMSE", "CC",
    "MAE_suav", "RMSE_suav", "CC_suav", "MAE_baixa_lt10", "n_baixa",
    "MAE_pico_ge10", "n_pico", "pred_max", "true_max",
    "ganho_mae_pct_vs_baseline", "n_teste",
]


def _desembrulha(res: dict) -> dict:
    """Extrai `resultado_final` de JSONs de otimizacao; passa treino direto."""
    return res.get("resultado_final", res)


def _linha_rica(label: str, res: dict):
    m = res["metricas"]["modelo"]
    ms = res.get("metricas_alvo_suavizado", {}).get("modelo", {})
    bm, bh = res["metricas"]["baseline_media"], res["metricas"]["baseline_historico"]
    cfg = res.get("config", {})
    yt = np.asarray(res["predicoes_teste"]["y_true"], dtype=float)
    yp = np.asarray(res["predicoes_teste"]["y_pred_modelo"], dtype=float)
    baixa, pico = yt < 10, yt >= 10
    melhor_base = min(bm["mae"], bh["mae"])

    def _mae(mask):
        return round(float(np.abs(yt[mask] - yp[mask]).mean()), 3) if mask.any() else None

    linha = {
        "metodo": label,
        "arquitetura": cfg.get("arquitetura", "?"),
        "modo": cfg.get("modo", "lagged"),
        "MAE": round(m["mae"], 3), "RMSE": round(m["rmse"], 3), "CC": round(m["cc"], 4),
        "MAE_suav": round(ms["mae"], 3) if ms else None,
        "RMSE_suav": round(ms["rmse"], 3) if ms else None,
        "CC_suav": round(ms["cc"], 4) if ms else None,
        "MAE_baixa_lt10": _mae(baixa), "n_baixa": int(baixa.sum()),
        "MAE_pico_ge10": _mae(pico), "n_pico": int(pico.sum()),
        "pred_max": round(float(yp.max()), 1), "true_max": round(float(yt.max()), 1),
        "ganho_mae_pct_vs_baseline": round((melhor_base - m["mae"]) / melhor_base * 100, 1),
        "n_teste": int(len(yt)),
    }
    return linha, yt, yp


def _linhas_baseline(res: dict) -> list[dict]:
    yt = np.asarray(res["predicoes_teste"]["y_true"], dtype=float)
    linhas = []
    for nome in ("baseline_media", "baseline_historico"):
        b = res["metricas"][nome]
        linhas.append({
            **{c: None for c in _COLUNAS},
            "metodo": nome, "arquitetura": "-", "modo": "-",
            "MAE": round(b["mae"], 3), "RMSE": round(b["rmse"], 3), "CC": round(b["cc"], 4),
            "true_max": round(float(yt.max()), 1),
            "ganho_mae_pct_vs_baseline": 0.0, "n_teste": int(len(yt)),
        })
    return linhas


def _plots(predicoes: dict, linhas: list[dict], out: Path, dpi: int) -> dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modelos = [l for l in linhas if l["MAE_pico_ge10"] is not None]
    nomes_m = [l["metodo"] for l in modelos]
    art: dict[str, Path] = {}

    def _grade(n):
        ncols = 2 if n > 1 else 1
        return math.ceil(n / ncols), ncols

    # 1) Real x Previsto (small multiples)
    nr, nc = _grade(len(nomes_m))
    fig, axes = plt.subplots(nr, nc, figsize=(7.5 * nc, 3.4 * nr), squeeze=False)
    for ax, k in zip(axes.flat, nomes_m):
        yt = np.array(predicoes[k]["y_true"]); yp = np.array(predicoes[k]["y_pred"])
        ax.plot(yt, color="0.15", lw=0.9, label="real")
        ax.plot(yp, color="tab:red", lw=0.9, alpha=0.85, label="previsto")
        ax.set_title(k); ax.set_ylabel("casos/dia"); ax.legend(fontsize=8)
    for ax in axes.flat[len(nomes_m):]:
        ax.axis("off")
    fig.suptitle("Real x Previsto no teste")
    fig.tight_layout()
    p = out / "cmp_real_vs_previsto.png"; fig.savefig(p, dpi=dpi); plt.close(fig)
    art["cmp_real_vs_previsto"] = p

    # 2) Dispersao real x previsto
    fig, axes = plt.subplots(nr, nc, figsize=(6 * nc, 5 * nr), squeeze=False)
    for ax, k in zip(axes.flat, nomes_m):
        yt = np.array(predicoes[k]["y_true"]); yp = np.array(predicoes[k]["y_pred"])
        ax.scatter(yt, yp, s=7, alpha=0.35, color="tab:blue")
        lim = float(max(yt.max(), yp.max()))
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_title(f"{k} (CC={np.corrcoef(yt, yp)[0,1]:.3f})")
        ax.set_xlabel("real"); ax.set_ylabel("previsto")
    for ax in axes.flat[len(nomes_m):]:
        ax.axis("off")
    fig.suptitle("Dispersao real x previsto")
    fig.tight_layout()
    p = out / "cmp_dispersao.png"; fig.savefig(p, dpi=dpi); plt.close(fig)
    art["cmp_dispersao"] = p

    # 3) MAE por faixa
    x = np.arange(len(modelos)); w = 0.38
    fig, ax = plt.subplots(figsize=(1.6 * len(modelos) + 3, 5))
    ax.bar(x - w / 2, [l["MAE_baixa_lt10"] for l in modelos], w, label="baixa (<10)", color="tab:green")
    ax.bar(x + w / 2, [l["MAE_pico_ge10"] for l in modelos], w, label="pico (>=10)", color="tab:red")
    ax.set_xticks(x); ax.set_xticklabels(nomes_m, rotation=15, ha="right")
    ax.set_ylabel("MAE"); ax.set_title("MAE por faixa de incidencia"); ax.legend()
    fig.tight_layout()
    p = out / "cmp_mae_por_faixa.png"; fig.savefig(p, dpi=dpi); plt.close(fig)
    art["cmp_mae_por_faixa"] = p

    # 4) MAE / RMSE por metodo (+ baselines)
    nomes_all = [l["metodo"] for l in linhas]
    x = np.arange(len(linhas)); w = 0.4
    cores = ["tab:blue" if l["MAE_pico_ge10"] is not None else "0.6" for l in linhas]
    fig, ax = plt.subplots(figsize=(1.4 * len(linhas) + 3, 5))
    ax.bar(x - w / 2, [l["MAE"] for l in linhas], w, label="MAE", color=cores)
    ax.bar(x + w / 2, [l["RMSE"] for l in linhas], w, label="RMSE", color=cores, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(nomes_all, rotation=20, ha="right")
    ax.set_ylabel("erro"); ax.set_title("MAE / RMSE por metodo (cinza = baseline)"); ax.legend()
    fig.tight_layout()
    p = out / "cmp_metricas.png"; fig.savefig(p, dpi=dpi); plt.close(fig)
    art["cmp_metricas"] = p

    # 5) Amplitude do pico
    tmax = modelos[0]["true_max"]
    fig, ax = plt.subplots(figsize=(1.6 * len(modelos) + 3, 5))
    ax.bar(x[:len(modelos)], [l["pred_max"] for l in modelos], color="tab:purple", alpha=0.85)
    ax.axhline(tmax, color="black", ls="--", lw=1.5, label=f"pico real = {tmax:.0f}")
    ax.set_xticks(x[:len(modelos)]); ax.set_xticklabels(nomes_m, rotation=15, ha="right")
    ax.set_ylabel("previsao maxima"); ax.set_title("Amplitude alcancada no pico"); ax.legend()
    fig.tight_layout()
    p = out / "cmp_amplitude_pico.png"; fig.savefig(p, dpi=dpi); plt.close(fig)
    art["cmp_amplitude_pico"] = p
    return art


def gera_comparacao_completa(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR, dpi: int = DEFAULT_DPI
) -> dict[str, Path]:
    """Tabela rica + predicoes + graficos, a partir dos resultados em outputs/."""
    resultados = descobre_resultados_por_arquitetura(output_dir)
    if len(resultados) < 2:
        raise SystemExit(
            f"Comparacao completa exige >= 2 arquiteturas em {output_dir}/. "
            f"Encontradas: {sorted(resultados) or 'nenhuma'}."
        )
    out = ensure_output_dir(Path(output_dir) / "comparacao")

    linhas, predicoes, primeiro = [], {}, None
    for label, caminho in resultados.items():
        res = _desembrulha(load_training_results(caminho))
        linha, yt, yp = _linha_rica(label, res)
        linhas.append(linha)
        predicoes[label] = {"y_true": yt.tolist(), "y_pred": yp.tolist()}
        primeiro = primeiro or res
    linhas += _linhas_baseline(primeiro)

    art: dict[str, Path] = {}
    csv_path = out / "tabela_comparacao_final.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUNAS)
        w.writeheader()
        w.writerows(linhas)
    art["tabela_comparacao_final"] = csv_path

    pred_path = out / "predicoes_comparacao.json"
    pred_path.write_text(json.dumps(predicoes), encoding="utf-8")
    art["predicoes_comparacao"] = pred_path

    art.update(_plots(predicoes, linhas, out, dpi))
    return art
