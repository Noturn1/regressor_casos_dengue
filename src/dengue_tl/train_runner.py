"""Runner de treino/validacao do pipeline de regressao de dengue (entrada 9x4).

Amarra as etapas do pipeline atual:
1) constroi (ou carrega do cache) a tabela lagged: clima em t-45, historico em t-30, alvo em t
2) janela as features em matrizes 9x4 (dia central = alvo)
3) split temporal treino/validacao/teste (sem embaralhar)
4) escala as features por-coluna (fit so no treino) e adapta a entrada a arquitetura
5) alvo: suaviza (media movel 7d centrada) e transforma em log-razao sobre o
   historico (ver TransformadorAlvo) — ou log1p puro com --alvo nivel
6) treina a arquitetura escolhida (dengue_tl.models)
7) avalia modelo e baselines (media do treino / historico t-30) na escala
   original bruta e contra o alvo suavizado
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
    arquitetura: str = "cnn_lstm"  # ver dengue_tl.models.ARQUITETURAS
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
    filtros: int = 32
    unidades_lstm: int = 32
    dense_unidades: int = 64
    dropout: float = 0.3
    # Formulacao do alvo. "razao": log1p(casos) - log1p(historico) — o modelo
    # preve o crescimento em ~30 dias, invariante a escala do surto; a
    # persistencia vira exatamente "prever 0". "nivel": log1p(casos) puro
    # (formulacao antiga; nao transfere entre regimes epidemicos).
    alvo: str = "razao"
    # Media movel centrada aplicada ao alvo (e ao historico usado na razao)
    # antes do treino/avaliacao: o serrilhado semanal de notificacao e ruido
    # de registro, nao sinal. 0 ou 1 desativam.
    suavizacao_alvo: int = 7
    # Sazonalidade: adiciona sin/cos do dia-do-ano do dia central como features
    # (ver LaggedTableConfig). Ligado por padrao — a dengue e fortemente sazonal
    # (jan-mai concentra ~94% dos casos) e o calendario e futuro conhecido, nao
    # vazamento.
    sazonalidade: bool = True
    data_inicial: str = "2007-01-01"
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


def _cache_path_efetivo(config: TreinoConfig) -> str:
    """Cache separado quando ha sazonalidade: evita reusar silenciosamente uma
    tabela de 4 features (sem sin/cos) quando a flag esta ligada, e vice-versa."""
    if not config.sazonalidade:
        return config.cache_path
    p = Path(config.cache_path)
    return str(p.with_name(f"{p.stem}_sazonal{p.suffix}"))


def carrega_tabela_lagged(config: TreinoConfig):
    """Constroi (ou carrega do cache) a tabela lagged a partir do CSV.

    Dateless: o dataset completo nao tem coluna de data; `build_lagged_table`
    aplica os lags posicionalmente. O cache evita recomputar os lags.
    """
    return build_or_load_lagged_table(
        config.csv_path,
        _cache_path_efetivo(config),
        LaggedTableConfig(
            lag_clima=config.lag_clima,
            lag_historico=config.lag_historico,
            sazonalidade=config.sazonalidade,
            data_inicial=config.data_inicial,
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


def _media_movel_centrada(v: np.ndarray, janela: int) -> np.ndarray:
    """Media movel centrada sobre uma serie diaria consecutiva (bordas: parcial).

    As janelas 9x4 deslizam de 1 em 1 dia, entao `y` e `historico` sao series
    diarias consecutivas — o rolling e valido.
    """
    if janela <= 1:
        return np.asarray(v, dtype=float)
    import pandas as pd

    return (
        pd.Series(np.asarray(v, dtype=float))
        .rolling(janela, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


@dataclass(frozen=True)
class TransformadorAlvo:
    """Transforma casos <-> espaco do modelo, conforme a formulacao do alvo.

    - `alvo="razao"`: o modelo aprende `log1p(casos) - log1p(historico)` (log da
      razao de crescimento em ~30 dias). Invariante a escala: um surto 30->90 no
      treino ensina o mesmo padrao que 200->600 no teste. Prever 0 == baseline
      de persistencia, entao qualquer aprendizado e ganho sobre o baseline.
    - `alvo="nivel"`: `log1p(casos)` puro (formulacao antiga).

    `teto_log` e SO anti-explosao numerica (uma extrapolacao em log explode no
    expm1: log 20 -> ~5e8 casos). E generoso de proposito — nao e um teto
    epidemiologico: limitar ao maximo do treino tornaria impossivel prever
    surtos maiores que os ja vistos.
    """

    alvo: str
    teto_log: float

    def transforma(self, casos: np.ndarray, historico: np.ndarray) -> np.ndarray:
        y_log = np.log1p(np.asarray(casos, dtype=float))
        if self.alvo == "razao":
            return y_log - np.log1p(np.asarray(historico, dtype=float))
        return y_log

    def inverte(self, y_modelo: np.ndarray, historico: np.ndarray) -> np.ndarray:
        log_casos = np.asarray(y_modelo, dtype=float)
        if self.alvo == "razao":
            log_casos = log_casos + np.log1p(np.asarray(historico, dtype=float))
        return np.expm1(np.clip(log_casos, 0.0, self.teto_log))


def preve_casos(
    model, x, transformador: TransformadorAlvo, historico: np.ndarray
) -> np.ndarray:
    """Prediz no espaco do modelo e inverte para a escala de casos (com clip)."""
    y_pred = model.predict(x, verbose=0).reshape(-1)
    return transformador.inverte(y_pred, historico)


@dataclass(frozen=True)
class DadosPreparados:
    """Dados do pipeline ja janelados, divididos, escalados e codificados.

    Materializa tudo que vem ANTES do treino, para que o `tune_runner` prepare
    os dados uma unica vez e rode varios trials de treino sobre eles (a
    codificacao em imagem da EfficientNet e cara para repetir por trial).
    """

    x_treino: np.ndarray
    x_val: np.ndarray
    x_teste: np.ndarray
    y_treino: np.ndarray  # alvo no espaco do modelo (ver TransformadorAlvo)
    y_val: np.ndarray  # alvo no espaco do modelo
    y_treino_raw: np.ndarray
    y_val_raw: np.ndarray
    y_teste_raw: np.ndarray
    y_teste_suave: np.ndarray  # alvo do teste suavizado (== raw se suavizacao <= 1)
    hist_val: np.ndarray  # historico suavizado do dia central (p/ inverter a razao)
    hist_teste: np.ndarray
    transformador: TransformadorAlvo
    baseline_media_pred: np.ndarray
    baseline_historico_pred: np.ndarray
    split: tuple[slice, slice, slice]


def prepara_dados(config: TreinoConfig) -> DadosPreparados:
    """Tabela lagged -> janelas 9x4 -> split temporal -> escala -> entrada da rede."""
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
    y_val_raw = y[val_sl].astype(float)
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

    # Alvo do modelo: casos e historico do dia central suavizados (media movel
    # centrada, ruido semanal de notificacao), depois log-razao ou log1p.
    # A suavizacao do historico usa valores com >= lag-3 dias de idade: sem
    # vazamento do presente.
    hist_raw = X[:, config.raio, idx_historico].astype(float)
    y_suave = _media_movel_centrada(y.astype(float), config.suavizacao_alvo)
    hist_suave = _media_movel_centrada(hist_raw, config.suavizacao_alvo)

    # Teto SO anti-explosao (~3 ordens de grandeza acima do maior surto do
    # treino); ver TransformadorAlvo.
    teto_log = float(np.log1p(1000.0 * max(1.0, y_treino_raw.max())))
    transformador = TransformadorAlvo(alvo=config.alvo, teto_log=teto_log)

    y_treino = transformador.transforma(y_suave[treino_sl], hist_suave[treino_sl])
    y_val = transformador.transforma(y_suave[val_sl], hist_suave[val_sl])

    # Import lazy: permite testar a logica (split/baselines) sem dependencias de DL.
    # Cada arquitetura mora em seu arquivo (dengue_tl/models/) e implementa a
    # mesma interface: prepara_entrada + treina + espaco_busca. Ver dengue_tl.models.
    from dengue_tl.models import seleciona_arquitetura

    modulo = seleciona_arquitetura(config.arquitetura)
    entradas = modulo.prepara_entrada(X_escalado)

    return DadosPreparados(
        x_treino=entradas[treino_sl],
        x_val=entradas[val_sl],
        x_teste=entradas[teste_sl],
        y_treino=y_treino,
        y_val=y_val,
        y_treino_raw=y_treino_raw,
        y_val_raw=y_val_raw,
        y_teste_raw=y_teste_raw,
        y_teste_suave=y_suave[teste_sl],
        hist_val=hist_suave[val_sl],
        hist_teste=hist_suave[teste_sl],
        transformador=transformador,
        baseline_media_pred=baseline_media_pred,
        baseline_historico_pred=baseline_historico_pred,
        split=(treino_sl, val_sl, teste_sl),
    )


def treina_e_avalia(config: TreinoConfig) -> dict[str, object]:
    """Executa treino em 2 fases e avalia modelo + baselines."""
    _set_semente(config.seed)

    dados = prepara_dados(config)
    treino_sl, val_sl, teste_sl = dados.split
    y_teste_raw = dados.y_teste_raw
    baseline_media_pred = dados.baseline_media_pred
    baseline_historico_pred = dados.baseline_historico_pred

    from dengue_tl.models import seleciona_arquitetura

    modulo = seleciona_arquitetura(config.arquitetura)
    model, historico = modulo.treina(
        dados.x_treino, dados.y_treino, dados.x_val, dados.y_val, config
    )

    y_pred = preve_casos(model, dados.x_teste, dados.transformador, dados.hist_teste)

    resultado: dict[str, object] = {
        "config": asdict(config),
        "split": {
            "treino": [treino_sl.start, treino_sl.stop],
            "validacao": [val_sl.start, val_sl.stop],
            "teste": [teste_sl.start, teste_sl.stop],
        },
        "historico": historico,
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

    # Metricas tambem contra o alvo suavizado: o serrilhado semanal do dado
    # bruto e ruido de notificacao irredutivel para features com lag >= 30d.
    if config.suavizacao_alvo > 1:
        y_teste_suave = dados.y_teste_suave
        resultado["metricas_alvo_suavizado"] = {
            "modelo": calcula_metricas(y_teste_suave, y_pred),
            "baseline_media": calcula_metricas(y_teste_suave, baseline_media_pred),
            "baseline_historico": calcula_metricas(
                y_teste_suave, baseline_historico_pred
            ),
        }
        resultado["predicoes_teste"]["y_true_suave"] = [
            float(v) for v in y_teste_suave
        ]

    return resultado


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina e avalia o pipeline de regressao de dengue (entrada 9x4)."
    )
    parser.add_argument("--csv", required=True, help="Caminho para o CSV de entrada.")
    parser.add_argument(
        "--arquitetura",
        default="cnn_lstm",
        help="Arquitetura do modelo (cnn_lstm | cnn2d | efficientnet).",
    )
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
    parser.add_argument("--filtros", type=int, default=32)
    parser.add_argument("--unidades-lstm", type=int, default=32)
    parser.add_argument("--dense-unidades", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--alvo",
        default="razao",
        choices=("razao", "nivel"),
        help="Formulacao do alvo: log-razao sobre o historico (padrao) ou log1p puro.",
    )
    parser.add_argument(
        "--suavizacao-alvo",
        type=int,
        default=7,
        help="Janela da media movel centrada do alvo (0/1 desativa).",
    )
    parser.add_argument(
        "--sazonalidade",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adiciona sin/cos do dia-do-ano como features (--no-sazonalidade desativa).",
    )
    parser.add_argument(
        "--data-inicial",
        default="2007-01-01",
        help="Data do 1o registro (serie dateless), base do calendario sazonal.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--treino-fracao", type=float, default=0.7)
    parser.add_argument("--validacao-fracao", type=float, default=0.15)
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
        filtros=args.filtros,
        unidades_lstm=args.unidades_lstm,
        dense_unidades=args.dense_unidades,
        dropout=args.dropout,
        alvo=args.alvo,
        suavizacao_alvo=args.suavizacao_alvo,
        sazonalidade=args.sazonalidade,
        data_inicial=args.data_inicial,
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
