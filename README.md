# 🦟 Estimativa de casos de dengue

> Estimar o número diário de casos de dengue em **Cascavel-PR** a partir de clima e
> histórico de casos **defasados**, comparando arquiteturas de deep learning sob
> avaliação honesta (split temporal, baselines, sem vazamento).

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/Keras%2FTensorFlow-DL-FF6F00?logo=keras&logoColor=white">
  <img alt="Optuna" src="https://img.shields.io/badge/Optuna-tuning-2C6BB3">
  <img alt="Testes" src="https://img.shields.io/badge/testes-pytest-0A9EDC?logo=pytest&logoColor=white">
</p>

Trabalho de disciplina. Cada dia `t` é estimado a partir de uma janela das variáveis
climáticas (`Precipitação`, `Temp. média`, `Umidade`) e do histórico de casos, **todas
defasadas em ≥30 dias** — então `Qtde_Casos[t]` nunca entra na entrada (sem vazamento).
O alvo é a **log-razão de crescimento**, e não o nível absoluto, para transferir entre
regimes epidêmicos.

---

## ✨ Destaques

- **4 abordagens** sob o mesmo pipeline: `MLP` (baseline), `CNN2D`, `CNN-LSTM` e um
  **modo sequência** em que o LSTM recebe a série crua e *descobre* a defasagem.
- **Alvo em log-razão** `log1p(casos) − log1p(hist)` — invariante à escala do surto.
- **Avaliação honesta**: split temporal sem embaralhar, baselines de persistência, e o
  teste **nunca** entra na busca de hiperparâmetros (Optuna).
- **Ablations** documentadas: sazonalidade (data como feature) e ponderação de pico.
- **Artefatos organizados** em `outputs/<modelo>/` + relatório comparativo com gráficos.
- **114 testes** (os de deep learning usam `importorskip`).

## 📊 Resultado em uma linha

Com hiperparâmetros tunados, **as quatro abordagens empatam em ~29–30 de MAE** e batem
os baselines em ~25%. O **MLP — que vê só 4 números do dia `t`, sem janela — fica a ~1
MAE das CNNs**: o problema é essencialmente *tabular*, e a arquitetura sofisticada rende
pouco. O erro vive quase todo no **pico epidêmico** (limite de regime), não na temporada baixa.

| Modelo | MAE ↓ | RMSE ↓ | CC ↑ |
|---|---|---|---|
| **CNN-LSTM** | **29.2** | **77.8** | 0.66 |
| CNN2D | 29.6 | 79.6 | **0.67** |
| CNN-LSTM (sequência) | 30.0 | 80.1 | 0.64 |
| MLP (só 4 features do dia t) | 30.3 | 82.5 | 0.65 |
| baseline (média / persistência) | ~40.2 | ~99–111 | ~0–0.55 |

*(Cascavel-PR, teste = últimos 15% da série (sem embaralhar). Números reprodutíveis com
`seed 42`; regeneráveis pelo fluxo abaixo.)*

---

## 🚀 Começo rápido

```bash
# 1) ambiente
python -m venv venv
venv/bin/pip install -e ".[dev,dl,opt,report]"   # dl: tensorflow · opt: optuna · report: matplotlib

# 2) menu interativo (treinar, otimizar, relatório, resumos, testes)
./dengue

# 3) ou direto: treina e avalia uma arquitetura
venv/bin/python -m dengue_tl.experiment --csv "data/Dados 2007-2024.csv" --arquitetura cnn2d
```

> **macOS/Apple Silicon:** `import tensorflow` dá *segfault* no Python do anaconda base.
> Use o venv isolado (`venv/bin/python …`).

## 🔬 Como funciona

```
CSV bruto ─▶ tabela lagged (clima t-45, histórico t-30) ─▶ janela ─▶ split temporal
   ─▶ escala (fit só no treino) ─▶ alvo log-razão (suavizado) ─▶ arquitetura ─▶ avaliação
```

Duas formas de janelar (flag `--modo`):

| Modo | Entrada | Ideia |
|---|---|---|
| `lagged` (padrão) | matriz `9×4` de features **já defasadas** | a defasagem epidemiológica é cravada na engenharia |
| `sequencia` | janela crua dos últimos `N` dias das 4 variáveis, terminando em `t−gap` | o **LSTM descobre** qual defasagem importa |

E quatro adaptações de entrada por arquitetura (`prepara_entrada`):

| Arquitetura | Entrada | Núcleo |
|---|---|---|
| `mlp` | `(4,)` — só o dia `t` | `Dense→Dense→Dense(1)` (baseline) |
| `cnn2d` | `(4, 9, 1)` | `Conv2D×2 → Dense` (recomendação do professor) |
| `cnn_lstm` | `(9, 4)` | `Conv1D×2 → LSTM → Dense` |

## 🧪 Fluxo completo (tunar → relatórios → comparação)

```bash
CSV="data/Dados 2007-2024.csv"
# tuna cada arquitetura (grava outputs/<arch>/{otimizacao,resultado}.json)
for a in cnn2d cnn_lstm mlp; do
  venv/bin/python -m dengue_tl.tune_runner --csv "$CSV" --arquitetura $a --n-trials 40
done
# variante sequência (rótulo distinto, sem sobrescrever o cnn_lstm lagged)
venv/bin/python -m dengue_tl.tune_runner --csv "$CSV" --arquitetura cnn_lstm \
  --modo sequencia --janela-dias 30 --gap-dias 30 --rotulo cnn_lstm_sequencia --n-trials 40
# relatório individual de cada + comparação (tabela rica + 5 gráficos)
for r in cnn2d cnn_lstm mlp cnn_lstm_sequencia; do
  venv/bin/python -m dengue_tl.report --csv "$CSV" --results outputs/$r/resultado.json
done
venv/bin/python -m dengue_tl.report --csv "$CSV" --comparar
```

Tudo gerado vai para `outputs/<modelo>/` (JSONs + `relatorio/`) e `outputs/comparacao/`
— pasta ignorada pelo git (regenerável).

## 🗂️ Estrutura

```
src/dengue_tl/
├── series_loader.py      carrega/valida a série diária
├── lagged_table.py       tabela defasada (clima t-45, histórico t-30) [+ sazonalidade]
├── matrix_windower.py    janelas 9×4 (modo lagged)
├── sequence_windower.py  janela crua causal (modo sequência)
├── scaler.py             escala por-feature (fit só no treino)
├── models/               uma arquitetura por arquivo, mesma interface
│   ├── mlp.py · cnn2d.py · cnn_lstm.py
├── train_runner.py       pipeline fim-a-fim, agnóstico à arquitetura
├── tune_runner.py        busca Optuna (--metrica-busca mae|rmse)
├── experiment.py         ponto de entrada: roda + resume
├── paths.py              convenção outputs/<rótulo>/
└── report/               tabelas, gráficos e comparação entre arquiteturas
data/Dados 2007-2024.csv  Cascavel, 6575 dias diários (sem coluna de data)
tests/                    114 testes (DL usa importorskip)
```

## 🧭 Decisões metodológicas

- **Sem vazamento**: todas as features são defasadas em ≥30 dias; `Qtde_Casos[t]` jamais
  entra na entrada.
- **Sazonalidade é *ablation*, desligada por padrão** (`--sazonalidade`): dar o dia-do-ano
  deixa o modelo memorizar o atalho sazonal ("é março → pico") em vez de aprender a relação
  clima→casos. Rende MAE, mas é um atalho — mantida fora do padrão de propósito.
- **Ponderação de pico** (`peso_pico`) e **métrica de busca** (`--metrica-busca`) são
  alavancas para o gargalo comum a todos os modelos: a **amplitude do pico** sob mudança de
  regime (treino ≤ 34 casos/dia, teste até 692).

Vocabulário completo e fatos dos dados: **[`CONTEXT.md`](CONTEXT.md)**. Guia detalhado do
pipeline: **[`GUIA_PROJETO.md`](GUIA_PROJETO.md)**. Roteiro histórico da Iniciação
Científica: `roteiro.md`.

## Origem

`series_loader` e `scaler` vêm de uma Iniciação Científica (mesmo autor). O restante é
próprio deste trabalho — um pipeline único, focado numa estimativa coerente e numa
comparação honesta entre arquiteturas.
