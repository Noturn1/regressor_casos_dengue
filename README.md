# Estimativa de casos de dengue com CNN-LSTM / CNN2D

Trabalho de disciplina: estimar `Qtde_Casos` de dengue de um dia a partir de uma janela
**9×4** (9 dias × 4 features defasadas), com CNN-LSTM (padrão) ou CNN2D.

- **O que fazer, passo a passo:** `roteiro.md`
- **Vocabulário e fatos dos dados:** `CONTEXT.md`

## Estrutura

```
src/dengue_tl/
  ├── series_loader.py   ← carrega/valida a série
  ├── scaler.py          ← log1p do alvo, min/max só do treino
  ├── window_builder.py  ← trilha GASF univariada legada
  ├── lagged_table.py    ← tabela defasada (clima t-45, histórico t-30) + sazonalidade + cache
  ├── matrix_windower.py ← janelas 9×4 da tabela lagged
  ├── models/            ← arquiteturas separadas por arquivo
  │   ├── __init__.py    ← seleciona_arquitetura (import preguiçoso)
  │   ├── cnn_lstm.py    ← Conv1D + LSTM sobre (9, 4) — padrão
  │   └── cnn2d.py       ← Conv2D pura sobre (4, 9, 1)
  ├── train_runner.py    ← pipeline 9×4 fim-a-fim, agnóstico à arquitetura
  └── experiment.py      ← ponto de entrada central: roda + resume resultados
tests/                   ← rodar pelo venv; os de DL usam importorskip
data/
  ├── AmostraDados.csv       ← amostra p/ desenvolvimento (9 dias, sem coluna de data)
  └── Dados 2007-2024.csv    ← dataset completo de Cascavel (6575 dias, sem coluna de data)
```

### Rodar (menu interativo)

```bash
./dengue
# menu com: treinar (cnn_lstm | cnn2d), otimizar hiperparâmetros (Optuna),
# gerar relatório (tabelas + gráficos), ver resumo de resultados, rodar testes
```

### Rodar o pipeline direto (sem menu)

```bash
venv/bin/python -m dengue_tl.experiment --csv "data/Dados 2007-2024.csv"
# padrão: cnn_lstm; para a CNN2D: --arquitetura cnn2d
# gera cache/tabela_lagged.csv e outputs/<arquitetura>/resultado.json (métricas + predições + resumo)
# sazonalidade (dia-do-ano) é ablation opt-in: --sazonalidade (desligada por padrão)
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev,dl,opt,report]"   # dl: tensorflow; opt: optuna; report: matplotlib
venv/bin/python -m pytest    # os testes de DL usam importorskip
```

> **macOS/Apple Silicon:** `import tensorflow` dá *segfault* no Python 3.13 do anaconda base.
> Use um venv isolado e rode os testes por ele: `venv/bin/python -m pytest`.

## Origem

`series_loader` e `scaler` vêm de uma Iniciação Científica (mesmo autor). O restante é
próprio deste trabalho. A IC compara 12 representações × cabeças recorrentes; **este
trabalho não** — é um pipeline único focado numa estimativa coerente.
