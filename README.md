# Estimativa de casos de dengue com CNN-LSTM / EfficientNet

Trabalho de disciplina: estimar `Qtde_Casos` de dengue de um dia a partir de uma janela
**9×4** (9 dias × 4 features defasadas), com CNN-LSTM (padrão) ou EfficientNet-B0.

- **O que fazer, passo a passo:** `roteiro.md`
- **Vocabulário e fatos dos dados:** `CONTEXT.md`

## Estrutura

```
src/dengue_tl/
  ├── series_loader.py   ← carrega/valida a série
  ├── scaler.py          ← log1p do alvo, min/max só do treino
  ├── window_builder.py  ← trilha GASF univariada legada
  ├── lagged_table.py    ← tabela defasada (clima t-45, histórico t-30) + cache
  ├── matrix_windower.py ← janelas 9×4 da tabela lagged
  ├── encoder.py         ← encode_gasf e encode_matrix (9×4 → 100×100×3)
  ├── models/            ← arquiteturas separadas por arquivo
  │   ├── __init__.py    ← seleciona_arquitetura (import preguiçoso)
  │   ├── cnn_lstm.py    ← Conv1D + LSTM sobre (9, 4) — padrão
  │   └── efficientnet.py← EfficientNet-B0 com transfer learning
  ├── train_runner.py    ← pipeline 9×4 fim-a-fim, agnóstico à arquitetura
  └── experiment.py      ← ponto de entrada central: roda + resume resultados
tests/                   ← 60 testes passam (rodar pelo venv; os de DL usam importorskip)
data/
  ├── AmostraDados.csv       ← amostra p/ desenvolvimento (9 dias, sem coluna de data)
  └── Dados 2007-2024.csv    ← dataset completo de Cascavel (6575 dias, sem coluna de data)
```

### Rodar o pipeline completo

```bash
venv/bin/python -m dengue_tl.experiment --csv "data/Dados 2007-2024.csv"
# padrão: cnn_lstm; para EfficientNet: --arquitetura efficientnet
# gera cache/tabela_lagged.csv e resultados_experimento.json (métricas + predições + resumo)
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev,dl]"   # dl traz tensorflow, pyts, pillow, scipy
venv/bin/python -m pytest    # 60 testes; os de DL usam importorskip
```

> **macOS/Apple Silicon:** `import tensorflow` dá *segfault* no Python 3.13 do anaconda base.
> Use um venv isolado e rode os testes por ele: `venv/bin/python -m pytest`.

## Origem

`series_loader` e `scaler` vêm de uma Iniciação Científica (mesmo autor). O restante é
próprio deste trabalho. A IC compara 12 representações × cabeças recorrentes; **este
trabalho não** — é um pipeline único focado numa estimativa coerente.
