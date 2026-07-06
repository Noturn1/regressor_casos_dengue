# Estimativa de casos de dengue com EfficientNet

Trabalho de disciplina: estimar os casos de dengue de um dia a partir de uma imagem
**GASF** da série ao redor, via **transfer learning com EfficientNet-B0**.

- **O que fazer, passo a passo:** `roteiro.md`
- **Vocabulário e fatos dos dados:** `CONTEXT.md`

## Estrutura

```
src/dengue_tl/
  ├── series_loader.py   ← reaproveitado, PRONTO (carrega/valida a série)
  ├── scaler.py          ← reaproveitado, PRONTO (log1p do alvo, min/max só do treino)
  ├── window_builder.py  ← PRONTO — trilha GASF univariada original (janela centrada)
  ├── lagged_table.py    ← PRONTO — tabela defasada (clima t-45, histórico t-30) + cache
  ├── matrix_windower.py ← PRONTO — janelas 9×4 (9 dias × 4 features) da tabela lagged
  ├── encoder.py         ← PRONTO — encode_gasf (1-D) e encode_matrix (9×4) → 100×100×3
  ├── model.py           ← PRONTO — EfficientNet-B0 (transfer learning)
  └── train_runner.py    ← PRONTO — pipeline 9×4 fim-a-fim (tabela→janela→escala→treino→avaliação)
tests/                   ← 48 testes passam (rodar pelo venv; os de DL usam importorskip)
data/
  ├── AmostraDados.csv       ← amostra p/ desenvolvimento (9 dias, sem coluna de data)
  └── Dados 2007-2024.csv    ← dataset completo de Cascavel (6575 dias, sem coluna de data)
```

### Rodar o pipeline completo

```bash
venv/bin/python -m dengue_tl.train_runner --csv "data/Dados 2007-2024.csv"
# gera cache/tabela_lagged.csv (cache dos lags) e resultados_treino.json (métricas + predições)
```

## Como proceder

1. **Criar o repositório e o ambiente:**
   ```bash
   cd trabalho-disciplina-dengue
   git init && git add -A && git commit -m "chore: esqueleto do trabalho"
   python -m venv venv && source venv/bin/activate
   pip install -e ".[dev]"        # núcleo (numpy, pandas) + pytest
   ```

2. **Baseline verde** — confirmar que os testes reaproveitados passam:
   ```bash
   pytest
   ```

3. **Seguir o `roteiro.md`.** Ordem dos módulos (cada um destrava o próximo):
   `window_builder` (Etapa 2 ✅) → `encoder` (Etapa 4 ✅) → `model` (Etapa 5 ✅)
   → treino/avaliação (Etapas 6–7, **próximo**). Escreva o teste antes da implementação (o repo segue TDD).

4. **Parte de deep learning** — o extra `dl` traz `pyts`+`scipy` (já usados pelo
   encoder da Etapa 4) e `tensorflow`+`pillow` (para o modelo, Etapa 5):
   ```bash
   pip install -e ".[dev,dl]"     # tensorflow, pyts, pillow, scipy
   ```
   Treine de preferência no desktop com GPU (GTX 1060); no macOS, use para
   prototipar (o TensorFlow roda em CPU/Metal, mais lento).

   > **macOS/Apple Silicon:** o `import tensorflow` dá *segfault* no Python 3.13
   > do anaconda base. Use um **venv isolado** (`python -m venv venv`) e rode os
   > testes de DL por ele: `venv/bin/python -m pytest`. Os testes de model usam
   > `pytest.importorskip("tensorflow")`, então são pulados onde o extra `dl` não está.

5. **Entrega** (ver `roteiro.md` §6): métricas do modelo **vs. baselines**
   (média / último valor), gráfico real × previsto, e uma nota honesta sobre o
   vazamento do dia central (mantê-lo em `incluir_dia_central=False`).

## Origem

`series_loader` e `scaler` vêm de uma Iniciação Científica (mesmo autor). O restante é
próprio deste trabalho. A IC compara 12 representações × cabeças recorrentes; **este
trabalho não** — é um pipeline único focado numa estimativa coerente.
