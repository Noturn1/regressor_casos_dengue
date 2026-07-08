# Guia do Projeto — Regressor de Casos de Dengue

Este documento resume **do que se trata o trabalho**, **o que já foi feito** e **como cada arquivo principal participa do pipeline**.

---

## 1) Qual é o objetivo do trabalho

O projeto estima `Qtde_Casos` de dengue de um dia (dia central) usando:

1. uma tabela com features **defasadas** (clima em `t-45`, histórico de casos em `t-30`),
2. uma **janela 9×4** (9 dias × 4 features) como entrada da rede,
3. regressão com **CNN-LSTM** (padrão) ou **EfficientNet-B0** (alternativa com transfer learning).

A ideia é produzir uma estimativa coerente e avaliá-la com split temporal e comparação com baselines. (A abordagem original codificava uma janela univariada de `Qtde_Casos` em imagem **GASF**; ver §3.)

---

## 2) Estado atual (o que já foi implementado)

Todos os módulos estão prontos e cobertos por testes:

- `series_loader.py` ✅
- `scaler.py` ✅
- `window_builder.py` ✅ (trilha GASF univariada, legada)
- `lagged_table.py` ✅ (tabela defasada + cache em disco)
- `matrix_windower.py` ✅ (janelas 9×4 da tabela lagged)
- `encoder.py` ✅ (`encode_gasf` e `encode_matrix`)
- `models/cnn_lstm.py` ✅ (Conv1D + LSTM sobre janela 9×4 — **padrão**)
- `models/efficientnet.py` ✅ (EfficientNet-B0 com transfer learning — alternativa)
- `train_runner.py` ✅ (pipeline 9×4 fim-a-fim, agnóstico à arquitetura)
- `experiment.py` ✅ (ponto de entrada central: roda pipeline + destila resumo pertinente)

Situação dos testes: **60 passed** (rodar pelo venv: `venv/bin/python -m pytest`;
os testes de DL usam `importorskip` e são pulados fora do venv).

O pipeline roda de ponta a ponta sobre o dataset completo (`data/Dados 2007-2024.csv`).
Próximo entregável: visualizações (gráfico real × previsto) via `visualizations.py` — o hook `--figuras-dir` já existe em `experiment.py`.

---

## 3) Pipeline completo (visão de ponta a ponta)

Pipeline **atual** (entrada em matriz 9×4 com features defasadas):

1. Montar/cachear a tabela lagged (`lagged_table.build_or_load_lagged_table`):
   clima em `t-45`, histórico de casos em `t-30`, alvo `Qtde_Casos[t]`.
2. Janelar 9 linhas × 4 features por dia central (`matrix_windower.build_matrix_windows`).
3. Split temporal treino/val/teste e escala por-feature (fit só no treino, `Scaler`).
4. Preparar entrada conforme a arquitetura: CNN-LSTM recebe `(n, 9, 4)` direto; EfficientNet codifica cada 9×4 em imagem 100×100×3 via `encoder.encode_matrix`.
5. Treinar o modelo via `models.seleciona_arquitetura` (CNN-LSTM: fase única; EfficientNet: 2 fases).
6. Avaliar na escala original (`expm1`) e comparar com baselines (`train_runner.treina_e_avalia`).
7. Destilar resultado em resumo pertinente (`experiment.resumo_pertinente`) e persistir JSON.

> **Trilha original (GASF univariado), agora superada como entrada da rede:**
> `window_builder.build_centrado` (janela 1-D de `Qtde_Casos`) → `encoder.encode_gasf`.
> Os módulos continuam no repo (e testados), mas o runner usa a matriz 9×4.

---

## 4) Convenções cruciais do projeto

### 4.1 Evitar vazamento do alvo

No `window_builder`, o padrão é:

- `incluir_dia_central=False`

Isso impede que o valor-alvo (dia central) vaze para a entrada da imagem.

### 4.2 Integridade temporal

`series_loader` valida contiguidade diária e levanta `SeriesGapError` se houver vão.

### 4.3 Escalonamento sem vazamento

`Scaler.fit` deve usar **apenas treino** para min/max.

### 4.4 Contrato da imagem GASF

`encode_gasf` produz `(100, 100, 3)` com canais idênticos e comportamento determinístico.

### 4.5 Entrada da EfficientNet

No `model.py`, há mapeamento explícito de `[-1, 1]` para `[0, 255]` antes do backbone, para não “normalizar duas vezes”.

---

## 5) Explicação dos arquivos principais

## 5.1 Raiz

- `README.md`  
  Resumo geral do projeto e comandos de setup/testes.

- `CONTEXT.md`  
  Vocabulário e contexto dos dados (inclui alerta de vazamento do dia central).

- `roteiro.md`  
  Especificação do trabalho por etapas (1–7), arquitetura e critérios de entrega.

- `pyproject.toml`  
  Configuração do pacote, dependências e pytest (`testpaths = ["tests"]`, `pythonpath = ["src"]`).

- `requirements.txt`  
  Dependências auxiliares do ambiente local (não é a fonte principal de configuração do pacote).

- `.github/copilot-instructions.md`  
  Instruções para sessões futuras do Copilot neste repositório.

- `.vscode/mcp.json`  
  Configuração MCP compartilhada para VS Code (filesystem, github, sqlite).

## 5.2 Código-fonte (`src/dengue_tl/`)

- `series_loader.py` — lê CSV, ordena, valida contiguidade diária.
- `scaler.py` — min/max por-feature (fit só no treino); `log1p`/`expm1` no alvo.
- `window_builder.py` — janelas centradas (trilha GASF legada, exclui dia central).
- `lagged_table.py` — tabela supervisionada: clima `t-45`, histórico `t-30`, alvo `t`.
- `matrix_windower.py` — janelas `(n, 9, 4)` da tabela lagged.
- `encoder.py` — `encode_gasf` (1-D → GASF) e `encode_matrix` (9×4 → 100×100×3).
- `models/__init__.py` — `seleciona_arquitetura(nome)`: import preguiçoso do módulo pedido.
- `models/cnn_lstm.py` — Conv1D × 2 + LSTM + Dense(1); treino em fase única.
- `models/efficientnet.py` — EfficientNet-B0 com cabeça densa; treino em 2 fases (congelar → fine-tune).
- `train_runner.py` — pipeline 9×4 agnóstico à arquitetura: tabela → janela → escala → treino → métricas + baselines.
- `tune_runner.py` — busca de hiperparâmetros (Optuna/TPE): minimiza o MAE de validação e retreina/avalia a melhor config no teste.
- `experiment.py` — ponto de entrada: chama `treina_e_avalia`, destila `resumo_pertinente`, imprime e salva JSON.
- `menu.py` — menu interativo que centraliza tudo (treinar, otimizar, relatório, resumos, testes); atalho `./dengue` na raiz.

## 5.3 Testes (`tests/`)

- `test_series_loader.py` — leitura, estrutura, erro em vão temporal.
- `test_scaler.py` — escalonamento e inversão `log1p`/`expm1`.
- `test_window_builder.py` — amostras, formato, alvos, com/sem dia central.
- `test_encoder.py` — shape, canais, simetria, determinismo, faixa de valores.
- `test_efficientnet.py` — smoke tests da EfficientNet (saída, congelar/descongelar, BatchNorm).
- `test_cnn_lstm.py` — smoke tests da CNN-LSTM (shape de saída, treino mínimo).
- `test_train_runner.py` — split temporal, métricas, baselines, fallback sem coluna de data.
- `test_tune_runner.py` — espaço de busca compatível com `TreinoConfig`; otimização fim-a-fim (CNN-LSTM, 2 trials).
- `test_experiment.py` — `resumo_pertinente`, `formata_resumo`, fluxo do `roda_experimento`.

---

## 6) Status do pipeline

Pipeline completo e funcional. O que está pronto:

1. split temporal treino/val/teste (sem shuffle),
2. escalonamento sem vazamento (fit só no treino),
3. treino agnóstico à arquitetura (CNN-LSTM fase única; EfficientNet 2 fases),
4. predição e inversão de escala do alvo (`expm1`),
5. métricas finais (MAE, RMSE, CC) + baselines (média do treino / histórico `t-30`),
6. exportação de resultado completo em JSON (histórico, predições, métricas),
7. resumo pertinente: melhor estimador por métrica, ganho % em MAE, melhor época.

Próximo incremento: `visualizations.py` — gráfico real × previsto. O hook `--figuras-dir` em `experiment.py` já está pronto para recebê-lo.

---

## 7) Comandos úteis

```bash
# Menu interativo (treinar, otimizar, relatório, resumos, testes)
./dengue

# Instalar (núcleo + testes + DL + otimização + relatório)
venv/bin/pip install -e ".[dev,dl,opt,report]"

# Rodar todos os testes
venv/bin/python -m pytest -q

# Rodar o experimento completo (CNN-LSTM, padrão)
venv/bin/python -m dengue_tl.experiment --csv "data/Dados 2007-2024.csv"

# Com EfficientNet
venv/bin/python -m dengue_tl.experiment --csv "data/Dados 2007-2024.csv" --arquitetura efficientnet

# Amostra pequena (raio menor para não estourar a série curta)
venv/bin/python -m dengue_tl.experiment --csv data/AmostraDados.csv --raio 3

# Busca de hiperparâmetros (requer extra `opt`: venv/bin/pip install -e ".[dl,opt]")
venv/bin/python -m dengue_tl.tune_runner --csv "data/Dados 2007-2024.csv" --arquitetura cnn_lstm --n-trials 50
venv/bin/python -m dengue_tl.tune_runner --csv "data/Dados 2007-2024.csv" --arquitetura efficientnet --n-trials 15
```
