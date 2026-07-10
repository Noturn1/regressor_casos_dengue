# Guia do Projeto — Regressor de Casos de Dengue

Este documento resume **do que se trata o trabalho**, **o que já foi feito** e **como cada arquivo principal participa do pipeline**.

---

## 1) Qual é o objetivo do trabalho

O projeto estima `Qtde_Casos` de dengue de um dia (dia central) usando:

1. uma tabela com features **defasadas** (clima em `t-45`, histórico de casos em `t-30`),
2. uma **janela** dessas features como entrada (matriz `9×4` no modo `lagged`, ou a
   série crua no modo `sequencia`),
3. alvo em **log-razão** (crescimento sobre o histórico), invariante à escala do surto,
4. regressão com **MLP** (baseline), **CNN2D** ou **CNN-LSTM**.

A ideia é comparar as arquiteturas de forma honesta (split temporal, baselines, teste fora
da busca de hiperparâmetros) e entender **o que** limita a estimativa — não perseguir um
número. Achado central: as arquiteturas empatam (~29–30 MAE) e o MLP quase alcança as CNNs,
logo o problema é essencialmente tabular; o erro vive no pico epidêmico.

---

## 2) Estado atual (o que já foi implementado)

Todos os módulos estão prontos e cobertos por testes:

- `series_loader.py` ✅ · `scaler.py` ✅
- `lagged_table.py` ✅ (tabela defasada + sazonalidade opcional + cache)
- `matrix_windower.py` ✅ (janelas 9×4 — modo `lagged`)
- `sequence_windower.py` ✅ (janela crua causal — modo `sequencia`)
- `models/mlp.py` ✅ · `models/cnn2d.py` ✅ · `models/cnn_lstm.py` ✅ (mesma interface)
- `train_runner.py` ✅ (pipeline fim-a-fim, agnóstico à arquitetura; log-razão, peso de pico)
- `tune_runner.py` ✅ (busca Optuna; `--metrica-busca mae|rmse`)
- `experiment.py` ✅ (ponto de entrada central: roda + destila resumo)
- `paths.py` ✅ (convenção `outputs/<rótulo>/`)
- `report/` ✅ (tabelas, gráficos e comparação entre arquiteturas: `report --comparar`)
- `window_builder.py` (trilha 1-D legada da IC, mantida por referência)

Situação dos testes: a suíte passa pelo venv (rodar: `venv/bin/python -m pytest`;
os testes de DL usam `importorskip` e são pulados fora do venv).

O pipeline roda de ponta a ponta sobre o dataset completo (`data/Dados 2007-2024.csv`),
com relatórios individuais (`outputs/<modelo>/relatorio/`) e comparativos
(`outputs/comparacao/`) já gerados pelo módulo `report`.

---

## 3) Pipeline completo (visão de ponta a ponta)

Pipeline **atual** (entrada em matriz 9×4 com features defasadas):

1. Montar/cachear a tabela lagged (`lagged_table.build_or_load_lagged_table`):
   clima em `t-45`, histórico de casos em `t-30`, alvo `Qtde_Casos[t]`.
2. Janelar 9 linhas × 4 features por dia central (`matrix_windower.build_matrix_windows`).
3. Split temporal treino/val/teste e escala por-feature (fit só no treino, `Scaler`).
4. Preparar entrada conforme a arquitetura via `prepara_entrada` (ver §4.4): CNN-LSTM `(n, 9, 4)`, CNN2D `(n, 4, 9, 1)`, MLP `(n, 4)`.
5. Treinar o modelo via `models.seleciona_arquitetura` (todas em fase única), com o alvo em log-razão e a ponderação de pico opcional.
6. Avaliar na escala original (`expm1`) e comparar com baselines (`train_runner.treina_e_avalia`).
7. Destilar resultado em resumo pertinente (`experiment.resumo_pertinente`) e persistir JSON.

> **Trilha original (GASF univariado):**
> `window_builder.build_centrado` (janela 1-D de `Qtde_Casos`) produz entrada 1-D.
> Os módulos continuam no repo (e testados) como referência histórica; o pipeline atual usa a matriz 9×4.

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

### 4.4 Preparação de entrada por arquitetura

`prepara_entrada` recebe a janela escalada e a adapta ao formato de cada rede:
- **CNN-LSTM:** passa direto como `(9, 4)` (ou `(N, 4)` no modo sequência).
- **CNN2D:** transpõe e adiciona dimensão de canal: `(4, 9, 1)`.
- **MLP:** colapsa para a linha central `(4,)` — só as features do dia `t`, sem vizinhos.

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
- `lagged_table.py` — tabela supervisionada: clima `t-45`, histórico `t-30`, alvo `t`; sazonalidade opcional (sin/cos do dia-do-ano).
- `matrix_windower.py` — janelas `(n, 9, 4)` da tabela lagged (modo `lagged`).
- `sequence_windower.py` — janela crua causal `(n, N, 4)` terminando em `t-gap` (modo `sequencia`).
- `models/__init__.py` — `seleciona_arquitetura(nome)`: import preguiçoso (`mlp | cnn2d | cnn_lstm`).
- `models/mlp.py` — Dense × 2 sobre só as 4 features do dia `t` (colapsa a janela); baseline.
- `models/cnn2d.py` — Conv2D(32)→Conv2D(64) 2×2 + Dense(64) sobre a janela transposta (4, 9, 1); spec do professor, sem pooling.
- `models/cnn_lstm.py` — Conv1D × 2 + LSTM + Dense(1); treino em fase única.
- `train_runner.py` — pipeline agnóstico à arquitetura: janela → escala → alvo log-razão → treino (peso de pico) → métricas + baselines.
- `tune_runner.py` — busca Optuna/TPE (`--metrica-busca mae|rmse`); grava `otimizacao.json` + `resultado.json` canônico.
- `experiment.py` — ponto de entrada: chama `treina_e_avalia`, destila `resumo_pertinente`, imprime e salva JSON.
- `paths.py` — convenção de saída `outputs/<rótulo>/` (`--rotulo` separa variantes da mesma arquitetura).
- `report/` — tabelas + gráficos por modelo (`--results`) e comparação entre arquiteturas (`--comparar`).
- `menu.py` — menu interativo que centraliza tudo (treinar, otimizar, relatório, resumos, testes); atalho `./dengue` na raiz.
- `window_builder.py` — janelas centradas (trilha 1-D legada da IC, mantida por referência).

## 5.3 Testes (`tests/`)

- `test_series_loader.py` — leitura, estrutura, erro em vão temporal.
- `test_scaler.py` — escalonamento e inversão `log1p`/`expm1`.
- `test_window_builder.py` — amostras, formato, alvos, com/sem dia central.
- `test_sequence_windower.py` — shapes, gap e anti-vazamento da janela crua.
- `test_mlp.py` · `test_cnn2d.py` · `test_cnn_lstm.py` — smoke tests de cada arquitetura.
- `test_train_runner.py` — split temporal, métricas, baselines, fallback sem coluna de data.
- `test_tune_runner.py` — espaço de busca compatível com `TreinoConfig`; otimização fim-a-fim; métrica de busca (mae/rmse).
- `test_paths.py` — convenção `outputs/<rótulo>/` e `rotulo_de`.
- `test_report_*.py` — tabelas, gráficos, e comparação completa entre arquiteturas.
- `test_experiment.py` — `resumo_pertinente`, `formata_resumo`, fluxo do `roda_experimento`.

---

## 6) Status do pipeline

Pipeline completo e funcional. O que está pronto:

1. split temporal treino/val/teste (sem shuffle),
2. escalonamento sem vazamento (fit só no treino),
3. treino agnóstico à arquitetura (ambas em fase única),
4. predição e inversão de escala do alvo (`expm1`),
5. métricas finais (MAE, RMSE, CC) + baselines (média do treino / histórico `t-30`),
6. exportação de resultado completo em JSON (histórico, predições, métricas),
7. resumo pertinente: melhor estimador por métrica, ganho % em MAE, melhor época.

Visualizações e relatórios já estão prontos no módulo `report`: tabelas + gráficos por
modelo (`report --results …`) e a comparação entre arquiteturas (`report --comparar`:
tabela rica, predições e 5 gráficos comparativos). Gargalo em aberto: a **amplitude do
pico** sob mudança de regime — todos os modelos empacam nela (ver `CONTEXT.md`).

---

## 7) Comandos úteis

```bash
# Menu interativo (treinar, otimizar, relatório, resumos, testes)
./dengue

# Instalar (núcleo + testes + DL + otimização + relatório)
venv/bin/pip install -e ".[dev,dl,opt,report]"

# Rodar todos os testes
venv/bin/python -m pytest -q

# Rodar o experimento (arquitetura: cnn_lstm | cnn2d | mlp)
venv/bin/python -m dengue_tl.experiment --csv "data/Dados 2007-2024.csv" --arquitetura cnn2d

# Modo sequência (o LSTM recebe a série crua e descobre a defasagem)
venv/bin/python -m dengue_tl.experiment --csv "data/Dados 2007-2024.csv" \
  --arquitetura cnn_lstm --modo sequencia --janela-dias 30 --gap-dias 30

# Busca de hiperparâmetros (extra `opt`); --metrica-busca mae|rmse
venv/bin/python -m dengue_tl.tune_runner --csv "data/Dados 2007-2024.csv" --arquitetura mlp --n-trials 40

# Relatório de um modelo e comparação entre todos (pasta outputs/)
venv/bin/python -m dengue_tl.report --csv "data/Dados 2007-2024.csv" --results outputs/cnn2d/resultado.json
venv/bin/python -m dengue_tl.report --csv "data/Dados 2007-2024.csv" --comparar
```
