# Guia do Projeto — Regressor de Casos de Dengue

Este documento resume **do que se trata o trabalho**, **o que já foi feito** e **como cada arquivo principal participa do pipeline**.

---

## 1) Qual é o objetivo do trabalho

O projeto estima `Qtde_Casos` de dengue de um dia (dia central) usando:

1. uma tabela com features **defasadas** (clima em `t-45`, histórico de casos em `t-30`),
2. uma **matriz 9×4** (9 dias × 4 features) transformada em imagem,
3. regressão com **EfficientNet-B0** (transfer learning).

A ideia é produzir uma estimativa coerente e avaliá-la de forma honesta com split temporal e comparação com baselines. (A abordagem original codificava uma janela univariada de `Qtde_Casos` em imagem **GASF**; ver §3.)

---

## 2) Estado atual (o que já foi implementado)

Os módulos base já estão prontos e cobertos por testes:

- `series_loader.py` ✅
- `scaler.py` ✅
- `window_builder.py` ✅ (trilha GASF univariada, original)
- `lagged_table.py` ✅ (tabela defasada + cache em disco)
- `matrix_windower.py` ✅ (janelas 9×4 da tabela lagged)
- `encoder.py` ✅ (`encode_gasf` e `encode_matrix`)
- `model.py` ✅
- `train_runner.py` ✅ (pipeline 9×4 fim-a-fim)

Situação dos testes: **48 passed** (rodar pelo venv: `venv/bin/python -m pytest`;
os testes de DL usam `importorskip` e são pulados fora do venv).

O pipeline 9×4 roda de ponta a ponta sobre o dataset completo (`data/Dados 2007-2024.csv`,
sem coluna de data). Entregáveis a evoluir: gráfico real × previsto automático e execução multi-seed.

---

## 3) Pipeline completo (visão de ponta a ponta)

Pipeline **atual** (entrada em matriz 9×4 com features defasadas):

1. Montar/cachear a tabela lagged (`lagged_table.build_or_load_lagged_table`):
   clima em `t-45`, histórico de casos em `t-30`, alvo `Qtde_Casos[t]`.
2. Janelar 9 linhas × 4 features por dia central (`matrix_windower.build_matrix_windows`).
3. Split temporal treino/val/teste e escala por-feature (fit só no treino, `Scaler`).
4. Codificar cada matriz 9×4 em imagem 100×100×3 (`encoder.encode_matrix`).
5. Treinar EfficientNet-B0 em 2 fases (`model.build_model` / `descongela_backbone`).
6. Avaliar na escala original (`expm1`) e comparar com baselines (`train_runner.treina_e_avalia`).

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

- `series_loader.py`  
  Lê CSV, parseia data, ordena, valida contiguidade diária e retorna variáveis esperadas.

- `scaler.py`  
  Escala features com min/max do treino; transforma alvo com `log1p` e inverte com `expm1`.

- `window_builder.py`  
  Gera janelas centradas e alvos. Padrão exclui dia central para evitar leakage.

- `encoder.py`  
  Converte janela 1-D em GASF (`pyts`), redimensiona para `100x100` e replica em 3 canais.

- `model.py`  
  Define modelo de regressão com EfficientNet-B0, cabeça densa e função para descongelar backbone (fine-tuning fase 2).

- `train_runner.py`  
  Runner fim-a-fim: carga de dados, split temporal, codificação GASF, treino em 2 fases, métricas e baselines, exportando JSON de resultados.

- `lagged_table.py`  
  Constrói a tabela supervisionada com lags separados: clima em 45 dias e histórico de casos em 30 dias.

- `__init__.py`  
  Arquivo de pacote.

## 5.3 Testes (`tests/`)

- `test_series_loader.py`  
  Valida leitura, estrutura e erro em vão temporal.

- `test_scaler.py`  
  Valida escalonamento e inversão `log1p`/`expm1`.

- `test_window_builder.py`  
  Valida quantidade de amostras, formato da janela, alvos e comportamento com/sem dia central.

- `test_encoder.py`  
  Valida shape, canais, simetria, determinismo e faixa de valores da GASF.

- `test_model.py`  
  Smoke tests do modelo, saída linear, congelamento/descongelamento e BatchNorm.

- `test_train_runner.py`  
  Testes unitários do runner (split temporal, métricas, baseline e fallback sem coluna de data).

---

## 6) Status das Etapas 6 e 7

Implementado no runner:

1. split temporal treino/val/teste (sem shuffle),
2. geração de `X`/`y` para treino com as funções já prontas,
3. treino em duas fases (congelado → fine-tuning),
4. predição e inversão de escala do alvo,
5. métricas finais (MAE, RMSE, CC),
6. baselines (média e último valor),
7. exportação de resultado em JSON com histórico e predições.

Próximos incrementos recomendados:

1. gráfico real × previsto automático,
2. tabela final de comparação pronta para relatório,
3. agregação de múltiplas seeds.

---

## 7) Comandos úteis já validados

Instalar dependências de desenvolvimento:

```bash
.venv/bin/pip install -e ".[dev]"
```

Rodar todos os testes:

```bash
.venv/bin/python -m pytest -q
```

Rodar um teste específico:

```bash
.venv/bin/python -m pytest -q tests/test_window_builder.py::test_raio_customizado
```

Para módulos de deep learning:

```bash
.venv/bin/pip install -e ".[dev,dl]"
```

Exemplo para a amostra pequena (`data/AmostraDados.csv`), usando raio compatível:

```bash
.venv/bin/python -m dengue_tl.train_runner \
  --csv data/AmostraDados.csv \
  --raio 3 \
  --output-json resultados_treino.json
```
