# Roteiro — Estimar casos de dengue com EfficientNet

> ⚠️ **Nota histórica:** A trilha de EfficientNet com imagens GASF descrita neste roteiro foi posteriormente removida do escopo do projeto. As arquiteturas atuais são **CNN-LSTM** e **CNN2D**, operando sobre a janela 9×4 bruta (matriz de defasagens) via `prepara_entrada`. Para detalhes atuais, ver `CONTEXT.md` e `GUIA_PROJETO.md`.

> Passo a passo do trabalho. Vocabulário e fatos dos dados em `CONTEXT.md`.
> Este é um repositório **standalone**: `series_loader.py` e `scaler.py` já vêm
> prontos (copiados de uma IC); os demais módulos você implementa.

---

## 1. Objetivo

Resolver um **problema real** — estimar o número de casos de dengue de um dia — com
**uma técnica de deep learning**: *transfer learning* com **EfficientNet-B0** aplicada
a uma **imagem que codifica a série temporal**.

O que importa é chegar a uma **estimativa coerente e defensável**, não comparar formas
de representação. Portanto: **uma** representação fixa (**GASF**), **uma** arquitetura,
e o esforço vai para **validação honesta** e **qualidade da previsão** (§6).

---

## ⚠️ Atualização de arquitetura — entrada em matriz 9×4 (lagged)

A entrada da rede **mudou** em relação ao plano original (GASF univariada). Agora:

- Uma **tabela defasada** (`lagged_table.py`) alinha, em cada dia `t`: clima em
  `t-45`, histórico de casos em `t-30` e o alvo `Qtde_Casos[t]`. É **cacheada em
  disco** (`build_or_load_lagged_table`) — reconstruída só se o arquivo não existir.
- Um **janelador** (`matrix_windower.py`) empilha **9 dias × 4 features** numa
  **matriz 9×4** por dia central; `encoder.encode_matrix` a transforma em imagem
  100×100×3 para a EfficientNet.
- **Dataset:** `data/Dados 2007-2024.csv` (6575 dias, **sem coluna de data** — os
  lags operam posicionalmente).
- **Vazamento:** deixa de ser problema aqui — como as 4 features são defasadas
  (≥30 dias), a janela `±4` nunca contém `Qtde_Casos[t]`; por isso a matriz é 9×4
  (inclui o dia central) e não 8×4.

As seções abaixo descrevem o plano **original** (GASF univariada), mantido como
referência histórica; `window_builder`/`encode_gasf` continuam no repo e testados.

---

## 2. Especificações travadas (plano original — GASF univariada)

- **Tarefa:** regressão — estimar `Qtde_Casos` do **dia central** de uma janela de 9 dias.
- **Janela:** 9 dias, simétrica (dia central + 4 de cada lado). Deslizante, passo 1 dia.
- **Representação (fixa):** **GASF**, univariada sobre `Qtde_Casos`. Imagem `L×L` →
  redimensionada para **100×100** → **replicada em 3 canais** RGB.
- **Backbone:** `EfficientNetB0`, pesos ImageNet; *fine-tuning* em 2 fases.
- **Alvo:** `log1p` no treino, `expm1` na avaliação (via `Scaler`).
- **Protocolo:** split **temporal** sem embaralhar; normalização só no treino; métricas
  MAE/RMSE/CC na escala original; algumas sementes para estimar variância.

### ⚠️ Vazamento do dia central — resolver antes de rodar

O alvo é o dia central; se a imagem codifica os casos incluindo esse dia, o valor-alvo
**vaza** para a entrada (a diagonal da GASF o codifica). Uma estimativa coerente exige:

- **Recomendado:** imagem só com os **8 vizinhos** (±4); o dia central fica **apenas
  como alvo**. Vira uma imputação honesta. Imagem `8×8`.
- Incluir o central (`9×9`) só como exercício didático — aí o RMSE fica falsamente bom.

O windower tem o flag `incluir_dia_central` (default **False**).

---

## 3. Módulos (o que já existe e o que falta)

| Módulo | Estado | Etapa |
|---|---|---|
| `series_loader.py` | **pronto** (reaproveitado) | 1 |
| `scaler.py` | **pronto** (reaproveitado) | 3 |
| `window_builder.py` | **pronto** — janela centrada 1-D (trilha GASF original) | 2 |
| `lagged_table.py` | **pronto** — tabela defasada + cache | — |
| `matrix_windower.py` | **pronto** — janelas 9×4 da tabela | — |
| `encoder.py` | **pronto** — `encode_gasf` + `encode_matrix` | 4 |
| `model.py` | **pronto** — EfficientNet-B0 | 5 |
| `train_runner.py` | **pronto** — pipeline 9×4 fim-a-fim | 6–7 |

---

## 4. Arquitetura

### 4a. Pipeline atual (matriz 9×4 lagged) — implementado

```
CSV diário (4 variáveis, sem data)
        │
        ▼
lagged_table: cada dia t → clima[t-45], histórico[t-30], alvo=Qtde_Casos[t]   (cache em disco)
        │
        ▼
matrix_windower: 9 dias × 4 features → matriz 9×4 por dia central (alvo = casos[t])
        │
        ▼
split temporal → escala por-feature (fit só no treino) → encode_matrix: 9×4 → 100×100×3
        │
        ▼
EfficientNet-B0 (ImageNet) ─► GAP ─► Dropout ─► Dense(1) linear
        │
        ▼
log1p ──► expm1 ──► casos estimados ──► MAE/RMSE/CC vs baselines (média / histórico t-30)
```

### 4b. Pipeline original (GASF univariada) — referência histórica

```
série diária (Qtde_Casos)
        │
        ▼
janela centrada: para o dia t → casos[t-4 .. t+4] (sem t, se incluir_dia_central=False)
        │                         alvo = casos[t]
        ▼
GASF (pyts) ──► imagem L×L ──► resize 100×100 ──► replicar canal ×3 ──► 100×100×3
        │
        ▼
EfficientNet-B0 (ImageNet, congelada) ─► GAP ─► Dropout ─► Dense(1) linear
        │
        ▼
log1p(casos) ──► expm1 ──► casos estimados ──► MAE/RMSE/CC + análise de erro
```

Pré-computar e cachear as imagens (base pequena, ~6,5 mil dias) evita recodificar por época.

---

## 5. Roteiro passo a passo

### Etapa 0 — Ambiente
- `pip install -e ".[dev]"`; rodar `pytest` (loader e scaler passam). Para o DL: `pip install -e ".[dev,dl]"`.
- **Checkpoint:** `EfficientNetB0(weights="imagenet")` instancia sem erro.

### Etapa 1 — Carregar a série
- `series_loader.load`. Na amostra sem coluna de data, usar índice sequencial.
- **Checkpoint:** vetor 1-D contíguo de `Qtde_Casos`.

### Etapa 2 — Janela centrada (`window_builder.py`) ✅
- `build_centrado(casos, config)`: para cada `t` com contexto completo,
  `alvo = casos[t]` e `janela` = os 8 vizinhos (ou 9, se `incluir_dia_central=True`).
- **TDD (feito):** nº de amostras = `N - 2*raio`; comprimento 8 ou 9; alvo é o dia central certo; bordas descartadas; raio configurável.

### Etapa 3 — Alvo e escalonamento
- `Scaler`: `fit` **só no treino**; alvo em `log1p`.
- **Checkpoint:** `inverse_target(transform_target(y)) == y`.

### Etapa 4 — Encoder GASF (`encoder.py`) ✅
- `encode_gasf(janela) → (100,100,3)`: `GramianAngularField(method="summation")` →
  resize `L×L→100×100` → replicar em 3 canais.
- Resize com `scipy.ndimage.zoom(order=1)`, **não** com o PIL: o resize float do
  PIL (modo `'F'`) extrapolava fora de `[-1,1]` e gerava `NaN`.
- `pyts` reescala cada janela internamente (por-amostra, não é vazamento).
- **TDD (feito):** shape `(100,100,3)`; 3 canais idênticos; matriz simétrica; determinismo; range `[-1,1]`; janela constante.

### Etapa 5 — Modelo EfficientNet (`model.py`) ✅
- `build_model`: `EfficientNetB0(include_top=False, weights="imagenet", input_shape=(100,100,3))`
  → `GlobalAveragePooling2D` → `Dropout` → `Dense(1)` linear (~4M params; 1281 treináveis na Fase 1).
  - **Fase 1:** backbone congelado, treina a cabeça (LR maior).
  - **Fase 2:** `descongela_backbone(model, n_camadas_finais=20)`, LR baixo; BatchNorm fica congelada.
- **§7 resolvido:** a `EfficientNetB0` do Keras embute Rescaling+Normalization e espera `[0,255]`;
  como a GASF sai em `[-1,1]`, o modelo mapeia com `Rescaling(scale=127.5, offset=127.5)` — sem normalizar duas vezes.
- **Smoke test (feito):** lote `(batch,100,100,3)` → saída `(batch,1)`; backbone congelado por padrão; saída linear.
- **Ambiente:** rodar pelo venv — no Python 3.13 do anaconda base o `import tensorflow` dá *segfault* (ver README).

### Etapa 6 — Split temporal e treino
- Treino nos anos iniciais, teste nos finais; validação = cauda do treino para
  *early stopping*. **Nunca embaralhar.**
- **Checkpoint:** nenhuma amostra de teste tocou `Scaler.fit` nem a normalização.

### Etapa 7 — Estimativa final e avaliação
- `expm1` antes de medir; MAE, RMSE, CC (Pearson) na escala original; 3–5 sementes → média ± desvio.
- Gráfico **real × previsto** ao longo do tempo (entrega central).

---

## 6. Como saber se a estimativa é "coerente" (seção-chave)

- **Baselines obrigatórios:** "prever a média do treino" e "prever o último/vizinho".
  Se a EfficientNet não bate esses, não há valor.
- **Validação temporal honesta:** split sem embaralhar, normalização só no treino, dia central fora da imagem (§2).
- **Análise de erro:** onde erra? Reportar erro separado em dias de baixa vs. surto.
- **Sanidade:** casos previstos ≥ 0; a curva real × previsto segue a tendência, não uma reta na média.
- **Variância entre sementes:** média ± desvio; desvio enorme = resultado frágil.

Entregável: pipeline funcionando + tabela (modelo vs. baselines) + gráfico real × previsto
+ parágrafo honesto sobre acertos, limites e o vazamento tratado.

---

## 7. Escalonamento — não normalizar duas vezes

1. **Alvo:** `log1p`/`expm1` via `Scaler`, min/max só do treino.
2. **Encoder:** `pyts` reescala cada janela (por-amostra). Ok.
3. **Entrada da EfficientNet:** a versão Keras espera pixels ~`[0,255]` e normaliza
   internamente. As imagens saem em `[0,1]`/`[-1,1]` — escolha **um** caminho e teste a faixa de ativações.

---

## 8. Hardware e custo

- Ryzen 5 5600X / GTX 1060 6 GB / 24 GB. EfficientNet-B0 (~5 M params) cabe folgado a 100×100, batch 32–64.
- Custo minúsculo: 1 representação × poucas sementes sobre ~6,5 mil imagens cacheadas — minutos.
- Cachear imagens e salvar checkpoints.

---

## 9. Pontos de atenção (resumo)

- **Vazamento do dia central** — §2, default `incluir_dia_central=False`; confirmar com o professor.
- **Janela curta** — imagem `8×8`/`9×9` upscalada para 100×100 é interpolação forte; válido, registrar no relatório.
- **Normalização dupla** — §7.
- **Baselines** — sem eles não há como afirmar que a estimativa é boa (§6).
- **Picos de surto** — a assimetria dos casos domina o RMSE; por isso `log1p` e a análise de erro separada.
