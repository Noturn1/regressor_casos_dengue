# Implementação detalhada — Runner de treino/validação

Este documento explica, em nível técnico, a implementação do arquivo:

- `src/dengue_tl/train_runner.py`

e os testes associados:

- `tests/test_train_runner.py`

O foco é explicar **o que cada parte faz** e, principalmente, **por que foi feita assim**.

---

## 1) Objetivo desta implementação

Antes desta mudança, o projeto já tinha os blocos principais prontos (`series_loader`, `window_builder`, `encoder`, `model`, `scaler`), mas faltava o “orquestrador” para:

1. carregar os dados,
2. construir janelas/alvos,
3. codificar GASF,
4. dividir temporalmente treino/validação/teste,
5. treinar em 2 fases,
6. avaliar modelo e baselines,
7. salvar resultados de forma reproduzível.

O `train_runner.py` foi criado para cumprir exatamente essa lacuna (etapas 6 e 7 do roteiro).

---

## 2) Visão geral do `train_runner.py`

O módulo foi estruturado em camadas para manter responsabilidade clara:

1. **Configuração** (`SplitConfig`, `TreinoConfig`)
2. **Pré-processamento temporal** (`carrega_serie_casos`, `split_temporal`)
3. **Baselines e métricas** (`baseline_media`, `baseline_ultimo_vizinho`, `calcula_metricas`)
4. **Treino e avaliação fim-a-fim** (`treina_e_avalia`)
5. **CLI** (`_parse_args`, `main`)

Essa separação facilita teste unitário e evita acoplamento entre parsing de argumentos e lógica científica.

---

## 3) Explicação detalhada do código (`train_runner.py`)

## 3.1 Cabeçalho e imports

```python
from __future__ import annotations
```

**Por quê:** permite usar anotações de tipo mais limpas sem custo de avaliação imediata de tipos.

```python
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
```

**Por quê:**  
- `argparse/json/Path` para CLI e persistência de saída;  
- `dataclass` para configs imutáveis e serializáveis;  
- `numpy/pandas` para manipulação numérica e leitura de dados.

```python
from dengue_tl.scaler import Scaler
from dengue_tl.series_loader import load
from dengue_tl.window_builder import JanelaCentradaConfig, build_centrado
```

**Por quê:** reutiliza os módulos já prontos, mantendo consistência de regras de domínio do próprio repositório.

---

## 3.2 `SplitConfig` e `TreinoConfig`

`SplitConfig` concentra apenas frações de split temporal.  
`TreinoConfig` concentra hiperparâmetros e caminhos do experimento.

**Decisão importante:** `TreinoConfig` inclui `split: SplitConfig`, em vez de variáveis soltas.  
**Por quê:** mantém o contrato semântico explícito (“isto é configuração de split”), ajuda serialização e evita parâmetros avulsos crescendo sem estrutura.

---

## 3.3 `carrega_serie_casos`

Fluxo:

1. lê apenas cabeçalho (`nrows=0`);
2. se `date_column` existir, usa `series_loader.load` (com validação de contiguidade);
3. se não existir, faz fallback para CSV sequencial usando `Qtde_Casos`.

**Por que esse fallback existe:** o próprio projeto tem amostra sem coluna de data (`data/AmostraDados.csv`), então o runner precisa funcionar nesse cenário sem quebrar.

**Por que validar `Qtde_Casos`:** mesmo sem data, o alvo é obrigatório; erro explícito previne execução silenciosa com schema incorreto.

---

## 3.4 `split_temporal`

A função recebe `n_amostras`, `treino_fracao`, `validacao_fracao` e devolve três `slice`s.

Validações feitas:

1. faixas válidas de frações;
2. soma de treino+validação menor que 1;
3. split viável para tamanho da série (garante conjuntos não vazios e ordem correta).

**Por quê `slice` e não arrays de índices:**  
- mais leve e simples para segmentação sequencial,  
- sem cópia desnecessária,  
- comunica claramente “segmento contínuo temporal”.

---

## 3.5 Baselines

## `baseline_media`

Prediz média do treino para todos os pontos do teste.

**Por quê:** baseline obrigatório de referência mínima; se o modelo não supera isso, não há ganho real.

## `baseline_ultimo_vizinho`

Usa `t-1` (persistência local) existente na janela para previsão.

**Por quê:** baseline temporal forte e simples, útil para séries com autocorrelação.

**Detalhe de índice:** com janela padrão sem dia central e raio=4, a posição de `t-1` é índice `3` (`raio - 1`).

---

## 3.6 Métricas (`mae`, `rmse`, `cc_pearson`, `calcula_metricas`)

- **MAE**: erro absoluto médio (intuitivo na unidade original de casos).
- **RMSE**: penaliza mais erros grandes (sensível a surtos/picos).
- **CC (Pearson)**: mede alinhamento de tendência temporal.

Tratamento implementado:

```python
if np.std(y_true) == 0 or np.std(y_pred) == 0:
    return float("nan")
```

**Por quê:** correlação é indefinida para vetor constante; `NaN` explicita condição matemática inválida em vez de mascarar.

---

## 3.7 `_set_semente`

Atualmente fixa seed do NumPy.

**Por quê:** melhora reprodutibilidade de partes estocásticas controladas por NumPy sem acoplar o módulo a backend específico além do necessário.

---

## 3.8 `treina_e_avalia` (núcleo da implementação)

Esta função executa o experimento completo.

### Passo a passo interno

1. define seed;
2. carrega série de casos;
3. monta janelas/alvos com config de vazamento (`incluir_dia_central`);
4. valida quantidade mínima de amostras;
5. gera split temporal treino/val/teste;
6. calcula baselines na escala original;
7. transforma alvo do modelo com `log1p` via `Scaler`;
8. **importa `encoder/model/keras de forma lazy`;**
9. codifica todas as janelas em imagens;
10. treina fase 1 (backbone congelado);
11. descongela últimas camadas e treina fase 2;
12. prediz no teste e aplica `expm1`;
13. calcula métricas de modelo e baselines;
14. retorna dicionário serializável com config, split, histórico, métricas e predições.

### Nova validação de tamanho mínimo para split

Foi adicionada a função `_min_amostras_para_split(...)`, que calcula o menor `n` capaz
de produzir treino/validação/teste válidos para as frações escolhidas.

**Por quê:** antes, um CSV pequeno com raio alto falhava com mensagem genérica.  
Agora o erro informa:

- linhas no CSV,
- raio usado,
- amostras geradas após janela,
- mínimo exigido pelo split,
- sugestão objetiva de ajuste (ex.: `--raio <= 3`).

Isso acelera diagnóstico e evita tentativa/erro manual.

### Por que import lazy de DL

```python
from dengue_tl.encoder import encode_gasf
from dengue_tl.model import build_model, descongela_backbone, keras
```

**Por quê:** permite testar/utilizar partes não-DL do runner sem exigir import imediato de dependências pesadas (`pyts`, `keras`, backend).

### Por que `EarlyStopping(... restore_best_weights=True)`

Evita continuar treinamento degradando validação e garante uso do melhor ponto da fase.

### Por que salvar histórico das duas fases

Permite auditoria de convergência e comparação entre fase congelada e fine-tuning.

---

## 3.9 CLI (`_parse_args` e `main`)

A CLI expõe parâmetros importantes de experimento (raio, épocas, LR, split, seed, etc.) e salva JSON final.

**Por quê JSON de saída:** formato simples para versionar resultados, comparar runs e alimentar análises posteriores.

Comando de uso:

```bash
.venv/bin/python -m dengue_tl.train_runner \
  --csv data/SEU_ARQUIVO.csv \
  --output-json resultados_treino.json
```

---

## 4) Explicação dos testes (`tests/test_train_runner.py`)

Os testes focam em comportamento crítico do runner sem depender de treino de DL.

1. `test_split_temporal_preserva_ordem`  
   Verifica fronteiras exatas dos slices no caso clássico.

2. `test_split_temporal_erro_quando_inviavel`  
   Garante falha explícita quando as frações inviabilizam split útil.

3. `test_baseline_ultimo_vizinho_usa_t_menos_1`  
   Confirma indexação correta do baseline persistente.

4. `test_calcula_metricas`  
   Valida fórmulas de MAE/RMSE/CC com exemplo conhecido.

5. `test_carrega_serie_casos_fallback_sem_data`  
   Garante compatibilidade com CSV sem coluna de data (amostra de desenvolvimento).

---

## 5) Decisões de projeto e “porquês” principais

1. **Reuso de módulos existentes**: reduz risco de desvio de regra de negócio.
2. **Split temporal obrigatório**: evita vazamento futuro-passado.
3. **Baselines no mesmo runner**: comparação justa no mesmo recorte de teste.
4. **Alvo em log1p/expm1**: coerência com convenção já estabelecida no projeto.
5. **Import lazy de DL**: melhora ergonomia de testes e robustez em ambiente sem extras.
6. **Saída consolidada em JSON**: facilita rastreabilidade de experimento.

---

## 6) Limitações atuais (intencionais nesta entrega)

1. O runner não gera gráfico automaticamente nesta implementação.
2. O runner não faz múltiplas seeds em uma chamada (run único por execução).
3. O split é por fração; não há split por data calendário explícita.

Esses pontos podem ser evoluídos na próxima etapa sem alterar o núcleo já montado.

---

## 7) Próxima integração natural

Após este runner:

1. adicionar geração de gráfico `real x previsto`,
2. adicionar tabela comparativa final pronta para relatório,
3. opcionalmente adicionar execução multi-seed com agregação de média/desvio.
