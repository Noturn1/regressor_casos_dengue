# Implementação detalhada — `build_lagged_table`

Este documento explica a nova etapa de preprocessamento implementada em:

- `src/dengue_tl/lagged_table.py`

e coberta por:

- `tests/test_lagged_table.py`

O objetivo aqui é transformar a série temporal original em uma **tabela supervisionada**
com atrasos específicos por tipo de variável, conforme a hipótese epidemiológica do
trabalho.

---

## 1) O problema que esta tabela resolve

A base original tem, para cada dia:

- `Precipitacao`
- `Temp_media`
- `Umidade_rel`
- `Qtde_Casos`

Mas o modelo não deve usar o valor “cru” do mesmo dia para explicar o mesmo dia quando
há defasagem causal esperada. A nova tabela organiza o dado para que a linha do dia `t`
carregue:

- clima de `t-45`
- histórico de casos de `t-30`
- alvo em `t`

Isso transforma um problema temporal em uma matriz tabular explícita de atributos
defasados.

---

## 2) Estrutura do módulo

O módulo foi dividido em funções pequenas para separar responsabilidades:

1. `LaggedTableConfig` — parâmetros dos lags
2. `_valida_contiguidade_diaria` — defesa contra série com buracos
3. `_normaliza_entrada` — aceita CSV ou DataFrame e padroniza a entrada
4. `build_lagged_table` — monta a tabela final

Essa divisão evita uma função gigante e torna cada decisão testável.

---

## 3) Explicação detalhada do código

## 3.1 Constantes

```python
VARIAVEIS_CLIMATICAS = ("Precipitacao", "Temp_media", "Umidade_rel")
VARIAVEL_ALVO = "Qtde_Casos"
```

**Por quê:** explicitam quais variáveis entram no preprocessamento.  
Isso deixa claro que:

- o clima entra como entrada defasada,
- `Qtde_Casos` aparece tanto como histórico quanto como alvo.

---

## 3.2 `LaggedTableConfig`

```python
@dataclass(frozen=True)
class LaggedTableConfig:
    lag_clima: int = 45
    lag_historico: int = 30
    date_column: str = "Data"
```

**Por quê usar dataclass:** deixa os parâmetros do preprocessamento agrupados e
imutáveis.

**Por quê os defaults são 45 e 30:** porque essa foi a regra de domínio informada:

- clima com influência em 45 dias,
- histórico de casos com influência em 30 dias.

---

## 3.3 `_valida_contiguidade_diaria`

Essa função reaplica a regra de série diária contínua.

**Por quê ela existe mesmo o loader já validando:** porque `build_lagged_table` também
aceita `DataFrame` direto. Nesse caso, a entrada pode não ter passado por `series_loader`.

Se houver buraco temporal, a defasagem deixa de representar “45 dias antes” de forma
confiável. Isso quebra a interpretação do lag.

---

## 3.4 `_normaliza_entrada`

Essa função resolve a diversidade de entrada:

- caminho de CSV,
- `DataFrame` com coluna de data,
- `DataFrame` com índice temporal,
- `DataFrame` simples sem data.

### Caso 1: caminho para CSV

Se o CSV contém `Data`, o módulo usa `series_loader.load`, que já:

- lê a data,
- ordena,
- valida contiguidade,
- retorna só as colunas esperadas.

Se o CSV não tem `Data`, ele é lido diretamente e preservado na ordem do arquivo.

**Por quê:** a amostra de desenvolvimento do projeto não tem coluna de data, então o
preprocessamento não pode falhar nesse cenário.

### Caso 2: `DataFrame` com coluna de data

O código:

1. converte a coluna para datetime,
2. ordena cronologicamente,
3. move a data para o índice,
4. valida contiguidade.

**Por quê:** o lag só faz sentido em ordem temporal correta.

### Caso 3: `DataFrame` com índice datetime

O índice é ordenado e validado.

### Caso 4: `DataFrame` simples sem data

O código retorna a base como está.

**Por quê:** em alguns fluxos de teste ou prototipagem, o dado já pode estar
sequencialmente ordenado e sem coluna temporal explícita.

---

## 3.5 `build_lagged_table`

Essa é a função principal.

### Validação de colunas

O código verifica se existem:

- `Precipitacao`
- `Temp_media`
- `Umidade_rel`
- `Qtde_Casos`

**Por quê:** falhar cedo é melhor do que produzir tabela incompleta ou errada.

### Construção das features de clima

```python
base[list(VARIAVEIS_CLIMATICAS)].shift(config.lag_clima)
```

Isso move as variáveis climáticas 45 linhas para baixo.

**Interpretação:** o valor que estava em `t-45` passa a ser associado à linha `t`.

Depois vem:

```python
.add_suffix(f"_lag{config.lag_clima}")
```

**Por quê:** os nomes das colunas passam a dizer explicitamente que são lags.

### Construção do histórico de casos

```python
base[[VARIAVEL_ALVO]].shift(config.lag_historico)
```

Isso cria a coluna de histórico de casos com 30 dias de atraso.

Em seguida ela é renomeada para:

- `Historico_lag30`

**Por quê usar um nome de “histórico” e não apenas `Qtde_Casos_lag30`:**
fica mais legível para o trabalho, porque destaca o papel semântico da variável.

### Manter o alvo original

O `Qtde_Casos` do dia atual permanece como alvo.

**Por quê:** a tabela final deve representar um problema supervisionado:

- entradas = passado,
- saída = caso atual.

### `dropna()`

As primeiras linhas da série não têm contexto suficiente para todos os lags.
Essas linhas naturalmente viram `NaN` e são descartadas.

**Por quê:** não há como inventar os 45 ou 30 dias anteriores sem introduzir viés.

---

## 4) O que a tabela representa, na prática

Se a linha atual é o dia `t`, a tabela diz:

- clima observado em `t-45`
- casos anteriores em `t-30`
- alvo em `t`

Exemplo intuitivo:

- `30/07` vira uma linha cuja entrada climática vem de `15/06`
- o histórico de transmissão vem de `30/06`
- o alvo é `Qtde_Casos[30/07]`

Isso captura a hipótese do trabalho sem precisar de janela centrada ou imagem.

---

## 5) O que os testes garantem

### `test_build_lagged_table_cria_colunas_esperadas`

Garante que os nomes de coluna deixam claro o lag aplicado.

### `test_build_lagged_table_descarta_linhas_sem_contexto`

Garante que as linhas iniciais sem histórico são removidas.

### `test_build_lagged_table_usa_45_dias_para_clima_e_30_para_historico`

Garante que cada tipo de variável pega o atraso correto.

### `test_build_lagged_table_permire_custom_lags`

Garante que os lags são configuráveis.

### `test_build_lagged_table_erro_com_colunas_ausentes`

Garante falha explícita quando faltam colunas obrigatórias.

---

## 6) Por que esta implementação é importante

Essa tabela muda o problema de:

- “usar uma representação de janela e converter em imagem”

para:

- “usar atributos defasados explícitos em uma tabela supervisionada”.

Isso é importante porque a regra epidemiológica foi refinada:

- clima influencia com um atraso diferente,
- histórico de casos tem outra janela temporal,
- o modelo final pode explorar esses atrasos de forma direta.

---

## 7) Limitações e próximos passos

Esta implementação resolve a construção da tabela, mas ainda deixa aberto:

1. decidir qual modelo tabular será usado depois,
2. integrar o runner de treino para consumir essa nova tabela,
3. definir se haverá normalização adicional por coluna,
4. validar se os lags de 45 e 30 devem variar por experimento ou ficar fixos.

---

## 8) Comportamento esperado ao usar o módulo

Uso básico:

```python
from dengue_tl.lagged_table import build_lagged_table

tabela = build_lagged_table("dados.csv")
```

Saída esperada:

- tabela ordenada temporalmente,
- colunas lagged explícitas,
- linhas iniciais descartadas,
- alvo pronto para modelagem supervisionada.

