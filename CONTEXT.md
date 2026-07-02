# Contexto — Estimativa de casos de dengue com EfficientNet

Trabalho de disciplina: estimar o número de casos de dengue de um dia a partir de uma
imagem que codifica a série temporal ao redor desse dia, usando **transfer learning**
com **EfficientNet-B0**. Aplicado a Cascavel-PR. Reaproveita parte do código de uma
Iniciação Científica, mas com escopo próprio (ver `roteiro.md`).

## Vocabulário

**Caso**: ocorrência diária confirmada de dengue. `Qtde_Casos` é a variável-alvo.

**Janela**: recorte de 9 dias centrado num dia (dia central + 4 de cada lado). Desliza
de 1 em 1 dia. O alvo é o `Caso` do dia central (nowcasting / imputação).

**Representação**: forma de codificar a Janela em imagem. Fixada em **GASF** (Gramian
Angular Summation Field) — é uma escolha de projeto, não um eixo de experimento.

**Codificador**: função que transforma a Janela na imagem GASF 100×100×3.

**Backbone**: EfficientNet-B0 pré-treinada (ImageNet), extratora de features.

## Fatos dos dados

- Série diária, contígua, sem faltantes. A base completa tem coluna de data; a amostra
  (`data/AmostraDados.csv`) **não** tem — em desenvolvimento, usar índice sequencial.
- Variáveis: `Precipitacao`, `Temp_media`, `Umidade_rel`, `Qtde_Casos`. Este trabalho
  usa **apenas `Qtde_Casos`** (univariada), replicada em 3 canais RGB.

## Ponto crítico — vazamento do dia central

Se a imagem incluir o dia central, o valor-alvo vaza para a entrada (a diagonal da GASF
o codifica) e a estimativa fica falsamente boa. Default: **excluir o dia central da
imagem** (`incluir_dia_central=False`), codificando só os 8 vizinhos. Ver `roteiro.md` §2.
