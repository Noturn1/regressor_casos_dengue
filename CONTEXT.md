# Contexto — Estimativa de casos de dengue com EfficientNet

Trabalho de disciplina: estimar o número de casos de dengue de um dia a partir de uma
imagem que codifica dados estruturados ao redor desse dia, usando **transfer learning**
com **EfficientNet-B0**. Aplicado a Cascavel-PR. Reaproveita parte do código de uma
Iniciação Científica, mas com escopo próprio (ver `roteiro.md`).

## Vocabulário

**Caso**: ocorrência diária confirmada de dengue. `Qtde_Casos` é a variável-alvo.

**Tabela lagged**: tabela supervisionada (`lagged_table`) que alinha, em cada linha `t`,
as features já defasadas e o alvo `Qtde_Casos[t]`. As 4 features são: `Precipitacao_lag45`,
`Temp_media_lag45`, `Umidade_rel_lag45` (clima com 45 dias de atraso) e `Historico_lag30`
(casos passados com 30 dias de atraso).

**Janela**: recorte de 9 dias centrado num dia (dia central + 4 de cada lado). Desliza
de 1 em 1 dia. Empilhada, vira uma **matriz 9×4** (9 dias × 4 features). O alvo é o
`Caso` do dia central (nowcasting / imputação).

**Representação**: forma de codificar a Janela em imagem. Fixada na **matriz 9×4**
empilhada (`encode_matrix`) — os dados são estruturados/multivariados, então a geometria
9×4 é a forma natural do bloco e preserva as 4 features. É uma escolha de projeto, não um
eixo de experimento. (O codificador GASF univariado da IC original ainda existe em
`encoder.py` como alternativa legada, mas não é a abordagem atual.)

**Codificador**: `encode_matrix` transforma a matriz 9×4 na imagem 100×100×3. O resize
9×4 → 100×100 **não é feature engineering**: é um adaptador de interface. A EfficientNet-B0
reduz a resolução espacial por 32× (5 estágios stride-2), então uma entrada 9×4 colapsaria
antes do fim da rede. 100×100 sobrevive ao downsampling (→ ~3×3) e é mais barato que os
224×224 nativos. A interpolação só espalha os 36 valores num canvas maior; não cria
informação.

**Backbone**: EfficientNet-B0 pré-treinada (ImageNet), extratora de features.

## Fatos dos dados

- Série diária, contígua, sem faltantes. A base completa tem coluna de data; a amostra
  (`data/AmostraDados.csv`) **não** tem — em desenvolvimento, usar índice sequencial.
- Variáveis originais: `Precipitacao`, `Temp_media`, `Umidade_rel`, `Qtde_Casos`. Este
  trabalho usa **as 4** (multivariado): as 3 climáticas e o histórico de casos entram como
  features defasadas na tabela lagged; `Qtde_Casos` do dia atual é o alvo. A imagem é
  replicada nos 3 canais RGB (a EfficientNet espera 3 canais).

## Ponto crítico — vazamento do dia central

Se a entrada incluísse `Qtde_Casos[t]`, o valor-alvo vazaria e a estimativa ficaria
falsamente boa. Na abordagem 9×4 isso é resolvido **pelos lags**: as 4 features são todas
defasadas em ≥30 dias, então nenhuma linha da janela `±4` carrega `Qtde_Casos[t]` — o alvo
nunca entra na matriz de entrada. (Na abordagem GASF univariada legada, o mesmo risco era
tratado excluindo o dia central da imagem; ver `roteiro.md` §2.)
