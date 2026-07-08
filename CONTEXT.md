# Contexto — Estimativa de casos de dengue com CNN-LSTM / EfficientNet

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

**Arquitetura padrão**: CNN-LSTM — duas `Conv1D` + `LSTM` sobre a janela `(9, 4)` diretamente, sem codificação em imagem. Rede minúscula (poucos milhares de parâmetros), treina em segundos na CPU.

**Arquitetura alternativa**: EfficientNet-B0 pré-treinada (ImageNet) — requer codificar a janela 9×4 em imagem 100×100×3 via `encode_matrix`. Treino em 2 fases (backbone congelado → fine-tune). Seleção via `--arquitetura` em `experiment.py`.

**Arquitetura cnn2d** (recomendação do professor): CNN 2D pura sobre a janela transposta
`(4, 9, 1)` — Conv2D(32)→Conv2D(64), kernels 2×2 `same`, Flatten, Dense(64), Dense(1).
Sem pooling (entrada minúscula) e sem codificação em imagem. A spec original não tem
dropout (`--dropout 0` a reproduz exatamente).

**Formulação do alvo** (padrão desde jul/2026): o modelo aprende a **log-razão**
`log1p(casos) − log1p(Historico_lag30)` — o crescimento em ~30 dias — em vez do nível
absoluto. Motivo: o treino (2007–~2019) tem máximo de 34 casos/dia e o teste chega a 692;
prever nível não transfere entre regimes (o modelo saturava em ~34 e era matematicamente
incapaz de bater o RMSE do baseline). Na razão, prever 0 == baseline de persistência, e a
formulação é invariante à escala do surto. O alvo (e o histórico do denominador) é
**suavizado com média móvel 7d centrada** — o serrilhado semanal é ruído de notificação —
e as métricas são reportadas nas duas escalas (bruta e `metricas_alvo_suavizado`). O clip
da predição é só anti-explosão do `expm1` (~1000× o máximo do treino), não um teto
epidemiológico. Flags: `--alvo nivel` e `--suavizacao-alvo 0` reproduzem o comportamento
antigo.

**Otimização de hiperparâmetros**: `tune_runner.py` roda uma busca Optuna (TPE) sobre o
`espaco_busca` declarado no módulo de cada arquitetura, minimizando o **MAE de validação
na escala original**. O teste nunca entra na busca: só é usado uma vez, no retreino final
com a melhor configuração. Os dados são preparados uma única vez (`prepara_dados`) e
reutilizados por todos os trials.

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
