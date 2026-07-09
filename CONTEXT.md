# Contexto — Estimativa de casos de dengue com CNN-LSTM / CNN2D

Trabalho de disciplina: estimar o número de casos de dengue de um dia a partir dos
dados estruturados ao redor desse dia (uma janela 9×4), com redes convolucionais/
recorrentes pequenas. Aplicado a Cascavel-PR. Reaproveita parte do código de uma
Iniciação Científica, mas com escopo próprio (ver `roteiro.md`). As arquiteturas
avaliadas são **CNN-LSTM** e **CNN2D** (e um MLP a ser adicionado).

## Vocabulário

**Caso**: ocorrência diária confirmada de dengue. `Qtde_Casos` é a variável-alvo.

**Tabela lagged**: tabela supervisionada (`lagged_table`) que alinha, em cada linha `t`,
as features já defasadas e o alvo `Qtde_Casos[t]`. As 4 features são: `Precipitacao_lag45`,
`Temp_media_lag45`, `Umidade_rel_lag45` (clima com 45 dias de atraso) e `Historico_lag30`
(casos passados com 30 dias de atraso).

**Janela**: recorte de 9 dias centrado num dia (dia central + 4 de cada lado). Desliza
de 1 em 1 dia. Empilhada, vira uma **matriz 9×4** (9 dias × 4 features). O alvo é o
`Caso` do dia central (nowcasting / imputação).

**Representação**: a Janela entra na rede como a **matriz 9×4** empilhada, sem
codificação em imagem — os dados são estruturados/multivariados, então a geometria
9×4 é a forma natural do bloco e preserva as 4 features. Cada arquitetura só adapta
o formato em `prepara_entrada` (CNN-LSTM: `(9, 4)` direto; CNN2D: transposta para
`(4, 9, 1)`).

**Arquitetura padrão**: CNN-LSTM — duas `Conv1D` + `LSTM` sobre a janela `(9, 4)` diretamente. Rede minúscula (poucos milhares de parâmetros), treina em segundos na CPU. Seleção via `--arquitetura` em `experiment.py`.

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

**Sazonalidade** (ablation, **DESLIGADA por padrão** desde jul/2026): a dengue em
Cascavel é fortemente sazonal — jan–mai concentra ~94% dos casos. Dá para dar essa
fase ao modelo codificando o **dia-do-ano do dia central** `t` como par
`sin_ano`/`cos_ano` (período 365.25; o par evita a descontinuidade 31/dez→1/jan),
que entra como **duas features extras** na matriz (9×4 → 9×6). O calendário é
reconstruível exatamente (6575 linhas = 2007-01-01…2024-12-31, série dateless mas
contígua). **Não é vazamento** (o calendário é futuro conhecido), mas é uma
**decisão metodológica mantê-la desligada**: dar a data deixa o modelo memorizar o
atalho sazonal ("é março → pico") em vez de aprender a relação clima→casos — que é
o objetivo do trabalho, e o clima já carrega a estação. Fica como **ablation**
(`--sazonalidade` liga; `LaggedTableConfig`/`TreinoConfig` default `False`;
`--data-inicial` fixa a data-base). O cache ganha sufixo `_sazonal` quando ligada,
para não reusar silenciosamente uma tabela de 4 features. **Efeito medido**
(re-tune justo, 40 trials): com a data, MAE cai muito (cnn2d 29.6→22.5, cnn_lstm
29.2→24.2) e o pico destrava — mas **removê-la volta as duas ao ~29.5 (≈ o
original)**, ou seja, o ganho vinha do atalho sazonal, exatamente o que a decisão
evita. A ponderação de pico **só rende junto com a data** (sozinha não bate o
original).

**Peso de pico** (`peso_pico`, default `0.0` = desligado): o erro do modelo se
concentra nos poucos dias de pico epidêmico (que dominam MAE/RMSE na escala de
casos), enquanto a fora-de-temporada já é quase perfeita. `peso_pico` pondera
cada dia de treino na loss por `1 + peso_pico·(nível/nível_max)` (nível =
casos suavizados do dia central de treino), fazendo o otimizador priorizar a
**amplitude** do pico. Não mexe na formulação log-razão — só realoca a atenção.
A validação fica sem peso (não distorce EarlyStopping nem o objetivo do Optuna).
Entra no `espaco_busca` das duas arquiteturas (faixa 0–8), então o Optuna testa
diferentes intensidades por trial: o `sample_weight` é recalculado a cada treino
a partir do `nivel_treino` guardado em `DadosPreparados` (barato), em vez de fixo
na preparação. Também dá para forçar um valor com `--peso-pico`. Ver
`train_runner.pesos_por_nivel`. **Efeito observado**: destrava a amplitude do pico
(cnn2d COM sazonalidade: MAE 29→22, RMSE 77→66, predição máxima ~240→409). **Mas
depende da data**: sem a sazonalidade a ponderação sozinha não bate o original (o
modelo não sabe *quando* alocar amplitude). Com a sazonalidade off por padrão, o
`peso_pico` continua no `espaco_busca`, mas o Optuna tende a escolher valores que
não ajudam no teste — o objetivo de validação descola do teste sob regime de
surto maior (gargalo em aberto, ver item "validação↔teste").

## Fatos dos dados

- Série diária, contígua, sem faltantes. A base completa tem coluna de data; a amostra
  (`data/AmostraDados.csv`) **não** tem — em desenvolvimento, usar índice sequencial.
- Variáveis originais: `Precipitacao`, `Temp_media`, `Umidade_rel`, `Qtde_Casos`. Este
  trabalho usa **as 4** (multivariado): as 3 climáticas e o histórico de casos entram como
  features defasadas na tabela lagged; `Qtde_Casos` do dia atual é o alvo.

## Ponto crítico — vazamento do dia central

Se a entrada incluísse `Qtde_Casos[t]`, o valor-alvo vazaria e a estimativa ficaria
falsamente boa. Na abordagem 9×4 isso é resolvido **pelos lags**: as 4 features são todas
defasadas em ≥30 dias, então nenhuma linha da janela `±4` carrega `Qtde_Casos[t]` — o alvo
nunca entra na matriz de entrada. (Na abordagem GASF univariada legada, o mesmo risco era
tratado excluindo o dia central da imagem; ver `roteiro.md` §2.)
