# Estimativa de casos de dengue com EfficientNet

Trabalho de disciplina: estimar os casos de dengue de um dia a partir de uma imagem
**GASF** da série ao redor, via **transfer learning com EfficientNet-B0**.

- **O que fazer, passo a passo:** `roteiro.md`
- **Vocabulário e fatos dos dados:** `CONTEXT.md`

## Estrutura

```
src/dengue_tl/
  ├── series_loader.py   ← reaproveitado, PRONTO (carrega/valida a série)
  ├── scaler.py          ← reaproveitado, PRONTO (log1p do alvo, min/max só do treino)
  ├── window_builder.py  ← STUB — implementar (Etapa 2): janela centrada de 9 dias
  ├── encoder.py         ← STUB — implementar (Etapa 4): GASF 100×100×3
  └── model.py           ← STUB — implementar (Etapa 5): EfficientNet-B0
tests/                   ← loader e scaler já passam; os demais você escreve (TDD)
data/AmostraDados.csv    ← amostra para desenvolvimento (sem coluna de data)
```

## Como proceder

1. **Criar o repositório e o ambiente:**
   ```bash
   cd trabalho-disciplina-dengue
   git init && git add -A && git commit -m "chore: esqueleto do trabalho"
   python -m venv venv && source venv/bin/activate
   pip install -e ".[dev]"        # núcleo (numpy, pandas) + pytest
   ```

2. **Baseline verde** — confirmar que os testes reaproveitados passam:
   ```bash
   pytest
   ```

3. **Seguir o `roteiro.md` a partir da Etapa 0.** Implementar os stubs nesta ordem
   (cada um destrava o próximo):
   `window_builder` (Etapa 2) → `encoder` (Etapa 4) → `model` (Etapa 5) → treino/avaliação (Etapas 6–7).
   Escreva o teste antes da implementação (o repo já segue TDD).

4. **Parte de deep learning** — instalar o extra só quando for treinar:
   ```bash
   pip install -e ".[dev,dl]"     # + tensorflow, pyts, pillow, scipy
   ```
   Treine de preferência no desktop com GPU (GTX 1060); no macOS, use para
   prototipar (o TensorFlow roda em CPU/Metal, mais lento).

5. **Entrega** (ver `roteiro.md` §6): métricas do modelo **vs. baselines**
   (média / último valor), gráfico real × previsto, e uma nota honesta sobre o
   vazamento do dia central (mantê-lo em `incluir_dia_central=False`).

## Origem

`series_loader` e `scaler` vêm de uma Iniciação Científica (mesmo autor). O restante é
próprio deste trabalho. A IC compara 12 representações × cabeças recorrentes; **este
trabalho não** — é um pipeline único focado numa estimativa coerente.
