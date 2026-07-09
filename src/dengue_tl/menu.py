"""Menu interativo: ponto de entrada unico do projeto.

Centraliza as tarefas do dia-a-dia sem exigir comandos verbosos:

1. treinar e avaliar um modelo (cnn_lstm | cnn2d);
2. buscar hiperparametros com Optuna (tune_runner);
3. gerar tabelas/graficos do relatorio (dengue_tl.report);
4. ver o resumo de um resultado JSON ja salvo;
5. rodar a suite de testes.

Todas as perguntas tem default (Enter aceita); os imports pesados (TensorFlow,
matplotlib, optuna) so acontecem quando a acao escolhida precisa deles.

Uso: `./dengue` na raiz do projeto (ou `python -m dengue_tl.menu`).
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from dengue_tl.models import ARQUITETURAS
from dengue_tl.paths import (
    caminho_best_config,
    caminho_otimizacao,
    caminho_resultado,
    garante_pai,
)
from dengue_tl.train_runner import SplitConfig, TreinoConfig

CSV_DIR = Path("data")

# Trials default por arquitetura (redes pequenas: treinam em segundos na CPU).
TRIALS_DEFAULT = {"cnn_lstm": 50, "cnn2d": 50, "mlp": 50}


# ---------------------------------------------------------------- prompts ----


def _pergunta(texto: str, default: str | None = None) -> str:
    sufixo = f" [{default}]" if default is not None else ""
    resposta = input(f"{texto}{sufixo}: ").strip()
    if not resposta and default is not None:
        return default
    return resposta


def _pergunta_int(texto: str, default: int) -> int:
    while True:
        resposta = _pergunta(texto, str(default))
        try:
            return int(resposta)
        except ValueError:
            print("  valor invalido, digite um inteiro.")


def _confirma(texto: str, default: bool = True) -> bool:
    sufixo = "S/n" if default else "s/N"
    resposta = input(f"{texto} [{sufixo}]: ").strip().lower()
    if not resposta:
        return default
    return resposta in ("s", "sim", "y", "yes")


def _escolhe(titulo: str, opcoes: list[str], default_idx: int = 0) -> str:
    """Mostra opcoes numeradas e retorna a escolhida (Enter = default)."""
    print(f"\n{titulo}")
    for i, opcao in enumerate(opcoes, start=1):
        marca = " (padrao)" if i - 1 == default_idx else ""
        print(f"  {i}. {opcao}{marca}")
    while True:
        resposta = input(f"Escolha [1-{len(opcoes)}]: ").strip()
        if not resposta:
            return opcoes[default_idx]
        if resposta.isdigit() and 1 <= int(resposta) <= len(opcoes):
            return opcoes[int(resposta) - 1]
        print("  opcao invalida.")


def _escolhe_multiplos(titulo: str, opcoes: list[str]) -> list[str]:
    """Seleção múltipla: mostra lista numerada, retorna as escolhidas (mín. 1)."""
    print(f"\n{titulo}")
    for i, opcao in enumerate(opcoes, start=1):
        print(f"  {i}. {opcao}")
    print(f"  (Enter = todas; ex.: 1 3 ou 1,3)")
    while True:
        resposta = input(f"Escolha [1-{len(opcoes)}]: ").strip()
        if not resposta:
            return opcoes
        tokens = resposta.replace(",", " ").split()
        try:
            indices = [int(t) for t in tokens]
        except ValueError:
            print("  entrada invalida — use numeros separados por espaco ou virgula.")
            continue
        if any(i < 1 or i > len(opcoes) for i in indices):
            print(f"  numeros fora do intervalo 1-{len(opcoes)}.")
            continue
        escolhidas = list(dict.fromkeys(opcoes[i - 1] for i in indices))
        return escolhidas


def _escolhe_csv() -> str:
    csvs = sorted(str(p) for p in CSV_DIR.glob("*.csv"))
    if not csvs:
        return _pergunta("Caminho do CSV de entrada")
    if len(csvs) == 1:
        print(f"\nCSV: {csvs[0]}")
        return csvs[0]
    return _escolhe("Qual CSV usar?", csvs)


def _escolhe_resultados() -> Path | None:
    """Lista os JSONs de resultado em outputs/ (mais recentes primeiro)."""
    from dengue_tl.paths import OUTPUTS_DIR

    # Coleta resultado.json e otimizacao.json de cada subpasta de arquitetura.
    _NOMES_RESULTADO = {"resultado.json", "otimizacao.json"}
    jsons = sorted(
        (
            p
            for p in OUTPUTS_DIR.glob("*/*.json")
            if p.name in _NOMES_RESULTADO
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not jsons:
        caminho = _pergunta("Nenhum resultado encontrado em outputs/. Caminho do JSON")
        return Path(caminho) if caminho else None
    escolhido = _escolhe(
        "Qual arquivo de resultados usar? (mais recente primeiro)",
        [str(p) for p in jsons],
    )
    return Path(escolhido)


# ------------------------------------------------------------ construcao ----


def _caminho_melhor_config(arquitetura: str) -> Path:
    return caminho_best_config(arquitetura)


def _salvar_melhor_config(arquitetura: str, hiperparametros: dict, val_mae: float) -> None:
    from datetime import datetime

    dados = {
        "arquitetura": arquitetura,
        "hiperparametros": hiperparametros,
        "val_mae": val_mae,
        "salvo_em": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    caminho = garante_pai(_caminho_melhor_config(arquitetura))
    caminho.write_text(json.dumps(dados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Melhor config Optuna persistida em: {caminho}")


def _oferecer_melhor_config(arquitetura: str) -> dict | None:
    """Se existir config salva pelo Optuna, mostra e pergunta se quer usar."""
    caminho = _caminho_melhor_config(arquitetura)
    if not caminho.exists():
        return None
    dados = _carrega_json(caminho)
    if dados is None:
        return None
    print(f"\nMelhor config Optuna encontrada ({dados.get('salvo_em', '?')}):")
    print(f"  val MAE: {dados['val_mae']:.4f}")
    for chave, valor in dados["hiperparametros"].items():
        print(f"  {chave} = {valor}")
    if _confirma("Usar esta configuracao?"):
        return dados["hiperparametros"]
    return None


def _config_interativa(arquitetura: str) -> TreinoConfig:
    """Monta o TreinoConfig perguntando so o essencial; o resto sao defaults."""
    csv = _escolhe_csv()
    config = TreinoConfig(csv_path=csv, arquitetura=arquitetura)

    if not _confirma("Usar configuracao padrao (epocas, batch, lr, lags)?"):
        config = replace(
            config,
            epocas=_pergunta_int("Epocas", config.epocas),
            batch_size=_pergunta_int("Batch size", config.batch_size),
            raio=_pergunta_int("Raio da janela", config.raio),
            lag_clima=_pergunta_int("Lag do clima (dias)", config.lag_clima),
            lag_historico=_pergunta_int(
                "Lag do historico (dias)", config.lag_historico
            ),
            seed=_pergunta_int("Seed", config.seed),
            split=SplitConfig(),
        )
    return config


def _carrega_json(caminho: Path) -> dict | None:
    try:
        return json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as erro:
        print(f"Nao consegui ler {caminho}: {erro}")
        return None


def _resultado_para_relatorio(caminho: Path) -> Path | None:
    """Adapta o JSON escolhido ao formato que o report espera.

    A saida do tune_runner embrulha a avaliacao final em `resultado_final`;
    o report le o dict de `treina_e_avalia` direto. Se for um JSON de
    otimizacao, extrai o bloco e materializa num arquivo derivado.
    """
    dados = _carrega_json(caminho)
    if dados is None:
        return None
    if "resultado_final" in dados:
        derivado = caminho.with_name(f"{caminho.stem}_final.json")
        derivado.write_text(
            json.dumps(dados["resultado_final"], indent=2), encoding="utf-8"
        )
        print(f"JSON de otimizacao: usando `resultado_final` ({derivado}).")
        return derivado
    return caminho


# ----------------------------------------------------------------- acoes ----


def acao_treinar() -> None:
    arquitetura = _escolhe("Qual arquitetura treinar?", list(ARQUITETURAS))
    melhor = _oferecer_melhor_config(arquitetura)
    if melhor is not None:
        csv = _escolhe_csv()
        config = replace(TreinoConfig(csv_path=csv, arquitetura=arquitetura), **melhor)
    else:
        config = _config_interativa(arquitetura)
    output = garante_pai(
        _pergunta("Salvar resultado em", str(caminho_resultado(arquitetura)))
    )

    from dengue_tl.experiment import formata_resumo, roda_experimento

    print("\nTreinando...")
    resultado = roda_experimento(config)
    print(formata_resumo(resultado["resumo"]))
    output.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nResultado completo salvo em: {output}")

    if _confirma("Gerar tabelas/graficos do relatorio agora?", default=False):
        _gera_relatorio(csv=config.csv_path, resultados=output)


def acao_otimizar() -> None:
    arquitetura = _escolhe(
        "Otimizar hiperparametros de qual arquitetura?", list(ARQUITETURAS)
    )
    config = _config_interativa(arquitetura)
    n_trials = _pergunta_int("Numero de trials", TRIALS_DEFAULT[arquitetura])
    output = garante_pai(
        _pergunta("Salvar resultado em", str(caminho_otimizacao(arquitetura)))
    )

    from dengue_tl.experiment import formata_resumo, resumo_pertinente
    from dengue_tl.tune_runner import otimiza

    print(f"\nRodando {n_trials} trials... (Ctrl+C interrompe)")
    resultado = otimiza(config, n_trials=n_trials)
    output.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _salvar_melhor_config(
        arquitetura, resultado["melhores_hiperparametros"], resultado["melhor_val_mae"]
    )

    print("\nMelhores hiperparametros:")
    for chave, valor in resultado["melhores_hiperparametros"].items():
        print(f"  {chave} = {valor}")
    print(f"MAE de validacao do melhor trial: {resultado['melhor_val_mae']:.4f}\n")
    print(formata_resumo(resumo_pertinente(resultado["resultado_final"])))
    print(f"\nResultado completo salvo em: {output}")


def _gera_relatorio(csv: str, resultados: Path) -> str | None:
    """Gera artefatos de um modelo. Retorna o output_dir usado, ou None em erro."""
    try:
        from dengue_tl.report.artifacts import save_all_report_artifacts
        from dengue_tl.report.data_io import DEFAULT_OUTPUT_DIR
    except ImportError:
        print(
            "Modulo de relatorio indisponivel — instale os extras: "
            'venv/bin/pip install -e ".[report]"'
        )
        return None

    caminho = _resultado_para_relatorio(resultados)
    if caminho is None:
        return None
    output_dir = _pergunta("Pasta de saida", str(DEFAULT_OUTPUT_DIR))
    artefatos = save_all_report_artifacts(
        csv_path=csv, results_path=caminho, output_dir=output_dir
    )
    print("\nArtefatos gerados:")
    for nome, caminho_artefato in artefatos.items():
        print(f"- {nome}: {caminho_artefato}")
    return output_dir


def _gera_comparacao(csv: str, output_dir: str) -> None:
    """Gera tabela + gráfico comparativo entre arquiteturas escolhidas pelo usuário."""
    try:
        from dengue_tl.report.artifacts import save_comparison_artifacts
    except ImportError:
        print("Modulo de relatorio indisponivel.")
        return

    # Monta mapa label -> caminho para todas as arquiteturas com resultado.
    disponiveis: dict[str, Path] = {}
    for arq in ARQUITETURAS:
        resultado = caminho_resultado(arq)
        otimizacao = caminho_otimizacao(arq)
        if resultado.exists():
            disponiveis[arq.upper().replace("_", "-")] = resultado
        elif otimizacao.exists():
            disponiveis[arq.upper().replace("_", "-")] = otimizacao

    if len(disponiveis) < 2:
        print(
            f"Comparativo requer ao menos 2 arquiteturas com resultados. "
            f"Disponíveis: {list(disponiveis) or 'nenhuma'}"
        )
        return

    labels_disponiveis = list(disponiveis)
    escolhidas = _escolhe_multiplos(
        "Quais arquiteturas incluir no comparativo?", labels_disponiveis
    )
    if len(escolhidas) < 2:
        print("Selecione ao menos 2 arquiteturas para o comparativo.")
        return

    caminhos = {label: disponiveis[label] for label in escolhidas}
    print("\nArquiteturas selecionadas:")
    for label, p in caminhos.items():
        print(f"  {label}: {p}")

    artefatos = save_comparison_artifacts(
        csv_path=csv, results_paths=caminhos, output_dir=output_dir
    )
    print("\nArtefatos comparativos:")
    for nome, caminho_artefato in artefatos.items():
        print(f"- {nome}: {caminho_artefato}")


def acao_relatorio() -> None:
    resultados = _escolhe_resultados()
    if resultados is None:
        return
    csv = _escolhe_csv()
    output_dir = _gera_relatorio(csv=csv, resultados=resultados)
    if output_dir and _confirma("Gerar comparativo entre arquiteturas?", default=False):
        _gera_comparacao(csv=csv, output_dir=output_dir)


def acao_comparar() -> None:
    """Gera o comparativo entre todas as arquiteturas com resultados em outputs/."""
    from dengue_tl.paths import OUTPUTS_DIR

    disponiveis = [
        arq for arq in ARQUITETURAS
        if caminho_resultado(arq).exists() or caminho_otimizacao(arq).exists()
    ]
    if len(disponiveis) < 2:
        print(
            f"Comparativo requer ao menos 2 arquiteturas com resultados em outputs/.\n"
            f"  Com resultado: {disponiveis or 'nenhuma'}\n"
            f"  Sem resultado: {[a for a in ARQUITETURAS if a not in disponiveis]}"
        )
        return

    csv = _escolhe_csv()
    try:
        from dengue_tl.report.data_io import DEFAULT_OUTPUT_DIR
        output_dir = _pergunta("Pasta de saida", str(DEFAULT_OUTPUT_DIR))
    except ImportError:
        output_dir = str(OUTPUTS_DIR)
    _gera_comparacao(csv=csv, output_dir=output_dir)


def acao_ver_resumo() -> None:
    caminho = _escolhe_resultados()
    if caminho is None:
        return
    dados = _carrega_json(caminho)
    if dados is None:
        return

    from dengue_tl.experiment import formata_resumo, resumo_pertinente

    if "resultado_final" in dados:  # saida do tune_runner
        print("\nMelhores hiperparametros:")
        for chave, valor in dados["melhores_hiperparametros"].items():
            print(f"  {chave} = {valor}")
        print(f"MAE de validacao do melhor trial: {dados['melhor_val_mae']:.4f}\n")
        dados = dados["resultado_final"]

    resumo = dados.get("resumo") or resumo_pertinente(dados)
    print(formata_resumo(resumo))


def acao_testes() -> None:
    subprocess.run([sys.executable, "-m", "pytest", "-q"], check=False)


# ------------------------------------------------------------------ loop ----

ACOES = [
    ("Treinar e avaliar um modelo", acao_treinar),
    ("Otimizar hiperparametros (Optuna)", acao_otimizar),
    ("Gerar relatorio (tabelas + graficos)", acao_relatorio),
    ("Gerar comparativo entre arquiteturas", acao_comparar),
    ("Ver resumo de um resultado salvo", acao_ver_resumo),
    ("Rodar os testes", acao_testes),
]


def main() -> None:
    print("=" * 56)
    print("Dengue 9x4 — estimativa de casos (CNN-LSTM / CNN2D / MLP)")
    print("=" * 56)
    while True:
        print("\nO que voce quer fazer?")
        for i, (nome, _) in enumerate(ACOES, start=1):
            print(f"  {i}. {nome}")
        print("  0. Sair")
        try:
            resposta = input("Escolha: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAte mais!")
            return
        if resposta == "0" or resposta.lower() in ("q", "sair"):
            print("Ate mais!")
            return
        if not resposta.isdigit() or not 1 <= int(resposta) <= len(ACOES):
            print("Opcao invalida.")
            continue
        _, acao = ACOES[int(resposta) - 1]
        try:
            acao()
        except (KeyboardInterrupt, EOFError):
            print("\nAcao interrompida — voltando ao menu.")
        except ValueError as erro:
            # Erros de validacao do pipeline (ex.: amostras insuficientes).
            print(f"\nErro: {erro}")


if __name__ == "__main__":
    main()
