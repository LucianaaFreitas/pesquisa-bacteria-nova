from __future__ import annotations

import argparse

import analise_perfis_nao_supervisionado as pipeline


# Rótulos usados somente na etapa de validação externa.
# Eles não são usados no Min-Max, PCA, escolha de k ou K-Means.
PREFIXOS_SENSIVEIS_VALIDACAO = (
    "554A_24",
    "559_240506",
)

SAIDA_PADRAO = "dados/analise_perfis_validacao_sensiveis_554A_559"


def montar_rotulos_sensiveis_por_prefixo() -> dict[str, str]:
    espectros, _ = pipeline.ler_espectros_e_mascaras()
    rotulos_extras = {
        nome: "sensivel"
        for nome in sorted(espectros.keys())
        if nome.startswith(PREFIXOS_SENSIVEIS_VALIDACAO)
    }

    if not rotulos_extras:
        raise ValueError(
            "Nenhuma amostra encontrada com os prefixos sensíveis configurados: "
            f"{PREFIXOS_SENSIVEIS_VALIDACAO}"
        )

    print("[VALIDACAO] Amostras marcadas como sensíveis apenas para as métricas:")
    for nome in sorted(rotulos_extras):
        print(f"  - {nome}: sensivel")

    return rotulos_extras


def executar_analise_validacao_sensiveis(k_fixo: int | None = None, saida: str = SAIDA_PADRAO):
    rotulos_extras = montar_rotulos_sensiveis_por_prefixo()
    return pipeline.executar_analise(
        k_fixo=k_fixo,
        saida=saida,
        rotulos_validacao_extra=rotulos_extras,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Executa a análise não supervisionada completa, mas adiciona 554A_24* "
            "e 559_240506* como sensíveis somente para ARI, NMI e pureza."
        )
    )
    parser.add_argument("--k-fixo", type=int, default=None, help="Força um valor de k, ignorando o critério automático.")
    parser.add_argument(
        "--saida",
        default=SAIDA_PADRAO,
        help="Pasta de saída. Ex.: dados/analise_perfis_validacao_sensiveis_554A_559_k4",
    )
    args = parser.parse_args()
    executar_analise_validacao_sensiveis(k_fixo=args.k_fixo, saida=args.saida)
