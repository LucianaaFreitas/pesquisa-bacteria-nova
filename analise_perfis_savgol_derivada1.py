from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import analise_perfis_nao_supervisionado as base


SAIDA_PADRAO = "dados/analise_perfis_savgol_derivada1"
WINDOW_LENGTH_PADRAO = 15
POLYORDER_PADRAO = 2
DELTA_PADRAO = 1.0
PREFIXOS_SENSIVEIS_VALIDACAO = (
    "554A_24",
    "559_240506",
)


@dataclass(frozen=True)
class ModeloSavgolDerivada1:
    pca: object
    kmeans: object
    k_final: int
    perfil_medio_atcc: np.ndarray
    window_length: int
    polyorder: int
    delta: float


def validar_parametros_savgol(n_bandas: int, window_length: int, polyorder: int) -> None:
    if window_length % 2 == 0:
        raise ValueError("window_length precisa ser ímpar para o Savitzky-Golay.")
    if window_length <= polyorder:
        raise ValueError("window_length precisa ser maior que polyorder.")
    if window_length > n_bandas:
        raise ValueError(
            f"window_length={window_length} é maior que o número de bandas ({n_bandas})."
        )
    if polyorder < 1:
        raise ValueError("polyorder precisa ser maior ou igual a 1.")


def aplicar_savgol_derivada1(
    espectros: dict[str, np.ndarray],
    window_length: int = WINDOW_LENGTH_PADRAO,
    polyorder: int = POLYORDER_PADRAO,
    delta: float = DELTA_PADRAO,
) -> dict[str, np.ndarray]:
    if not espectros:
        raise ValueError("Nenhum espectro recebido para aplicar Savitzky-Golay.")

    n_bandas = next(iter(espectros.values())).shape[1]
    validar_parametros_savgol(n_bandas, window_length, polyorder)

    espectros_derivados: dict[str, np.ndarray] = {}
    for nome, X in sorted(espectros.items()):
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.shape[1] != n_bandas:
            raise ValueError(f"{nome}: número de bandas diferente do esperado ({n_bandas}).")

        X_derivada = savgol_filter(
            X_arr,
            window_length=window_length,
            polyorder=polyorder,
            deriv=1,
            delta=delta,
            axis=1,
            mode="interp",
        )
        espectros_derivados[nome] = np.asarray(X_derivada, dtype=np.float32)
        print(
            f"[SAVGOL] {nome}: derivada_1={espectros_derivados[nome].shape}, "
            f"window={window_length}, polyorder={polyorder}"
        )

    return espectros_derivados


def montar_rotulos_sensiveis_por_prefixo(
    nomes_amostras: Iterable[str],
    prefixos: Iterable[str],
) -> dict[str, str]:
    prefixos_tuple = tuple(prefixos)
    rotulos_extras = {
        nome: "sensivel"
        for nome in sorted(nomes_amostras)
        if nome.startswith(prefixos_tuple)
    }

    if not rotulos_extras:
        print(f"[VALIDACAO] Nenhuma amostra encontrada com os prefixos sensíveis: {prefixos_tuple}")
        return {}

    print("[VALIDACAO] Amostras marcadas como sensíveis somente para métricas externas:")
    for nome in sorted(rotulos_extras):
        print(f"  - {nome}: sensivel")

    return rotulos_extras


def analisar_centroides_bandas_derivada(
    pca: object,
    kmeans: object,
    top_n: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    os.makedirs(base.CAMINHO_SAIDA_ANALISE, exist_ok=True)

    centroides_pca = np.asarray(kmeans.cluster_centers_, dtype=float)
    centroides_derivada = np.asarray(pca.inverse_transform(centroides_pca), dtype=float)
    n_clusters, n_bandas = centroides_derivada.shape

    linhas_centroides: list[dict] = []
    for cid in range(n_clusters):
        for banda_idx in range(n_bandas):
            linhas_centroides.append(
                {
                    "cluster": cid,
                    "banda_indice": banda_idx,
                    "banda_numero": banda_idx + 1,
                    "centroide_derivada_savgol": float(centroides_derivada[cid, banda_idx]),
                }
            )

    amplitude = centroides_derivada.max(axis=0) - centroides_derivada.min(axis=0)
    desvio = centroides_derivada.std(axis=0)
    cluster_min = centroides_derivada.argmin(axis=0)
    cluster_max = centroides_derivada.argmax(axis=0)
    ordem_global = np.argsort(-np.abs(amplitude))

    linhas_bandas: list[dict] = []
    for ranking, banda_idx in enumerate(ordem_global, start=1):
        linhas_bandas.append(
            {
                "ranking": ranking,
                "banda_indice": int(banda_idx),
                "banda_numero": int(banda_idx + 1),
                "amplitude_entre_centroides_derivada": float(amplitude[banda_idx]),
                "amplitude_abs_entre_centroides_derivada": float(abs(amplitude[banda_idx])),
                "desvio_padrao_centroides_derivada": float(desvio[banda_idx]),
                "cluster_menor_valor": int(cluster_min[banda_idx]),
                "cluster_maior_valor": int(cluster_max[banda_idx]),
                "menor_centroide_derivada": float(centroides_derivada[cluster_min[banda_idx], banda_idx]),
                "maior_centroide_derivada": float(centroides_derivada[cluster_max[banda_idx], banda_idx]),
            }
        )

    linhas_pares: list[dict] = []
    for cluster_a in range(n_clusters):
        for cluster_b in range(cluster_a + 1, n_clusters):
            diff = np.abs(centroides_derivada[cluster_a] - centroides_derivada[cluster_b])
            ordem_par = np.argsort(-diff)
            for ranking, banda_idx in enumerate(ordem_par, start=1):
                linhas_pares.append(
                    {
                        "cluster_a": cluster_a,
                        "cluster_b": cluster_b,
                        "ranking_no_par": ranking,
                        "banda_indice": int(banda_idx),
                        "banda_numero": int(banda_idx + 1),
                        "diferenca_derivada_abs": float(diff[banda_idx]),
                        "centroide_a_derivada": float(centroides_derivada[cluster_a, banda_idx]),
                        "centroide_b_derivada": float(centroides_derivada[cluster_b, banda_idx]),
                    }
                )

    df_centroides = pd.DataFrame(linhas_centroides)
    df_bandas = pd.DataFrame(linhas_bandas)
    df_pares = pd.DataFrame(linhas_pares)

    caminho_centroides = os.path.join(base.CAMINHO_SAIDA_ANALISE, "centroides_clusters_bandas_derivada.csv")
    caminho_bandas = os.path.join(base.CAMINHO_SAIDA_ANALISE, "bandas_influentes_centroides_derivada.csv")
    caminho_pares = os.path.join(base.CAMINHO_SAIDA_ANALISE, "bandas_influentes_por_par_clusters_derivada.csv")

    df_centroides.to_csv(caminho_centroides, index=False)
    df_bandas.to_csv(caminho_bandas, index=False)
    df_pares.to_csv(caminho_pares, index=False)

    print(f"[SAIDA] {caminho_centroides}")
    print(f"[SAIDA] {caminho_bandas}")
    print(f"[SAIDA] {caminho_pares}")

    top = df_bandas.head(min(top_n, len(df_bandas))).sort_values(
        "amplitude_abs_entre_centroides_derivada",
        ascending=True,
    )
    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(top) + 2)))
    ax.barh(
        top["banda_numero"].astype(str),
        top["amplitude_abs_entre_centroides_derivada"],
        color="darkorange",
        alpha=0.85,
    )
    ax.set_xlabel("Diferença absoluta entre centroides derivados")
    ax.set_ylabel("Banda espectral")
    ax.set_title(f"Top {len(top)} bandas mais influentes - Savitzky-Golay derivada 1")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    base._salvar_figura(fig, "top_bandas_influentes_centroides_derivada")

    fig, ax = plt.subplots(figsize=(12, 5))
    eixo_bandas = np.arange(1, n_bandas + 1)
    for cid in range(n_clusters):
        ax.plot(eixo_bandas, centroides_derivada[cid], linewidth=1.6, label=f"Cluster {cid}")
    for banda_numero in df_bandas.head(min(10, len(df_bandas)))["banda_numero"]:
        ax.axvline(int(banda_numero), color="black", alpha=0.08, linewidth=1)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xlabel("Banda espectral")
    ax.set_ylabel("Centroide no espaço da derivada Savitzky-Golay")
    ax.set_title("Perfis derivados dos centroides dos clusters")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    base._salvar_figura(fig, "centroides_espectrais_derivada_clusters")

    return df_centroides, df_bandas, df_pares


def salvar_modelo_e_config_savgol(
    pca: object,
    kmeans: object,
    k_final: int,
    resultado_k: dict,
    perfil_medio_atcc: np.ndarray,
    k_fixo: int | None,
    window_length: int,
    polyorder: int,
    delta: float,
    rotulos_validacao_extra: dict[str, str] | None = None,
) -> None:
    os.makedirs(base.CAMINHO_SAIDA_ANALISE, exist_ok=True)
    np.save(os.path.join(base.CAMINHO_SAIDA_ANALISE, "perfil_medio_atcc.npy"), perfil_medio_atcc)

    preprocessamento = {
        "tipo": "savitzky_golay_derivada_primeira_ordem",
        "substitui_minmax": True,
        "window_length": window_length,
        "polyorder": polyorder,
        "deriv": 1,
        "delta": delta,
        "axis": 1,
        "mode": "interp",
    }

    config = {
        "abordagem": "nao_supervisionada_savgol_derivada1_com_referencia_atcc_pos_hoc",
        "usa_rotulos_no_treino": False,
        "usa_todas_as_imagens_no_agrupamento": True,
        "preprocessamento": preprocessamento,
        "rotulos_usados_apenas_na_validacao_externa": base.CAMINHO_ROTULOS,
        "rotulos_validacao_extra": rotulos_validacao_extra or {},
        "metricas_validacao_externa": ["ARI", "NMI", "pureza_global_clusters", "pureza_por_cluster"],
        "relatorios_adicionais": [
            "estabilidade_por_k.csv",
            "distancias_euclidianas_por_imagem.csv",
            "distancias_euclidianas_por_cluster.csv",
            "distancias_euclidianas_entre_centroides.csv",
            "centroides_clusters_bandas_derivada.csv",
            "bandas_influentes_centroides_derivada.csv",
            "bandas_influentes_por_par_clusters_derivada.csv",
            "PCA 2D e t-SNE 2D coloridos por cluster e rotulo",
        ],
        "k_final": k_final,
        "k_criterio_final": resultado_k["criterio_final"],
        "k_fixo": k_fixo,
        "pca_variancia": base.PCA_VARIANCIA,
        "random_state": base.RANDOM_STATE,
        "kmeans_random_state": base.KMEANS_RANDOM_STATE,
        "observacao": (
            "Savitzky-Golay com derivada de primeira ordem substitui Min-Max. "
            "ATCC e rotulos extras sao usados somente apos o clustering."
        ),
    }

    with open(os.path.join(base.CAMINHO_SAIDA_ANALISE, "config_analise.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    with open(os.path.join(base.CAMINHO_SAIDA_ANALISE, "modelo.pkl"), "wb") as f:
        pickle.dump(
            {
                "preprocessamento": preprocessamento,
                "pca": pca,
                "kmeans": kmeans,
                "k_final": k_final,
                "perfil_medio_atcc": perfil_medio_atcc,
            },
            f,
        )


def executar_analise_savgol_derivada1(
    k_fixo: int | None = None,
    saida: str = SAIDA_PADRAO,
    window_length: int = WINDOW_LENGTH_PADRAO,
    polyorder: int = POLYORDER_PADRAO,
    delta: float = DELTA_PADRAO,
    rotulos_validacao_extra: dict[str, str] | None = None,
) -> ModeloSavgolDerivada1:
    base.configurar_saida(saida)

    print("[PIPELINE-SAVGOL] Lendo dados processados.")
    espectros, mascaras = base.ler_espectros_e_mascaras()

    print("\n[SAVGOL] Aplicando Savitzky-Golay com derivada de primeira ordem.")
    espectros_derivados = aplicar_savgol_derivada1(
        espectros,
        window_length=window_length,
        polyorder=polyorder,
        delta=delta,
    )
    X_total, _, fatias = base.concatenar_espectros(espectros_derivados)
    print(f"[CONCAT] X_total_derivada={X_total.shape}")

    print("\n[PCA] Ajustando PCA global sobre espectros derivados, sem rotulos.")
    pca, Z_total = base.executar_pca_global(X_total)

    print("\n[K-SELECAO] Avaliando k sem rotulos.")
    resultado_k, k_final = base.avaliar_k(Z_total, k_fixo=k_fixo)
    base.salvar_relatorio_escolha_k(resultado_k)
    base.plotar_escolha_k(resultado_k)

    print(f"\n[KMEANS] Treinando K-Means global sem rotulos (k={k_final}).")
    kmeans = base.treinar_kmeans(Z_total, k_final)
    labels_total = kmeans.predict(Z_total)
    np.save(os.path.join(base.CAMINHO_SAIDA_ANALISE, "labels_total.npy"), labels_total)

    print("\n[DISTANCIAS] Calculando distancias Euclidianas aos centroides.")
    base.salvar_distancias_euclidianas(Z_total, labels_total, fatias, kmeans)

    print("\n[ESTABILIDADE] Executando K-Means completo para diferentes valores de k.")
    base.calcular_estabilidade_multiplos_k(
        Z_total,
        fatias,
        ks=resultado_k["ks"],
        rotulos_extras=rotulos_validacao_extra,
    )

    print("\n[PERFIS] Calculando assinaturas por imagem e similaridade com ATCC.")
    df_perfis = base.calcular_perfis_por_imagem(labels_total, fatias, k_final)
    df_perfis, perfil_medio_atcc = base.adicionar_similaridade_atcc(df_perfis, k_final)
    base.salvar_perfis(df_perfis, perfil_medio_atcc)

    mapas = base.criar_mapas_labels(labels_total, fatias, mascaras, k_final)
    base.plotar_mosaico_clusters(mapas, k_final)
    base.plotar_similaridade_atcc(df_perfis)
    base.plotar_perfis_clusters(df_perfis, k_final)

    print("\n[PROJECAO] Gerando PCA 2D e t-SNE 2D coloridos por cluster e rotulo.")
    rotulos_pixels = base.rotulos_para_visualizacao(fatias, rotulos_extras=rotulos_validacao_extra)
    base.plotar_projecao_2d(Z_total[:, :2], labels_total, rotulos_pixels, "PCA 2D")
    base.plotar_tsne_2d(Z_total, labels_total, rotulos_pixels)

    print("\n[VALIDACAO] Calculando ARI, NMI e pureza dos clusters com rotulos conhecidos.")
    df_validacao, df_pureza_clusters, df_imagens_rotuladas = base.calcular_validacao_externa(
        labels_total,
        fatias,
        k_final,
        rotulos_extras=rotulos_validacao_extra,
    )
    base.salvar_validacao_externa(df_validacao, df_pureza_clusters, df_imagens_rotuladas)

    print("\n[CENTROIDES] Analisando bandas mais influentes no espaço da derivada.")
    analisar_centroides_bandas_derivada(pca, kmeans)

    salvar_modelo_e_config_savgol(
        pca,
        kmeans,
        k_final,
        resultado_k,
        perfil_medio_atcc,
        k_fixo,
        window_length,
        polyorder,
        delta,
        rotulos_validacao_extra=rotulos_validacao_extra,
    )

    print("\n[PIPELINE-SAVGOL] Análise não supervisionada com derivada finalizada com sucesso.")
    print(f"[PIPELINE-SAVGOL] Saídas em: {base.CAMINHO_SAIDA_ANALISE}")

    return ModeloSavgolDerivada1(
        pca=pca,
        kmeans=kmeans,
        k_final=k_final,
        perfil_medio_atcc=perfil_medio_atcc,
        window_length=window_length,
        polyorder=polyorder,
        delta=delta,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Analise nao supervisionada usando Savitzky-Golay com derivada de primeira ordem "
            "no lugar da normalizacao Min-Max."
        )
    )
    parser.add_argument("--k-fixo", type=int, default=None, help="Força um valor de k, ignorando o critério automático.")
    parser.add_argument("--saida", default=SAIDA_PADRAO, help=f"Pasta de saída. Padrão: {SAIDA_PADRAO}")
    parser.add_argument("--window-length", type=int, default=WINDOW_LENGTH_PADRAO, help="Janela ímpar do Savitzky-Golay.")
    parser.add_argument("--polyorder", type=int, default=POLYORDER_PADRAO, help="Ordem do polinômio do Savitzky-Golay.")
    parser.add_argument("--delta", type=float, default=DELTA_PADRAO, help="Espaçamento entre bandas para a derivada.")
    parser.add_argument(
        "--sem-validacao-sensiveis",
        action="store_true",
        help="Não adiciona 554A_24* e 559_240506* como sensíveis na validação externa.",
    )
    parser.add_argument(
        "--prefixo-sensivel",
        action="append",
        default=None,
        help="Prefixo extra para marcar como sensível somente na validação externa. Pode repetir.",
    )

    args = parser.parse_args()

    espectros_tmp, _ = base.ler_espectros_e_mascaras()
    prefixos = args.prefixo_sensivel or list(PREFIXOS_SENSIVEIS_VALIDACAO)
    rotulos_extras = (
        {}
        if args.sem_validacao_sensiveis
        else montar_rotulos_sensiveis_por_prefixo(espectros_tmp.keys(), prefixos)
    )

    executar_analise_savgol_derivada1(
        k_fixo=args.k_fixo,
        saida=args.saida,
        window_length=args.window_length,
        polyorder=args.polyorder,
        delta=args.delta,
        rotulos_validacao_extra=rotulos_extras,
    )
