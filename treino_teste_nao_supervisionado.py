from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


# --- Caminhos
CAMINHO_IMGS_PROCESSADAS = "dados/img_processadas"
CAMINHO_SAIDA_EXPERIMENTO = "dados/experimento_treino_teste"
CAMINHO_SAIDA_FIGURAS = os.path.join(CAMINHO_SAIDA_EXPERIMENTO, "figuras")

# --- Split fixo do experimento
# As imagens abaixo ficam totalmente fora do ajuste do Min-Max, PCA, escolha de k e K-Means.
TESTE_ATIVA = "ATCC13_240506-161053"
TESTE_NORMAIS = [
    "3491B_240506-155911",
    "351_1_240506-160920",
]

# --- Parâmetros de treino
K_MIN = 2
K_MAX = 10
K_AMOSTRAS_ESCOLHA_K = 10_000
K_CRITERIO_FINAL = "silhouette"  # "silhouette" | "cotovelo" | "media"
KMEANS_N_INIT = 20
KMEANS_RANDOM_STATE = 42
PCA_VARIANCIA = 0.95
RANDOM_STATE = 42
MOSAICO_COLS = 4
ATCC_CLUSTER_MIN_FRAC = 0.05
LIMIAR_DECISAO_ATIVA = 0.50

_FIG_COUNTER = 0


def configurar_saida(caminho_saida: str) -> None:
    global CAMINHO_SAIDA_EXPERIMENTO, CAMINHO_SAIDA_FIGURAS, _FIG_COUNTER
    CAMINHO_SAIDA_EXPERIMENTO = caminho_saida
    CAMINHO_SAIDA_FIGURAS = os.path.join(CAMINHO_SAIDA_EXPERIMENTO, "figuras")
    _FIG_COUNTER = 0


@dataclass(frozen=True)
class ModeloTreinado:
    min_global: np.ndarray
    max_global: np.ndarray
    pca: object
    kmeans: object
    k_final: int
    cluster_ativa: int
    clusters_ativos: list[int]
    cluster_normal: int | None


def _sanitize_filename(texto: str) -> str:
    texto = texto.strip().replace(" ", "_")
    allowed = [ch for ch in texto if ch.isalnum() or ch in {"-", "_"}]
    out = "".join(allowed)
    return out[:120] if out else "fig"


def _salvar_figura(fig: plt.Figure, nome_base: str) -> None:
    global _FIG_COUNTER
    _FIG_COUNTER += 1
    os.makedirs(CAMINHO_SAIDA_FIGURAS, exist_ok=True)
    nome = f"{_FIG_COUNTER:04d}_{_sanitize_filename(nome_base)}.png"
    fig.savefig(os.path.join(CAMINHO_SAIDA_FIGURAS, nome), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _criar_cmap_discreta(k: int, base_cmap: str = "gist_ncar") -> ListedColormap:
    cmap_base = plt.get_cmap(base_cmap)
    cores = cmap_base(np.linspace(0.0, 1.0, k))
    return ListedColormap(cores, name=f"{base_cmap}_{k}")


def ler_espectros_e_mascaras(
    pasta_processadas: str = CAMINHO_IMGS_PROCESSADAS,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    csv_paths = sorted(glob.glob(os.path.join(pasta_processadas, "**", "espectros.csv"), recursive=True))
    if not csv_paths:
        raise FileNotFoundError(f"Nenhum espectros.csv encontrado em '{pasta_processadas}'.")

    espectros: dict[str, np.ndarray] = {}
    mascaras: dict[str, np.ndarray] = {}

    for csv_path in csv_paths:
        pasta_img = os.path.dirname(csv_path)
        nome = os.path.basename(pasta_img)
        mask_path = os.path.join(pasta_img, "mascara_roi.npy")

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Máscara não encontrada para '{nome}': {mask_path}")

        mascara = np.load(mask_path).astype(bool)
        X = pd.read_csv(csv_path).values.astype(np.float32)

        if X.shape[0] != int(np.count_nonzero(mascara)):
            raise ValueError(f"{nome}: espectros.csv não bate com os pixels ativos da máscara.")

        espectros[nome] = X
        mascaras[nome] = mascara
        print(f"[LEITURA] {nome}: espectros={X.shape}, mascara={mascara.shape}")

    return espectros, mascaras


def separar_treino_teste(
    espectros: dict[str, np.ndarray],
    mascaras: dict[str, np.ndarray],
    teste_ativa: str = TESTE_ATIVA,
    teste_normais: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    teste_normais = teste_normais or TESTE_NORMAIS
    nomes_teste = [teste_ativa, *teste_normais]
    faltantes = [nome for nome in nomes_teste if nome not in espectros]
    if faltantes:
        raise KeyError(f"Imagens de teste não encontradas nos dados processados: {faltantes}")

    espectros_teste = {nome: espectros[nome] for nome in nomes_teste}
    mascaras_teste = {nome: mascaras[nome] for nome in nomes_teste}
    espectros_treino = {nome: X for nome, X in espectros.items() if nome not in nomes_teste}
    mascaras_treino = {nome: m for nome, m in mascaras.items() if nome not in nomes_teste}

    if not any(nome.startswith("ATCC") for nome in espectros_treino):
        raise ValueError("O treino precisa manter pelo menos uma ATCC para interpretar o cluster ativo.")

    print(f"[SPLIT] Treino: {len(espectros_treino)} imagens")
    print(f"[SPLIT] Teste: {len(espectros_teste)} imagens ({', '.join(nomes_teste)})")
    return espectros_treino, mascaras_treino, espectros_teste, mascaras_teste


def ajustar_minmax_treino(espectros_treino: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    X_treino_total = np.vstack([np.asarray(X, dtype=np.float32) for X in espectros_treino.values()])
    min_global = X_treino_total.min(axis=0)
    max_global = X_treino_total.max(axis=0)
    return min_global, max_global


def aplicar_minmax(
    espectros: dict[str, np.ndarray],
    min_global: np.ndarray,
    max_global: np.ndarray,
) -> dict[str, np.ndarray]:
    den = np.where((max_global - min_global) == 0, 1e-8, max_global - min_global)
    return {
        nome: np.clip((np.asarray(X, dtype=np.float32) - min_global) / den, 0.0, 1.0)
        for nome, X in espectros.items()
    }


def concatenar_espectros(espectros_norm: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, slice]]:
    nomes_ord = sorted(espectros_norm.keys())
    blocos_x: list[np.ndarray] = []
    blocos_nome: list[np.ndarray] = []
    fatias: dict[str, slice] = {}
    offset = 0

    for nome in nomes_ord:
        Xn = np.asarray(espectros_norm[nome], dtype=np.float32)
        n = int(Xn.shape[0])
        blocos_x.append(Xn)
        blocos_nome.append(np.asarray([nome] * n, dtype=object))
        fatias[nome] = slice(offset, offset + n)
        offset += n

    return np.vstack(blocos_x), np.concatenate(blocos_nome, axis=0), fatias


def executar_pca_treino(X_treino: np.ndarray) -> tuple[object, np.ndarray]:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=float(PCA_VARIANCIA), random_state=RANDOM_STATE)
    Z_treino = pca.fit_transform(X_treino)
    print(f"[PCA] Componentes mantidos no treino: {Z_treino.shape[1]}")
    return pca, Z_treino


def _subamostrar(Z: np.ndarray, max_amostras: int = K_AMOSTRAS_ESCOLHA_K) -> np.ndarray:
    n = Z.shape[0]
    if n <= max_amostras:
        return Z
    idx = np.sort(np.random.default_rng(RANDOM_STATE).choice(n, size=max_amostras, replace=False))
    return Z[idx]


def _intervalo_k(n_amostras: int) -> list[int]:
    k_superior = min(K_MAX, max(K_MIN, n_amostras // 500))
    return list(range(K_MIN, k_superior + 1))


def _detectar_k_cotovelo(ks: list[int], inertias: list[float]) -> int:
    if len(ks) <= 1:
        return ks[0]

    x = np.asarray(ks, dtype=float)
    y = np.asarray(inertias, dtype=float)
    x1, y1, x2, y2 = x[0], y[0], x[-1], y[-1]
    denom = np.hypot(y2 - y1, x2 - x1)
    if denom <= 1e-12:
        return int(ks[0])

    dists = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
    return int(ks[int(np.argmax(dists))])


def avaliar_k_no_treino(Z_treino: np.ndarray, k_fixo: int | None = None) -> tuple[dict, int]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    Z_sub = _subamostrar(Z_treino)
    ks = _intervalo_k(Z_sub.shape[0])
    inertias: list[float] = []
    silhouettes: list[float] = []

    for k in ks:
        km = KMeans(
            n_clusters=int(k),
            init="k-means++",
            n_init=KMEANS_N_INIT,
            random_state=KMEANS_RANDOM_STATE,
            algorithm="lloyd",
        )
        labels = km.fit_predict(Z_sub)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(Z_sub, labels, metric="euclidean")))
        print(f"[K-SELECAO] k={k}: inercia={inertias[-1]:.2f}, silhouette={silhouettes[-1]:.4f}")

    k_cotovelo = _detectar_k_cotovelo(ks, inertias)
    k_silhouette = int(ks[int(np.argmax(silhouettes))])

    if k_fixo is not None:
        if k_fixo not in ks:
            raise ValueError(f"k_fixo={k_fixo} fora do intervalo avaliado: {ks}")
        k_final = int(k_fixo)
        criterio_final = "fixo"
    elif K_CRITERIO_FINAL == "cotovelo":
        k_final = k_cotovelo
        criterio_final = K_CRITERIO_FINAL
    elif K_CRITERIO_FINAL == "media":
        k_final = int(round((k_cotovelo + k_silhouette) / 2))
        k_final = max(K_MIN, min(k_final, max(ks)))
        criterio_final = K_CRITERIO_FINAL
    else:
        k_final = k_silhouette
        criterio_final = K_CRITERIO_FINAL

    resultado_k = {
        "ks": ks,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "k_cotovelo": k_cotovelo,
        "k_silhouette": k_silhouette,
        "k_final": k_final,
        "criterio_final": criterio_final,
        "n_amostras_avaliacao": int(Z_sub.shape[0]),
    }
    print(f"[K-SELECAO] k_final={k_final} (criterio='{criterio_final}')")
    return resultado_k, k_final


def salvar_relatorio_escolha_k(resultado_k: dict) -> None:
    os.makedirs(CAMINHO_SAIDA_EXPERIMENTO, exist_ok=True)
    linhas = []
    for k, inertia, sil in zip(resultado_k["ks"], resultado_k["inertias"], resultado_k["silhouettes"]):
        linhas.append(
            {
                "k": k,
                "inercia": inertia,
                "silhouette": sil,
                "k_cotovelo": resultado_k["k_cotovelo"],
                "k_silhouette": resultado_k["k_silhouette"],
                "k_final": resultado_k["k_final"],
                "criterio_final": resultado_k["criterio_final"],
                "n_amostras_avaliacao": resultado_k["n_amostras_avaliacao"],
            }
        )
    caminho = os.path.join(CAMINHO_SAIDA_EXPERIMENTO, "escolha_k_treino.csv")
    pd.DataFrame(linhas).to_csv(caminho, index=False)
    print(f"[SAIDA] Relatório de k salvo em {caminho}")


def plotar_escolha_k(resultado_k: dict) -> None:
    ks = resultado_k["ks"]
    inertias = resultado_k["inertias"]
    silhouettes = resultado_k["silhouettes"]
    k_cotovelo = resultado_k["k_cotovelo"]
    k_silhouette = resultado_k["k_silhouette"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ks, inertias, "o-", color="steelblue")
    axes[0].axvline(k_cotovelo, color="crimson", linestyle="--", label=f"cotovelo k={k_cotovelo}")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inércia (WCSS)")
    axes[0].set_title("Escolha de k no treino - Cotovelo")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ks, silhouettes, "o-", color="darkgreen")
    axes[1].axvline(k_silhouette, color="crimson", linestyle="--", label=f"max silhouette k={k_silhouette}")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette médio")
    axes[1].set_title("Escolha de k no treino - Silhouette")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _salvar_figura(fig, "escolha_k_treino")


def treinar_kmeans(Z_treino: np.ndarray, k: int) -> object:
    from sklearn.cluster import KMeans

    km = KMeans(
        n_clusters=int(k),
        init="k-means++",
        n_init=KMEANS_N_INIT,
        random_state=KMEANS_RANDOM_STATE,
        algorithm="lloyd",
    )
    km.fit(Z_treino)
    return km


def interpretar_clusters_ativos(labels_treino: np.ndarray, fatias_treino: dict[str, slice], k: int) -> tuple[int, list[int], int | None]:
    atcc_treino = sorted(nome for nome in fatias_treino if nome.startswith("ATCC"))
    if not atcc_treino:
        raise ValueError("Nenhuma ATCC ficou no treino para interpretar o cluster ativo.")

    soma = {cid: 0 for cid in range(k)}
    total_atcc = 0
    for nome in atcc_treino:
        labs = labels_treino[fatias_treino[nome]]
        contagens = {cid: int(np.sum(labs == cid)) for cid in range(k)}
        for cid, count in contagens.items():
            soma[cid] += count
        total_atcc += int(labs.shape[0])
        total = max(1, labs.shape[0])
        dist = ", ".join(f"cluster{cid}={contagens[cid] / total:.3f}" for cid in range(k))
        print(f"[INTERPRETACAO] ATCC treino {nome}: {dist}")

    cluster_ativa = max(soma, key=soma.get)
    frac_atcc = {cid: soma[cid] / max(1, total_atcc) for cid in range(k)}
    clusters_ativos = [cid for cid, frac in frac_atcc.items() if frac >= ATCC_CLUSTER_MIN_FRAC]
    if not clusters_ativos:
        clusters_ativos = [int(cluster_ativa)]

    cluster_normal = next((cid for cid in range(k) if cid != cluster_ativa), None) if k == 2 else None
    ativos_str = ", ".join(f"{cid} ({frac_atcc[cid]:.3f})" for cid in clusters_ativos)
    print(f"[INTERPRETACAO] Cluster ativo principal inferido pelas ATCC de treino: {cluster_ativa}")
    print(f"[INTERPRETACAO] Conjunto de clusters ativos: {ativos_str}")
    return int(cluster_ativa), [int(cid) for cid in clusters_ativos], cluster_normal


def _labels_para_mapa_2d(labels_roi: np.ndarray, mascara_roi: np.ndarray) -> np.ndarray:
    label_map_2d = np.full(mascara_roi.shape, -1, dtype=np.int64)
    label_map_2d[mascara_roi] = labels_roi.reshape(-1)
    return label_map_2d


def criar_mapas_labels(
    labels: np.ndarray,
    fatias: dict[str, slice],
    mascaras: dict[str, np.ndarray],
    sufixo_titulo: str,
) -> dict[str, np.ndarray]:
    mapas: dict[str, np.ndarray] = {}
    for nome in sorted(fatias.keys()):
        mapa = _labels_para_mapa_2d(labels[fatias[nome]], mascaras[nome])
        mapa_plot = mapa.astype(np.float32)
        mapa_plot[mapa < 0] = np.nan
        mapas[f"{nome} {sufixo_titulo}"] = mapa_plot
    return mapas


def plotar_mosaico_clusters(mapas: dict[str, np.ndarray], k: int, titulo: str) -> None:
    if not mapas:
        return

    n = len(mapas)
    rows = (n + MOSAICO_COLS - 1) // MOSAICO_COLS
    fig, axes = plt.subplots(rows, MOSAICO_COLS, figsize=(MOSAICO_COLS * 4, rows * 4))
    axes = np.atleast_1d(axes).flatten()
    cmap = _criar_cmap_discreta(max(2, k))
    cmap.set_bad(color="white")

    for i, (nome, mapa) in enumerate(mapas.items()):
        axes[i].imshow(np.ma.masked_invalid(mapa), cmap=cmap, vmin=-0.5, vmax=k - 0.5)
        axes[i].set_title(nome[:28], fontsize=8)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(titulo, y=0.98)
    plt.tight_layout()
    _salvar_figura(fig, titulo)


def resumir_por_imagem(
    labels: np.ndarray,
    fatias: dict[str, slice],
    conjunto: str,
    cluster_ativa: int,
    clusters_ativos: list[int],
    k: int,
    esperados: dict[str, str],
) -> list[dict]:
    linhas: list[dict] = []

    for nome in sorted(fatias.keys()):
        labs = labels[fatias[nome]]
        total = int(labs.shape[0])
        contagens = {cid: int(np.sum(labs == cid)) for cid in range(k)}
        frac_ativa = contagens.get(cluster_ativa, 0) / max(1, total)
        frac_clusters_ativos = sum(contagens.get(cid, 0) for cid in clusters_ativos) / max(1, total)
        cluster_dominante = max(contagens, key=contagens.get)
        decisao = "ativa" if frac_clusters_ativos >= LIMIAR_DECISAO_ATIVA else "normal"
        esperado = esperados.get(nome, "treino")
        acerto = "" if esperado == "treino" else str(decisao == esperado)

        linha = {
            "imagem": nome,
            "conjunto": conjunto,
            "esperado": esperado,
            "decisao": decisao,
            "acerto": acerto,
            "cluster_dominante": cluster_dominante,
            "cluster_ativa": cluster_ativa,
            "clusters_ativos": "|".join(str(cid) for cid in clusters_ativos),
            "frac_cluster_ativa": frac_ativa,
            "frac_clusters_ativos": frac_clusters_ativos,
            "pureza": max(contagens.values()) / max(1, total),
            "n_pixels": total,
        }
        for cid in range(k):
            linha[f"cluster_{cid}_pixels"] = contagens.get(cid, 0)
            linha[f"cluster_{cid}_frac"] = contagens.get(cid, 0) / max(1, total)
        linhas.append(linha)

    return linhas


def salvar_resultados(linhas: list[dict], nome_arquivo: str) -> None:
    os.makedirs(CAMINHO_SAIDA_EXPERIMENTO, exist_ok=True)
    caminho = os.path.join(CAMINHO_SAIDA_EXPERIMENTO, nome_arquivo)
    df = pd.DataFrame(linhas)
    df.to_csv(caminho, index=False)
    print(f"[SAIDA] {caminho}")
    print(df.to_string(index=False))


def plotar_resultado_teste(df_teste: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, max(4, 0.7 * len(df_teste))))
    y_pos = np.arange(len(df_teste))
    coluna_frac = "frac_clusters_ativos" if "frac_clusters_ativos" in df_teste.columns else "frac_cluster_ativa"
    ax.barh(y_pos, df_teste[coluna_frac].values, color="darkgreen", alpha=0.75)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_teste["imagem"].tolist(), fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fração dos pixels no cluster ativo")
    ax.set_title("Resultado das imagens de teste")
    ax.grid(axis="x", alpha=0.3)

    for y, valor, decisao in zip(y_pos, df_teste[coluna_frac].values, df_teste["decisao"].values):
        ax.text(min(valor + 0.02, 0.98), y, f"{valor:.1%} - {decisao}", va="center", fontsize=8)

    plt.tight_layout()
    _salvar_figura(fig, "resultado_teste_frac_cluster_ativa")


def executar_experimento(k_fixo: int | None = None, saida_experimento: str | None = None) -> ModeloTreinado:
    if saida_experimento is not None:
        configurar_saida(saida_experimento)

    print("[PIPELINE] Lendo dados processados.")
    espectros, mascaras = ler_espectros_e_mascaras()

    espectros_treino, mascaras_treino, espectros_teste, mascaras_teste = separar_treino_teste(
        espectros=espectros,
        mascaras=mascaras,
    )

    print("\n[NORMALIZACAO] Ajustando Min-Max apenas no treino.")
    min_global, max_global = ajustar_minmax_treino(espectros_treino)
    espectros_treino_norm = aplicar_minmax(espectros_treino, min_global, max_global)
    espectros_teste_norm = aplicar_minmax(espectros_teste, min_global, max_global)

    X_treino, _, fatias_treino = concatenar_espectros(espectros_treino_norm)
    X_teste, _, fatias_teste = concatenar_espectros(espectros_teste_norm)
    print(f"[CONCAT] Treino: X={X_treino.shape}; Teste: X={X_teste.shape}")

    print("\n[PCA] Ajustando PCA apenas no treino.")
    pca, Z_treino = executar_pca_treino(X_treino)
    Z_teste = pca.transform(X_teste)

    print("\n[K-SELECAO] Escolhendo k apenas no treino.")
    resultado_k, k_final = avaliar_k_no_treino(Z_treino, k_fixo=k_fixo)
    salvar_relatorio_escolha_k(resultado_k)
    plotar_escolha_k(resultado_k)

    print(f"\n[KMEANS] Treinando K-Means apenas no treino (k={k_final}).")
    kmeans = treinar_kmeans(Z_treino, k_final)
    labels_treino = kmeans.predict(Z_treino)
    labels_teste = kmeans.predict(Z_teste)

    cluster_ativa, clusters_ativos, cluster_normal = interpretar_clusters_ativos(labels_treino, fatias_treino, k_final)

    esperados_teste = {TESTE_ATIVA: "ativa", **{nome: "normal" for nome in TESTE_NORMAIS}}
    linhas_treino = resumir_por_imagem(labels_treino, fatias_treino, "treino", cluster_ativa, clusters_ativos, k_final, {})
    linhas_teste = resumir_por_imagem(labels_teste, fatias_teste, "teste", cluster_ativa, clusters_ativos, k_final, esperados_teste)
    salvar_resultados(linhas_treino, "resultados_treino.csv")
    salvar_resultados(linhas_teste, "resultados_teste.csv")
    plotar_resultado_teste(pd.DataFrame(linhas_teste))

    mapas_treino = criar_mapas_labels(labels_treino, fatias_treino, mascaras_treino, f"(treino, k={k_final})")
    mapas_teste = criar_mapas_labels(labels_teste, fatias_teste, mascaras_teste, f"(teste, k={k_final})")
    plotar_mosaico_clusters(mapas_treino, k_final, f"mosaico_clusters_treino_k_{k_final}")
    plotar_mosaico_clusters(mapas_teste, k_final, f"mosaico_clusters_teste_k_{k_final}")

    np.save(os.path.join(CAMINHO_SAIDA_EXPERIMENTO, "min_global.npy"), min_global)
    np.save(os.path.join(CAMINHO_SAIDA_EXPERIMENTO, "max_global.npy"), max_global)
    np.save(os.path.join(CAMINHO_SAIDA_EXPERIMENTO, "labels_treino.npy"), labels_treino)
    np.save(os.path.join(CAMINHO_SAIDA_EXPERIMENTO, "labels_teste.npy"), labels_teste)

    config = {
        "teste_ativa": TESTE_ATIVA,
        "teste_normais": TESTE_NORMAIS,
        "treino_imagens": sorted(espectros_treino.keys()),
        "k_final": k_final,
        "cluster_ativa": cluster_ativa,
        "clusters_ativos": clusters_ativos,
        "cluster_normal": cluster_normal,
        "pca_variancia": PCA_VARIANCIA,
        "k_criterio_final": resultado_k["criterio_final"],
        "k_fixo": k_fixo,
        "atcc_cluster_min_frac": ATCC_CLUSTER_MIN_FRAC,
        "limiar_decisao_ativa": LIMIAR_DECISAO_ATIVA,
        "random_state": RANDOM_STATE,
        "kmeans_random_state": KMEANS_RANDOM_STATE,
    }
    with open(os.path.join(CAMINHO_SAIDA_EXPERIMENTO, "config_experimento.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    with open(os.path.join(CAMINHO_SAIDA_EXPERIMENTO, "modelo.pkl"), "wb") as f:
        pickle.dump(
            {
                "min_global": min_global,
                "max_global": max_global,
                "pca": pca,
                "kmeans": kmeans,
                "k_final": k_final,
                "cluster_ativa": cluster_ativa,
                "clusters_ativos": clusters_ativos,
                "cluster_normal": cluster_normal,
            },
            f,
        )

    print("\n[PIPELINE] Experimento treino/teste finalizado com sucesso.")
    print(f"[PIPELINE] Saídas em: {CAMINHO_SAIDA_EXPERIMENTO}")

    return ModeloTreinado(
        min_global=min_global,
        max_global=max_global,
        pca=pca,
        kmeans=kmeans,
        k_final=k_final,
        cluster_ativa=cluster_ativa,
        clusters_ativos=clusters_ativos,
        cluster_normal=cluster_normal,
    )


if __name__ == "__main__":
    from analise_perfis_nao_supervisionado import executar_analise

    parser = argparse.ArgumentParser(
        description=(
            "Compatibilidade: o fluxo de treino/teste foi substituído pela análise "
            "não supervisionada com referência ATCC pós-clustering."
        )
    )
    parser.add_argument("--k-fixo", type=int, default=None, help="Força um valor de k, ignorando o critério automático.")
    parser.add_argument(
        "--saida",
        default=None,
        help="Pasta de saída. Ex.: dados/analise_perfis_nao_supervisionado_k4",
    )
    args = parser.parse_args()
    print(
        "[AVISO] O fluxo treino/teste foi descontinuado para esta abordagem. "
        "Executando analise_perfis_nao_supervisionado.py."
    )
    executar_analise(k_fixo=args.k_fixo, saida=args.saida)
