from __future__ import annotations

import glob
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

# --- Caminhos e Parâmetros Alinhados ao Processamento
CAMINHO_IMGS_PROCESSADAS = "dados/img_processadas"
CAMINHO_IMGS_ORIGINAL = "dados/img_original"
CAMINHO_ROTULOS = "dados/rotulos.csv"
CAMINHO_SAIDA_FIGURAS = "dados/figuras"

# Configurações de Visualização
MOSAICO_COLS = 4
BANDA_VISUALIZACAO = 1  
MOSAICO_CMAP = "gray"

# Parâmetros de Projeção e Treinamento Global
KMEANS_N_INIT = 20
KMEANS_RANDOM_STATE = 42
K_MIN = 2
K_MAX = 10
K_AMOSTRAS_ESCOLHA_K = 10_000
K_CRITERIO_FINAL = "silhouette"  # "silhouette" | "cotovelo" | "media"
PCA_MAX_AMOSTRAS = 10_000
PCA_USAR_PADRONIZACAO = True
PCA_VARIANCIA_PARA_TREINO_GLOBAL = 0.95
TSNE_MAX_AMOSTRAS = 5_000
TSNE_PERPLEXITY = 30.0
TSNE_MAX_ITER = 1000  
PROJECAO_RANDOM_STATE = 42
ROTULO_FALLBACK = "sensivel"

SALVAR_FIGURAS = True
IMPRIMIR_FIGURAS = False

_FIG_COUNTER = 0


def _sanitize_filename(texto: str) -> str:
    texto = texto.strip().replace(" ", "_")
    allowed = [ch for ch in texto if ch.isalnum() or ch in {"-", "_"}]
    out = "".join(allowed)
    return out[:120] if out else "fig"


def _gerar_id_fig(prefixo: str) -> str:
    global _FIG_COUNTER
    _FIG_COUNTER += 1
    return f"{_FIG_COUNTER:04d}_{_sanitize_filename(prefixo)}"


def _salvar_ou_mostrar(fig: plt.Figure, nome_base: str) -> None:
    if not SALVAR_FIGURAS and not IMPRIMIR_FIGURAS:
        plt.close(fig)
        return
    os.makedirs(CAMINHO_SAIDA_FIGURAS, exist_ok=True)
    nome_id = _gerar_id_fig(nome_base)
    if SALVAR_FIGURAS:
        fig.savefig(os.path.join(CAMINHO_SAIDA_FIGURAS, f"{nome_id}.png"), dpi=200, bbox_inches="tight")
    if IMPRIMIR_FIGURAS:
        plt.show()
    plt.close(fig)


def _criar_cmap_discreta(k: int, base_cmap: str = "tab20") -> ListedColormap:
    cmap_base = plt.get_cmap(base_cmap)
    cores = cmap_base(np.linspace(0.0, 1.0, k))
    return ListedColormap(cores, name=f"{base_cmap}_{k}")


def printar_mosaico_2d(imagens_2d: dict[str, np.ndarray], cols: int = MOSAICO_COLS, cmap: str = MOSAICO_CMAP, titulo: str | None = None, vmin: float | None = 0.0, vmax: float | None = 1.0) -> None:
    n = len(imagens_2d)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.atleast_1d(axes).flatten()
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")

    for i, (chave, img) in enumerate(imagens_2d.items()):
        ax = axes[i]
        img_plot = np.ma.masked_invalid(np.asarray(img))
        ax.imshow(img_plot, cmap=cmap_obj, vmin=vmin, vmax=vmax)
        ax.set_title(str(chave)[:24], fontsize=8)
        ax.axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if titulo:
        fig.suptitle(titulo, y=0.98)
    plt.tight_layout()
    _salvar_ou_mostrar(fig, nome_base=titulo or "mosaico_2d")


def _printar_mosaico_clusters_discreto(label_maps_2d: dict[str, np.ndarray], k_max: int, cols: int, titulo: str | None = None) -> None:
    cmap_discreta = _criar_cmap_discreta(k_max, base_cmap="gist_ncar")
    printar_mosaico_2d(label_maps_2d, cols=cols, cmap=cmap_discreta, titulo=titulo, vmin=-0.5, vmax=k_max - 0.5)


def ler_espectros_e_mascaras(pasta_processadas: str = CAMINHO_IMGS_PROCESSADAS, verbose: bool = True) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    csv_paths = sorted(glob.glob(os.path.join(pasta_processadas, "**", "espectros.csv"), recursive=True))
    if not csv_paths:
        raise FileNotFoundError(f"Nenhum espectros.csv encontrado em '{pasta_processadas}'.")

    espectros, mascaras = {}, {}
    for csv_path in csv_paths:
        pasta_img = os.path.dirname(csv_path)
        nome = os.path.basename(pasta_img)
        mask_path = os.path.join(pasta_img, "mascara_roi.npy")

        mascara = np.load(mask_path).astype(bool)
        X = pd.read_csv(csv_path).values.astype(np.float32)

        if X.shape[0] != int(np.count_nonzero(mascara)):
            raise ValueError(f"{nome}: Inconsistência de dimensões entre o CSV e a máscara.")

        espectros[nome] = X
        mascaras[nome] = mascara
        if verbose:
            print(f"[PIPELINE] {nome}: espectros carregados {X.shape}, mascara {mascara.shape}.")
    return espectros, mascaras


def aplicar_minmax_global(espectros: dict[str, np.ndarray], verbose: bool = True) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    blocos = [np.asarray(X, dtype=np.float32) for X in espectros.values()]
    X_total = np.vstack(blocos)
    min_global = X_total.min(axis=0)
    max_global = X_total.max(axis=0)
    den = np.where((max_global - min_global) == 0, 1e-8, max_global - min_global)

    espectros_norm = {}
    for nome, X in espectros.items():
        X_norm = np.clip((np.asarray(X, dtype=np.float32) - min_global) / den, 0.0, 1.0)
        espectros_norm[nome] = X_norm
        if verbose:
            print(f"[NORMALIZACAO] Aplicando Min-Max global em {nome}.")
    return espectros_norm, (min_global, max_global)


def espectros_2d_para_cubo_3d(espectros_2d: np.ndarray, mascara_roi: np.ndarray) -> np.ndarray:
    h, w = mascara_roi.shape
    cubo = np.full((h, w, espectros_2d.shape[1]), np.nan, dtype=np.float32)
    cubo[mascara_roi] = espectros_2d
    return cubo


def cubos_para_imagens_recortadas_2d(cubos: dict[str, np.ndarray], banda_vis: int = BANDA_VISUALIZACAO) -> dict[str, np.ndarray]:
    imgs_2d = {}
    for nome, cubo in cubos.items():
        banda_idx = int(np.clip(banda_vis - 1, 0, cubo.shape[-1] - 1))
        imgs_2d[nome] = cubo[:, :, banda_idx]
    return imgs_2d


def ler_rotulos_por_imagem(caminho_csv: str = CAMINHO_ROTULOS) -> dict[str, str]:
    df = pd.read_csv(caminho_csv)
    return {str(row["nome"]).strip(): str(row["rotulo"]).strip() for _, row in df.iterrows()}


def concatenar_espectros_e_rotulos(espectros_norm: dict[str, np.ndarray], rotulos_por_imagem: dict[str, str], rotulo_fallback: str = ROTULO_FALLBACK) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, slice]]:
    nomes_ord = sorted(espectros_norm.keys())
    blocos_x, blocos_nome, blocos_rot, fatias = [], [], [], {}
    offset = 0

    for nome in nomes_ord:
        Xn = np.asarray(espectros_norm[nome], dtype=np.float32)
        n = int(Xn.shape[0])
        blocos_x.append(Xn)
        blocos_nome.append(np.asarray([nome] * n, dtype=object))
        rot = rotulos_por_imagem.get(nome, rotulo_fallback)
        blocos_rot.append(np.asarray([rot] * n, dtype=object))
        fatias[nome] = slice(offset, offset + n)
        offset += n

    return np.vstack(blocos_x), np.concatenate(blocos_nome, axis=0), np.concatenate(blocos_rot, axis=0), fatias


def executar_pca_para_treino_global(X_total: np.ndarray, variancia: float = PCA_VARIANCIA_PARA_TREINO_GLOBAL, random_state: int = PROJECAO_RANDOM_STATE) -> tuple[object, np.ndarray]:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=float(variancia), random_state=int(random_state))
    Z_total = pca.fit_transform(X_total)
    print(f"[PIPELINE] PCA global treinado: componentes={Z_total.shape[1]}")
    return pca, Z_total


def treinar_kmeans_global(Z_total: np.ndarray, k: int, n_init: int = KMEANS_N_INIT, random_state: int = KMEANS_RANDOM_STATE) -> object:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=int(k), init="k-means++", n_init=int(n_init), random_state=int(random_state), algorithm="lloyd")
    km.fit(Z_total)
    return km


def _intervalo_k(n_amostras: int, k_min: int = K_MIN, k_max: int = K_MAX) -> list[int]:
    """Define intervalo de k testável conforme tamanho da amostra."""
    k_superior = min(k_max, max(k_min, n_amostras // 500))
    return list(range(k_min, k_superior + 1))


def _subamostrar_para_escolha_k(Z_total: np.ndarray, max_amostras: int = K_AMOSTRAS_ESCOLHA_K, random_state: int = PROJECAO_RANDOM_STATE) -> np.ndarray:
    n = Z_total.shape[0]
    if n <= max_amostras:
        return Z_total
    idx = np.sort(np.random.default_rng(random_state).choice(n, size=max_amostras, replace=False))
    return Z_total[idx]


def _detectar_k_cotovelo(ks: list[int], inertias: list[float]) -> int:
    """Cotovelo: maior distância perpendicular à reta (k_min, inertia_min) → (k_max, inertia_max)."""
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


def avaliar_k_cotovelo_silhouette(
    Z_total: np.ndarray,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    n_init: int = KMEANS_N_INIT,
    random_state: int = KMEANS_RANDOM_STATE,
    verbose: bool = True,
) -> dict:
    """
    Varre valores de k, calcula inércia (cotovelo) e silhouette médio.
    Retorna métricas, gráficos e k sugeridos por cada critério.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    Z_sub = _subamostrar_para_escolha_k(Z_total, random_state=random_state)
    ks = _intervalo_k(Z_sub.shape[0], k_min=k_min, k_max=k_max)

    inertias: list[float] = []
    silhouettes: list[float] = []

    for k in ks:
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=n_init,
            random_state=random_state,
            algorithm="lloyd",
        )
        labels = km.fit_predict(Z_sub)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(Z_sub, labels, metric="euclidean")))

        if verbose:
            print(f"[K-SELECAO] k={k}: inercia={inertias[-1]:.2f}, silhouette={silhouettes[-1]:.4f}")

    k_cotovelo = _detectar_k_cotovelo(ks, inertias)
    k_silhouette = int(ks[int(np.argmax(silhouettes))])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ks, inertias, "o-", color="steelblue")
    axes[0].axvline(k_cotovelo, color="crimson", linestyle="--", label=f"cotovelo k={k_cotovelo}")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inércia (WCSS)")
    axes[0].set_title("Método do Cotovelo")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ks, silhouettes, "o-", color="darkgreen")
    axes[1].axvline(k_silhouette, color="crimson", linestyle="--", label=f"max silhouette k={k_silhouette}")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette médio")
    axes[1].set_title("Silhouette Score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _salvar_ou_mostrar(fig, nome_base="cotovelo_silhouette_k")

    resultado = {
        "ks": ks,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "k_cotovelo": k_cotovelo,
        "k_silhouette": k_silhouette,
        "n_amostras_avaliacao": int(Z_sub.shape[0]),
    }
    return resultado


def escolher_k_final(resultado_k: dict, criterio: str = K_CRITERIO_FINAL) -> int:
    """Define k final a partir dos critérios cotovelo e silhouette."""
    k_cotovelo = int(resultado_k["k_cotovelo"])
    k_silhouette = int(resultado_k["k_silhouette"])

    if criterio == "cotovelo":
        k_final = k_cotovelo
    elif criterio == "media":
        k_final = int(round((k_cotovelo + k_silhouette) / 2))
        k_final = max(K_MIN, min(k_final, max(resultado_k["ks"])))
    else:
        k_final = k_silhouette

    print(
        f"[K-SELECAO] k_cotovelo={k_cotovelo}, k_silhouette={k_silhouette}, "
        f"k_final={k_final} (criterio='{criterio}')"
    )
    return k_final


def salvar_relatorio_escolha_k(resultado_k: dict, k_final: int, caminho: str = "dados/escolha_k.csv") -> None:
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    linhas = []
    for k, inertia, sil in zip(resultado_k["ks"], resultado_k["inertias"], resultado_k["silhouettes"]):
        linhas.append(
            {
                "k": k,
                "inercia": inertia,
                "silhouette": sil,
                "k_cotovelo": resultado_k["k_cotovelo"],
                "k_silhouette": resultado_k["k_silhouette"],
                "k_final": k_final,
                "criterio_final": K_CRITERIO_FINAL,
                "n_amostras_avaliacao": resultado_k["n_amostras_avaliacao"],
            }
        )
    pd.DataFrame(linhas).to_csv(caminho, index=False)


def _labels_para_mapa_2d(labels_roi: np.ndarray, mascara_roi: np.ndarray) -> np.ndarray:
    label_map_2d = np.full(mascara_roi.shape, -1, dtype=np.int64)
    label_map_2d[mascara_roi] = labels_roi.reshape(-1)
    return label_map_2d


def calcular_importancia_bandas(centroids: np.ndarray, nome: str) -> np.ndarray:
    variancia = np.var(centroids, axis=0)
    fig = plt.figure(figsize=(12, 4))
    plt.bar(np.arange(variancia.shape[0]), variancia, color="steelblue", alpha=0.8)
    plt.title(f"Importância das bandas — {nome}")
    _salvar_ou_mostrar(fig, nome_base=f"importancia_bandas_{nome}")
    return variancia


def validar_cluster_resistencia(labels_por_imagem: dict[str, np.ndarray], resistentes: list[str], k: int) -> int:
    print(f"\n[AVALIACAO] Validacao com imagens resistentes conhecidas (k={k}).")
    soma = {cid: 0 for cid in range(k)}

    for nome in resistentes:
        if nome not in labels_por_imagem:
            continue
        labs = labels_por_imagem[nome]
        tot = labs.shape[0]
        contagens = {cid: int(np.sum(labs == cid)) for cid in range(k)}
        for cid, count in contagens.items():
            soma[cid] += count
        dist_str = ", ".join(f"cluster{cid}={contagens[cid]/tot:.3f}" for cid in range(k))
        print(f"[AVALIACAO] {nome}: n={tot}, {dist_str}, pureza={max(contagens.values())/tot:.3f}")

    cluster_resistencia = max(soma, key=soma.get)
    print(f"[AVALIACAO] Decisao agregada: cluster_resistencia={cluster_resistencia}")
    return cluster_resistencia


def plotar_projecao_2d_cluster_vs_rotulo(Z: np.ndarray, labels_kmeans: np.ndarray, rotulos_pixels: np.ndarray, titulo_prefixo: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    k_ids = np.unique(labels_kmeans)
    k_max = int(k_ids.max()) + 1 if k_ids.size else 1
    cmap_k = _criar_cmap_discreta(max(2, k_max), base_cmap="gist_ncar")
    
    sc0 = axes[0].scatter(Z[:, 0], Z[:, 1], c=labels_kmeans, cmap=cmap_k, vmin=-0.5, vmax=k_max-0.5, s=4, alpha=0.65, linewidths=0)
    axes[0].set_title(f"{titulo_prefixo} — por cluster K-Means")
    plt.colorbar(sc0, ax=axes[0], label="cluster")

    rotulos_unicos = sorted(np.unique(rotulos_pixels))
    cores = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(rotulos_unicos), 2)))
    for i, rlab in enumerate(rotulos_unicos):
        mask = rotulos_pixels == rlab
        axes[1].scatter(Z[mask, 0], Z[mask, 1], c=[cores[i % len(cores)]], s=4, alpha=0.65, linewidths=0, label=str(rlab))
    axes[1].set_title(f"{titulo_prefixo} — por rótulo real")
    axes[1].legend(markerscale=3, fontsize=8)
    plt.tight_layout()
    _salvar_ou_mostrar(fig, nome_base=titulo_prefixo)


def executar_pca_2d_e_plotar(X_total: np.ndarray, labels_kmeans_total: np.ndarray, rotulos_pixels: np.ndarray, max_amostras: int = PCA_MAX_AMOSTRAS, random_state: int = PROJECAO_RANDOM_STATE) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    idx = np.sort(np.random.default_rng(random_state).choice(X_total.shape[0], size=min(X_total.shape[0], max_amostras), replace=False))
    X_sub = X_total[idx]
    if PCA_USAR_PADRONIZACAO:
        X_sub = StandardScaler().fit_transform(X_sub)
    Z = PCA(n_components=2, random_state=random_state).fit_transform(X_sub)
    plotar_projecao_2d_cluster_vs_rotulo(Z, labels_kmeans_total[idx], rotulos_pixels[idx], "PCA 2D")
    return Z


def executar_tsne_2d_e_plotar(X_total: np.ndarray, labels_kmeans_total: np.ndarray, rotulos_pixels: np.ndarray, max_amostras: int = TSNE_MAX_AMOSTRAS, perplexity: float = TSNE_PERPLEXITY, max_iter: int = TSNE_MAX_ITER, random_state: int = PROJECAO_RANDOM_STATE) -> np.ndarray:
    from sklearn.manifold import TSNE
    idx = np.sort(np.random.default_rng(random_state).choice(X_total.shape[0], size=min(X_total.shape[0], max_amostras), replace=False))
    X_sub = X_total[idx]
    perp = min(perplexity, float(X_sub.shape[0] - 1))
    Z = TSNE(n_components=2, perplexity=perp, max_iter=max_iter, random_state=random_state, init="pca").fit_transform(X_sub)
    plotar_projecao_2d_cluster_vs_rotulo(Z, labels_kmeans_total[idx], rotulos_pixels[idx], "t-SNE 2D")
    return Z


def calcular_metricas_avaliacao(labels_kmeans: np.ndarray, rotulos_pixels: np.ndarray, nomes_pixels: np.ndarray, verbose: bool = True) -> pd.DataFrame:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    def _norm(s): return str(s).strip().lower().replace(" ", "_").replace("ñ", "n")
    rot_str = np.array([_norm(r) for r in rotulos_pixels], dtype=object)
    nomes_str = np.array([str(n).strip() for n in nomes_pixels], dtype=object)

    y_true = np.full(rot_str.shape[0], -1, dtype=np.int64)
    y_true[(rot_str == "resistente") | (rot_str == "1")] = 1
    y_true[(rot_str == "sensivel") | (rot_str == "sensiveis") | (rot_str == "0")] = 0

    valid = y_true >= 0
    linhas = []
    
    # Global
    linhas.append({"imagem": "GLOBAL", "ARI": adjusted_rand_score(y_true[valid], labels_kmeans[valid]), "NMI": normalized_mutual_info_score(y_true[valid], labels_kmeans[valid]), "Pureza": 1.0, "n_pixels": int(np.sum(valid)), "rotulo_imagem": "misto"})
    
    for nome_img in sorted(np.unique(nomes_str)):
        mask = (nomes_str == nome_img) & valid
        if np.sum(mask) == 0:
            continue
        labs_img = labels_kmeans[mask]
        y_img = y_true[mask]
        pureza = float(np.max(np.bincount(labs_img))) / labs_img.shape[0] if labs_img.size else 0.0
        linhas.append(
            {
                "imagem": str(nome_img),
                "ARI": adjusted_rand_score(y_img, labs_img),
                "NMI": normalized_mutual_info_score(y_img, labs_img),
                "Pureza": pureza,
                "n_pixels": int(np.sum(mask)),
                "rotulo_imagem": "resistente" if y_img[0] == 1 else "sensiveis",
            }
        )

    df = pd.DataFrame(linhas)
    os.makedirs("dados", exist_ok=True)
    df.to_csv(os.path.join("dados", "metricas_avaliacao.csv"), index=False)
    if verbose: print(df.to_string(index=False))
    return df


def plotar_metricas_por_imagem(df_metricas: pd.DataFrame) -> None:
    df_imgs = df_metricas[df_metricas["imagem"] != "GLOBAL"]
    imagens = df_imgs["imagem"].tolist()
    y_pos = np.arange(len(imagens))
    
    fig, axes = plt.subplots(1, 3, figsize=(16, max(4, 0.35 * len(imagens) + 2)), sharey=True)
    for ax, col in zip(axes, ["ARI", "NMI", "Pureza"]):
        ax.barh(y_pos, df_imgs[col].values, color="blue", alpha=0.6)
        ax.set_title(col)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(imagens, fontsize=8)
        ax.invert_yaxis()
    plt.tight_layout()
    _salvar_ou_mostrar(fig, nome_base="metricas_avaliacao_por_imagem")


def plotar_centroides_por_imagem(pasta_processadas: str, nomes_imagens: list[str], rotulos_por_imagem: dict[str, str]) -> None:
    n = len(nomes_imagens)
    rows = (n + MOSAICO_COLS - 1) // MOSAICO_COLS
    fig, axes = plt.subplots(rows, MOSAICO_COLS, figsize=(MOSAICO_COLS * 5, rows * 4))
    axes = np.atleast_1d(axes).flatten()

    for i, nome in enumerate(nomes_imagens):
        centroids = np.load(os.path.join(pasta_processadas, nome, "kmeans_centroids.npy"))
        ax = axes[i]
        for c in range(centroids.shape[0]):
            ax.plot(centroids[c], linewidth=1, label=f"Cluster {c}")
        ax.set_title(f"{nome[:10]} - {rotulos_por_imagem.get(nome, ROTULO_FALLBACK)}", fontsize=8)
        ax.legend(fontsize=7)
    
    for j in range(i + 1, len(axes)): axes[j].axis("off")
    plt.tight_layout()
    _salvar_ou_mostrar(fig, nome_base="centroides_por_imagem")


if __name__ == "__main__":
    print("[PIPELINE] Iniciando leitura de espectros.")
    espectros, mascaras_roi = ler_espectros_e_mascaras()

    print("\n[NORMALIZACAO] Aplicando Min-Max GLOBAL por banda.")
    espectros_norm, _ = aplicar_minmax_global(espectros)

    print("\n[AVALIACAO] Concatenando espectros e rotulos.")
    rotulos_map = ler_rotulos_por_imagem(CAMINHO_ROTULOS)
    X_total, nomes_pixels, rotulos_pixels, fatias_por_imagem = concatenar_espectros_e_rotulos(espectros_norm, rotulos_map)

    print("\n[PIPELINE] Executando PCA global.")
    _, Z_total = executar_pca_para_treino_global(X_total)

    print("\n[K-SELECAO] Avaliando k pelo metodo do cotovelo e silhouette score.")
    resultado_k = avaliar_k_cotovelo_silhouette(Z_total)
    k_final = escolher_k_final(resultado_k)
    salvar_relatorio_escolha_k(resultado_k, k_final)

    print(f"\n[KMEANS] Treinando K-Means GLOBAL (k={k_final}) no espaco PCA.")
    kmeans_global = treinar_kmeans_global(Z_total, k=k_final)
    labels_kmeans_total = kmeans_global.predict(Z_total)

    # ==========================
    # Propagacao e reconstrucao espacial
    # ==========================
    label_maps_para_mosaico = {}
    labels_por_imagem = {}

    for nome in sorted(espectros_norm.keys()):
        sl = fatias_por_imagem[nome]
        Xn = espectros_norm[nome]
        labs = labels_kmeans_total[sl]
        labels_por_imagem[nome] = labs

        label_map_2d = _labels_para_mapa_2d(labs, mascaras_roi[nome])
        label_map_plot = label_map_2d.astype(np.float32)
        label_map_plot[label_map_2d < 0] = np.nan
        label_maps_para_mosaico[f"{nome} (k={k_final})"] = label_map_plot

        pasta_saida_img = os.path.join(CAMINHO_IMGS_PROCESSADAS, nome)
        np.save(os.path.join(pasta_saida_img, "kmeans_labels.npy"), labs)

        # Centroides em espaço ORIGINAL de bandas
        centroids_band = []
        for cid in range(k_final):
            mask_c = labs == cid
            centroids_band.append(Xn[mask_c].mean(axis=0) if np.any(mask_c) else np.zeros(Xn.shape[1]))
        centroids = np.vstack(centroids_band)
        np.save(os.path.join(pasta_saida_img, "kmeans_centroids.npy"), centroids)
        calcular_importancia_bandas(centroids, nome)

    if label_maps_para_mosaico:
        _printar_mosaico_clusters_discreto(
            label_maps_para_mosaico,
            k_max=k_final,
            cols=MOSAICO_COLS,
            titulo=f"Mosaico de clusters — K-Means GLOBAL (k={k_final})",
        )

    resistentes_conhecidas = ["ATCC13_240506-161053", "ATCC16_240506-161158", "ATCC27_240506-161129"]
    validar_cluster_resistencia(labels_por_imagem, resistentes_conhecidas, k=k_final)

    executar_pca_2d_e_plotar(X_total, labels_kmeans_total, rotulos_pixels)
    executar_tsne_2d_e_plotar(X_total, labels_kmeans_total, rotulos_pixels)
    df_metricas = calcular_metricas_avaliacao(labels_kmeans_total, rotulos_pixels, nomes_pixels)
    plotar_metricas_por_imagem(df_metricas)
    plotar_centroides_por_imagem(CAMINHO_IMGS_PROCESSADAS, sorted(espectros_norm.keys()), rotulos_map)
    print("\n[PIPELINE] Script finalizado com sucesso!")
