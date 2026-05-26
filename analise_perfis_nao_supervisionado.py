from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import unicodedata
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


CAMINHO_IMGS_PROCESSADAS = "dados/img_processadas"
CAMINHO_ROTULOS = "dados/rotulos.csv"
CAMINHO_SAIDA_ANALISE = "dados/analise_perfis_nao_supervisionado"
CAMINHO_SAIDA_FIGURAS = os.path.join(CAMINHO_SAIDA_ANALISE, "figuras")

K_MIN = 2
K_MAX = 10
K_AMOSTRAS_ESCOLHA_K = 10_000
K_CRITERIO_FINAL = "silhouette"  # "silhouette" | "cotovelo" | "media"
KMEANS_N_INIT = 20
KMEANS_RANDOM_STATE = 42
PCA_VARIANCIA = 0.95
RANDOM_STATE = 42
MOSAICO_COLS = 4

_FIG_COUNTER = 0


@dataclass(frozen=True)
class ModeloNaoSupervisionado:
    min_global: np.ndarray
    max_global: np.ndarray
    pca: object
    kmeans: object
    k_final: int
    perfil_medio_atcc: np.ndarray


def configurar_saida(caminho_saida: str) -> None:
    global CAMINHO_SAIDA_ANALISE, CAMINHO_SAIDA_FIGURAS, _FIG_COUNTER
    CAMINHO_SAIDA_ANALISE = caminho_saida
    CAMINHO_SAIDA_FIGURAS = os.path.join(CAMINHO_SAIDA_ANALISE, "figuras")
    _FIG_COUNTER = 0


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


def aplicar_minmax_global(espectros: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    X_total = np.vstack([np.asarray(X, dtype=np.float32) for X in espectros.values()])
    min_global = X_total.min(axis=0)
    max_global = X_total.max(axis=0)
    den = np.where((max_global - min_global) == 0, 1e-8, max_global - min_global)

    espectros_norm = {
        nome: np.clip((np.asarray(X, dtype=np.float32) - min_global) / den, 0.0, 1.0)
        for nome, X in espectros.items()
    }
    return espectros_norm, min_global, max_global


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


def executar_pca_global(X_total: np.ndarray) -> tuple[object, np.ndarray]:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=float(PCA_VARIANCIA), random_state=RANDOM_STATE)
    Z_total = pca.fit_transform(X_total)
    print(f"[PCA] Componentes mantidos: {Z_total.shape[1]}")
    return pca, Z_total


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


def avaliar_k(Z_total: np.ndarray, k_fixo: int | None = None) -> tuple[dict, int]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    Z_sub = _subamostrar(Z_total)
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
    os.makedirs(CAMINHO_SAIDA_ANALISE, exist_ok=True)
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
    caminho = os.path.join(CAMINHO_SAIDA_ANALISE, "escolha_k.csv")
    pd.DataFrame(linhas).to_csv(caminho, index=False)
    print(f"[SAIDA] {caminho}")


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
    axes[0].set_title("Escolha de k - Cotovelo")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ks, silhouettes, "o-", color="darkgreen")
    axes[1].axvline(k_silhouette, color="crimson", linestyle="--", label=f"max silhouette k={k_silhouette}")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette médio")
    axes[1].set_title("Escolha de k - Silhouette")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _salvar_figura(fig, "escolha_k")


def treinar_kmeans(Z_total: np.ndarray, k: int) -> object:
    from sklearn.cluster import KMeans

    km = KMeans(
        n_clusters=int(k),
        init="k-means++",
        n_init=KMEANS_N_INIT,
        random_state=KMEANS_RANDOM_STATE,
        algorithm="lloyd",
    )
    km.fit(Z_total)
    return km


def _labels_para_mapa_2d(labels_roi: np.ndarray, mascara_roi: np.ndarray) -> np.ndarray:
    label_map_2d = np.full(mascara_roi.shape, -1, dtype=np.int64)
    label_map_2d[mascara_roi] = labels_roi.reshape(-1)
    return label_map_2d


def criar_mapas_labels(
    labels: np.ndarray,
    fatias: dict[str, slice],
    mascaras: dict[str, np.ndarray],
    k: int,
) -> dict[str, np.ndarray]:
    mapas: dict[str, np.ndarray] = {}
    for nome in sorted(fatias.keys()):
        mapa = _labels_para_mapa_2d(labels[fatias[nome]], mascaras[nome])
        mapa_plot = mapa.astype(np.float32)
        mapa_plot[mapa < 0] = np.nan
        mapas[f"{nome} (k={k})"] = mapa_plot
    return mapas


def plotar_mosaico_clusters(mapas: dict[str, np.ndarray], k: int) -> None:
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

    fig.suptitle(f"Mosaico de clusters - análise não supervisionada (k={k})", y=0.98)
    plt.tight_layout()
    _salvar_figura(fig, f"mosaico_clusters_k_{k}")


def calcular_perfis_por_imagem(
    labels: np.ndarray,
    fatias: dict[str, slice],
    k: int,
) -> pd.DataFrame:
    linhas: list[dict] = []

    for nome in sorted(fatias.keys()):
        labs = labels[fatias[nome]]
        total = int(labs.shape[0])
        contagens = {cid: int(np.sum(labs == cid)) for cid in range(k)}
        fracoes = {cid: contagens[cid] / max(1, total) for cid in range(k)}
        cluster_dominante = max(contagens, key=contagens.get)

        linha = {
            "imagem": nome,
            "referencia_atcc": nome.startswith("ATCC"),
            "cluster_dominante": cluster_dominante,
            "pureza": fracoes[cluster_dominante],
            "n_pixels": total,
        }
        for cid in range(k):
            linha[f"cluster_{cid}_pixels"] = contagens[cid]
            linha[f"cluster_{cid}_frac"] = fracoes[cid]
        linhas.append(linha)

    return pd.DataFrame(linhas)


def adicionar_similaridade_atcc(df_perfis: pd.DataFrame, k: int) -> tuple[pd.DataFrame, np.ndarray]:
    col_fracs = [f"cluster_{cid}_frac" for cid in range(k)]
    df = df_perfis.copy()
    df_atcc = df[df["referencia_atcc"]]
    if df_atcc.empty:
        raise ValueError("Nenhuma ATCC encontrada para calcular perfil de referência.")

    perfil_medio_atcc = df_atcc[col_fracs].to_numpy(dtype=float).mean(axis=0)
    perfis = df[col_fracs].to_numpy(dtype=float)

    # Distância L1 normalizada: 0 = idêntico ao perfil médio ATCC, 1 = totalmente diferente.
    distancia = 0.5 * np.abs(perfis - perfil_medio_atcc).sum(axis=1)
    similaridade = 1.0 - distancia
    ordem_similaridade = pd.Series(similaridade).rank(ascending=False, method="min").astype(int).to_numpy()

    df["distancia_perfil_atcc"] = distancia
    df["similaridade_perfil_atcc"] = similaridade
    df["ranking_similaridade_atcc"] = ordem_similaridade
    return df.sort_values(["ranking_similaridade_atcc", "imagem"]), perfil_medio_atcc


def salvar_perfis(df_perfis: pd.DataFrame, perfil_medio_atcc: np.ndarray) -> None:
    os.makedirs(CAMINHO_SAIDA_ANALISE, exist_ok=True)
    caminho_perfis = os.path.join(CAMINHO_SAIDA_ANALISE, "perfis_clusters_por_imagem.csv")
    caminho_atcc = os.path.join(CAMINHO_SAIDA_ANALISE, "perfil_medio_atcc.csv")

    df_perfis.to_csv(caminho_perfis, index=False)
    pd.DataFrame(
        [
            {
                "cluster": cid,
                "frac_media_atcc": float(frac),
            }
            for cid, frac in enumerate(perfil_medio_atcc)
        ]
    ).to_csv(caminho_atcc, index=False)

    print(f"[SAIDA] {caminho_perfis}")
    print(f"[SAIDA] {caminho_atcc}")
    print(df_perfis.to_string(index=False))


def plotar_similaridade_atcc(df_perfis: pd.DataFrame) -> None:
    df_plot = df_perfis.sort_values("similaridade_perfil_atcc", ascending=True)
    cores = ["darkgreen" if bool(v) else "steelblue" for v in df_plot["referencia_atcc"].values]

    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(df_plot) + 2)))
    y_pos = np.arange(len(df_plot))
    ax.barh(y_pos, df_plot["similaridade_perfil_atcc"].values, color=cores, alpha=0.75)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["imagem"].tolist(), fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Similaridade com o perfil médio ATCC")
    ax.set_title("Ranking exploratório de similaridade com ATCC")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _salvar_figura(fig, "ranking_similaridade_atcc")


def plotar_perfis_clusters(df_perfis: pd.DataFrame, k: int) -> None:
    col_fracs = [f"cluster_{cid}_frac" for cid in range(k)]
    df_plot = df_perfis.sort_values("ranking_similaridade_atcc")
    bottom = np.zeros(len(df_plot))

    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(df_plot) + 2)))
    y_pos = np.arange(len(df_plot))
    cmap = plt.get_cmap("tab20")

    for cid, col in enumerate(col_fracs):
        vals = df_plot[col].values
        ax.barh(y_pos, vals, left=bottom, color=cmap(cid), alpha=0.8, label=f"Cluster {cid}")
        bottom += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["imagem"].tolist(), fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fração de pixels por cluster")
    ax.set_title("Assinatura de clusters por imagem")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    _salvar_figura(fig, "assinaturas_clusters_por_imagem")


def _normalizar_texto(texto: object) -> str:
    texto_norm = unicodedata.normalize("NFKD", str(texto).strip().lower())
    texto_norm = "".join(ch for ch in texto_norm if not unicodedata.combining(ch))
    return texto_norm.replace(" ", "_")


def _classe_rotulo(rotulo: object) -> str | None:
    rot = _normalizar_texto(rotulo)
    if rot in {"resistente", "resistentes", "ativa", "ativo", "atcc", "1"}:
        return "resistente"
    if rot in {"sensivel", "sensiveis", "normal", "normais", "0"}:
        return "sensivel"
    return None


def ler_rotulos_conhecidos(
    caminho_rotulos: str = CAMINHO_ROTULOS,
    rotulos_extras: dict[str, str] | None = None,
) -> dict[str, str]:
    if not os.path.exists(caminho_rotulos):
        print(f"[VALIDACAO] Arquivo de rótulos não encontrado: {caminho_rotulos}")
        return {}

    df = pd.read_csv(caminho_rotulos)
    if "nome" not in df.columns or "rotulo" not in df.columns:
        raise ValueError(f"{caminho_rotulos} precisa conter as colunas 'nome' e 'rotulo'.")

    rotulos: dict[str, str] = {}
    for _, row in df.iterrows():
        classe = _classe_rotulo(row["rotulo"])
        if classe is not None:
            rotulos[str(row["nome"]).strip()] = classe

    for nome, rotulo in (rotulos_extras or {}).items():
        classe = _classe_rotulo(rotulo)
        if classe is not None:
            rotulos[str(nome).strip()] = classe
    return rotulos


def _rotulos_pixel_a_pixel(
    fatias: dict[str, slice],
    rotulos_conhecidos: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    classes_ord = sorted(set(rotulos_conhecidos.values()))
    mapa_classes = {classe: idx for idx, classe in enumerate(classes_ord)}
    n_total = max(sl.stop for sl in fatias.values())
    y_true = np.full(n_total, -1, dtype=np.int64)
    nomes_rotulos = np.asarray(["desconhecido"] * n_total, dtype=object)

    for nome, sl in fatias.items():
        classe = rotulos_conhecidos.get(nome)
        if classe is None:
            continue
        y_true[sl] = mapa_classes[classe]
        nomes_rotulos[sl] = classe

    return y_true, nomes_rotulos, mapa_classes


def calcular_validacao_externa(
    labels_total: np.ndarray,
    fatias: dict[str, slice],
    k: int,
    caminho_rotulos: str = CAMINHO_ROTULOS,
    rotulos_extras: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    rotulos_conhecidos = ler_rotulos_conhecidos(caminho_rotulos, rotulos_extras=rotulos_extras)
    y_true, nomes_rotulos, mapa_classes = _rotulos_pixel_a_pixel(fatias, rotulos_conhecidos)
    valid = y_true >= 0
    labels_validos = labels_total[valid]
    y_valid = y_true[valid]

    n_pixels_validos = int(np.sum(valid))
    classes_presentes = sorted(np.unique(y_valid).astype(int).tolist()) if n_pixels_validos else []
    n_classes_validas = len(classes_presentes)
    rotulos_presentes = [
        classe
        for classe, idx in sorted(mapa_classes.items(), key=lambda item: item[1])
        if idx in classes_presentes
    ]
    avaliacao_confiavel = n_pixels_validos > 0 and n_classes_validas >= 2

    if n_pixels_validos:
        ari = float(adjusted_rand_score(y_valid, labels_validos))
        nmi = float(normalized_mutual_info_score(y_valid, labels_validos))
    else:
        ari = float("nan")
        nmi = float("nan")

    linhas_clusters: list[dict] = []
    soma_maiorias = 0
    for cid in range(k):
        mask_cluster = valid & (labels_total == cid)
        n_cluster = int(np.sum(mask_cluster))
        linha = {
            "cluster": cid,
            "n_pixels_rotulados": n_cluster,
            "classe_majoritaria": "",
            "pureza_cluster": float("nan"),
        }

        if n_cluster > 0:
            counts = pd.Series(nomes_rotulos[mask_cluster]).value_counts()
            classe_majoritaria = str(counts.index[0])
            maioria = int(counts.iloc[0])
            soma_maiorias += maioria
            linha["classe_majoritaria"] = classe_majoritaria
            linha["pureza_cluster"] = maioria / n_cluster
            for classe in sorted(mapa_classes.keys()):
                linha[f"pixels_{classe}"] = int(counts.get(classe, 0))
                linha[f"frac_{classe}"] = int(counts.get(classe, 0)) / n_cluster
        else:
            for classe in sorted(mapa_classes.keys()):
                linha[f"pixels_{classe}"] = 0
                linha[f"frac_{classe}"] = float("nan")
        linhas_clusters.append(linha)

    pureza_global = soma_maiorias / n_pixels_validos if n_pixels_validos else float("nan")
    aviso = (
        "Métricas calculadas com duas ou mais classes conhecidas."
        if avaliacao_confiavel
        else "Interpretação limitada: há menos de duas classes conhecidas nos rótulos; ARI, NMI e pureza não validam separação entre resistente e sensível."
    )

    df_resumo = pd.DataFrame(
        [
            {
                "ARI": ari,
                "NMI": nmi,
                "pureza_global_clusters": pureza_global,
                "n_pixels_rotulados": n_pixels_validos,
                "n_imagens_rotuladas": len(rotulos_conhecidos),
                "n_classes_validas": n_classes_validas,
                "rotulos_presentes": "|".join(rotulos_presentes),
                "avaliacao_confiavel": avaliacao_confiavel,
                "aviso": aviso,
            }
        ]
    )

    linhas_imagens = []
    for nome, sl in sorted(fatias.items()):
        classe = rotulos_conhecidos.get(nome)
        if classe is None:
            continue
        labs_img = labels_total[sl]
        contagens = {cid: int(np.sum(labs_img == cid)) for cid in range(k)}
        linha = {
            "imagem": nome,
            "rotulo": classe,
            "n_pixels": int(labs_img.shape[0]),
            "cluster_dominante": max(contagens, key=contagens.get),
            "pureza_interna_imagem": max(contagens.values()) / max(1, int(labs_img.shape[0])),
        }
        for cid in range(k):
            linha[f"cluster_{cid}_pixels"] = contagens[cid]
            linha[f"cluster_{cid}_frac"] = contagens[cid] / max(1, int(labs_img.shape[0]))
        linhas_imagens.append(linha)

    return df_resumo, pd.DataFrame(linhas_clusters), pd.DataFrame(linhas_imagens)


def salvar_validacao_externa(
    df_resumo: pd.DataFrame,
    df_clusters: pd.DataFrame,
    df_imagens: pd.DataFrame,
) -> None:
    os.makedirs(CAMINHO_SAIDA_ANALISE, exist_ok=True)
    caminho_resumo = os.path.join(CAMINHO_SAIDA_ANALISE, "metricas_validacao_externa.csv")
    caminho_clusters = os.path.join(CAMINHO_SAIDA_ANALISE, "pureza_clusters.csv")
    caminho_imagens = os.path.join(CAMINHO_SAIDA_ANALISE, "metricas_imagens_rotuladas.csv")

    df_resumo.to_csv(caminho_resumo, index=False)
    df_clusters.to_csv(caminho_clusters, index=False)
    df_imagens.to_csv(caminho_imagens, index=False)

    print(f"[SAIDA] {caminho_resumo}")
    print(f"[SAIDA] {caminho_clusters}")
    print(f"[SAIDA] {caminho_imagens}")
    print(df_resumo.to_string(index=False))


def salvar_modelo_e_config(
    min_global: np.ndarray,
    max_global: np.ndarray,
    pca: object,
    kmeans: object,
    k_final: int,
    resultado_k: dict,
    perfil_medio_atcc: np.ndarray,
    k_fixo: int | None,
    rotulos_validacao_extra: dict[str, str] | None = None,
) -> None:
    os.makedirs(CAMINHO_SAIDA_ANALISE, exist_ok=True)
    np.save(os.path.join(CAMINHO_SAIDA_ANALISE, "min_global.npy"), min_global)
    np.save(os.path.join(CAMINHO_SAIDA_ANALISE, "max_global.npy"), max_global)
    np.save(os.path.join(CAMINHO_SAIDA_ANALISE, "perfil_medio_atcc.npy"), perfil_medio_atcc)

    config = {
        "abordagem": "nao_supervisionada_com_referencia_atcc_pos_hoc",
        "usa_rotulos_no_treino": False,
        "usa_todas_as_imagens_no_agrupamento": True,
        "rotulos_usados_apenas_na_validacao_externa": CAMINHO_ROTULOS,
        "rotulos_validacao_extra": rotulos_validacao_extra or {},
        "metricas_validacao_externa": ["ARI", "NMI", "pureza_global_clusters", "pureza_por_cluster"],
        "k_final": k_final,
        "k_criterio_final": resultado_k["criterio_final"],
        "k_fixo": k_fixo,
        "pca_variancia": PCA_VARIANCIA,
        "random_state": RANDOM_STATE,
        "kmeans_random_state": KMEANS_RANDOM_STATE,
        "observacao": "ATCC e usada somente apos o clustering para calcular perfil medio de referencia.",
    }
    with open(os.path.join(CAMINHO_SAIDA_ANALISE, "config_analise.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    with open(os.path.join(CAMINHO_SAIDA_ANALISE, "modelo.pkl"), "wb") as f:
        pickle.dump(
            {
                "min_global": min_global,
                "max_global": max_global,
                "pca": pca,
                "kmeans": kmeans,
                "k_final": k_final,
                "perfil_medio_atcc": perfil_medio_atcc,
            },
            f,
        )


def executar_analise(
    k_fixo: int | None = None,
    saida: str | None = None,
    rotulos_validacao_extra: dict[str, str] | None = None,
) -> ModeloNaoSupervisionado:
    if saida is not None:
        configurar_saida(saida)

    print("[PIPELINE] Lendo dados processados.")
    espectros, mascaras = ler_espectros_e_mascaras()

    print("\n[NORMALIZACAO] Aplicando Min-Max global em todas as imagens.")
    espectros_norm, min_global, max_global = aplicar_minmax_global(espectros)
    X_total, _, fatias = concatenar_espectros(espectros_norm)
    print(f"[CONCAT] X_total={X_total.shape}")

    print("\n[PCA] Ajustando PCA global sem rotulos.")
    pca, Z_total = executar_pca_global(X_total)

    print("\n[K-SELECAO] Avaliando k sem rotulos.")
    resultado_k, k_final = avaliar_k(Z_total, k_fixo=k_fixo)
    salvar_relatorio_escolha_k(resultado_k)
    plotar_escolha_k(resultado_k)

    print(f"\n[KMEANS] Treinando K-Means global sem rotulos (k={k_final}).")
    kmeans = treinar_kmeans(Z_total, k_final)
    labels_total = kmeans.predict(Z_total)
    np.save(os.path.join(CAMINHO_SAIDA_ANALISE, "labels_total.npy"), labels_total)

    print("\n[VALIDACAO] Calculando ARI, NMI e pureza dos clusters com rotulos conhecidos.")
    df_validacao, df_pureza_clusters, df_imagens_rotuladas = calcular_validacao_externa(
        labels_total,
        fatias,
        k_final,
        rotulos_extras=rotulos_validacao_extra,
    )
    salvar_validacao_externa(df_validacao, df_pureza_clusters, df_imagens_rotuladas)

    print("\n[PERFIS] Calculando assinaturas por imagem e similaridade com ATCC.")
    df_perfis = calcular_perfis_por_imagem(labels_total, fatias, k_final)
    df_perfis, perfil_medio_atcc = adicionar_similaridade_atcc(df_perfis, k_final)
    salvar_perfis(df_perfis, perfil_medio_atcc)

    mapas = criar_mapas_labels(labels_total, fatias, mascaras, k_final)
    plotar_mosaico_clusters(mapas, k_final)
    plotar_similaridade_atcc(df_perfis)
    plotar_perfis_clusters(df_perfis, k_final)

    salvar_modelo_e_config(
        min_global,
        max_global,
        pca,
        kmeans,
        k_final,
        resultado_k,
        perfil_medio_atcc,
        k_fixo,
        rotulos_validacao_extra=rotulos_validacao_extra,
    )

    print("\n[PIPELINE] Análise não supervisionada finalizada com sucesso.")
    print(f"[PIPELINE] Saídas em: {CAMINHO_SAIDA_ANALISE}")

    return ModeloNaoSupervisionado(
        min_global=min_global,
        max_global=max_global,
        pca=pca,
        kmeans=kmeans,
        k_final=k_final,
        perfil_medio_atcc=perfil_medio_atcc,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analise perfis espectrais sem rotulos no treino.")
    parser.add_argument("--k-fixo", type=int, default=None, help="Força um valor de k, ignorando o critério automático.")
    parser.add_argument(
        "--saida",
        default=None,
        help="Pasta de saída. Ex.: dados/analise_perfis_nao_supervisionado_k4",
    )
    args = parser.parse_args()
    executar_analise(k_fixo=args.k_fixo, saida=args.saida)
