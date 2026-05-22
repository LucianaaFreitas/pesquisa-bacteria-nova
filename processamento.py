import os
import glob
import csv
import matplotlib.pyplot as plt
import numpy as np
import spectral.io.envi as envi
import cv2

# --- Configurações de Caminhos
CAMINHO_IMGS_ORIGINAL = "dados/img_original"
CAMINHO_IMGS_PROCESSADAS = "dados/img_processadas"

# --- Parâmetros CHT (Ajustado shrink_radius_px para 20 conforme sugerido)
CHT_PARAMS = dict(
    bandas_rgb=(180, 120, 70),
    dp=1.4,
    min_dist=59,
    param1=170,
    param2=30,
    shrink_radius_px=20,  # Reduzido o raio para focar apenas no centro puro da colônia
    min_radius=None,
    max_radius=None,
)


def calibrate_reflectance(image_main, image_dark, image_white):
    """
    Aplica calibração de reflectância usando DARKREF e WHITEREF.
    Usa média por banda para resolver problemas de dimensões.
    """
    if image_dark is None or image_white is None:
        raise ValueError("image_dark e image_white são obrigatórios para calibrar.")

    try:
        image_main = image_main[:, :, :]
    except Exception:
        image_main = np.asarray(image_main)

    try:
        image_dark = image_dark[:, :, :]
    except Exception:
        image_dark = np.asarray(image_dark)

    try:
        image_white = image_white[:, :, :]
    except Exception:
        image_white = np.asarray(image_white)

    mean_dark = np.mean(image_dark, axis=(0, 1))  # (n_bandas,)
    mean_white = np.mean(image_white, axis=(0, 1))  # (n_bandas,)

    denominador = mean_white - mean_dark
    denominador = np.where(denominador == 0, 1e-8, denominador)

    array_calibrated = (image_main - mean_dark) / denominador
    array_calibrated = np.clip(array_calibrated, 0, 1)
    return array_calibrated


def ler_imagens_hyperspectrais_e_processadas(pasta_base: str) -> dict[str, np.ndarray]:
    """
    Para cada pasta de captura (.../<imagem>/capture):
    - lê a imagem principal (<base>.hdr)
    - lê DARKREF_<base>.hdr e WHITEREF_<base>.hdr
    - calibra a imagem principal chamando calibrate_reflectance
    """
    capture_dirs = sorted(glob.glob(os.path.join(pasta_base, "**", "capture"), recursive=True))
    imagens_calibradas: dict[str, np.ndarray] = {}

    for capture_dir in capture_dirs:
        hdr_paths = sorted(glob.glob(os.path.join(capture_dir, "*.hdr")))

        main_hdrs = [
            p
            for p in hdr_paths
            if not os.path.basename(p).upper().startswith("DARKREF_")
            and not os.path.basename(p).upper().startswith("WHITEREF_")
        ]

        for main_hdr_path in main_hdrs:
            base = os.path.splitext(os.path.basename(main_hdr_path))[0]
            dark_hdr_path = os.path.join(capture_dir, f"DARKREF_{base}.hdr")
            white_hdr_path = os.path.join(capture_dir, f"WHITEREF_{base}.hdr")

            main_raw_path = os.path.splitext(main_hdr_path)[0] + ".raw"
            dark_raw_path = os.path.splitext(dark_hdr_path)[0] + ".raw"
            white_raw_path = os.path.splitext(white_hdr_path)[0] + ".raw"

            img_main = envi.open(main_hdr_path, image=main_raw_path)
            img_dark = envi.open(dark_hdr_path, image=dark_raw_path)
            img_white = envi.open(white_hdr_path, image=white_raw_path)

            imagens_calibradas[os.path.basename(main_hdr_path)] = calibrate_reflectance(
                image_main=img_main,
                image_dark=img_dark,
                image_white=img_white,
            )

    if not imagens_calibradas:
        raise ValueError(f"Nenhuma imagem principal encontrada em '{pasta_base}'.")

    return imagens_calibradas


def remover_bandas_ruidosas_refletancia(
    imagens_calibradas: dict[str, np.ndarray],
    n_primeiras: int = 15,
    n_ultimas: int = 15,
    mostrar_shapes: bool = True,
) -> dict[str, np.ndarray]:
    """
    Remove bandas ruidosas após a calibração.
    """
    imagens_limpa: dict[str, np.ndarray] = {}

    for nome, img in imagens_calibradas.items():
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        if img.ndim < 3:
            raise ValueError(f"Imagem '{nome}' esperada com eixo de bandas no último dimension, recebeu shape {img.shape}.")

        n_bands = img.shape[-1]
        corte_inicio = n_primeiras
        corte_fim = n_bands - n_ultimas

        if corte_fim <= corte_inicio:
            raise ValueError(
                f"Imagem '{nome}' não tem bandas suficientes para remover {n_primeiras} iniciais e {n_ultimas} finais (n_bands={n_bands})."
            )

        img_limpa = img[..., corte_inicio:corte_fim]
        imagens_limpa[nome] = img_limpa

        if mostrar_shapes:
            print(
                f"{nome}: shape antes={img.shape} -> depois={img_limpa.shape} "
                f"(cortando {n_primeiras} primeiras e {n_ultimas} ultimas bandas)"
            )

    return imagens_limpa


def _normalizar_0_255(img_2d: np.ndarray) -> np.ndarray:
    """Normaliza imagem 2D para uint8 [0, 255]."""
    img_2d = np.asarray(img_2d, dtype=np.float32)
    vmin = float(np.min(img_2d))
    vmax = float(np.max(img_2d))
    den = vmax - vmin
    if den <= 1e-8:
        return np.zeros_like(img_2d, dtype=np.uint8)
    return (255.0 * (img_2d - vmin) / den).astype(np.uint8)


def gerar_rgb_sintetico(img_hsi: np.ndarray, bandas_rgb: tuple[int, int, int] = (180, 120, 60)) -> np.ndarray:
    """Gera RGB sintético usando bandas específicas (1-based)."""
    img_hsi = np.asarray(img_hsi)
    if img_hsi.ndim < 3:
        raise ValueError(f"Esperado cubo HSI 3D, recebido shape={img_hsi.shape}.")

    n_bands = img_hsi.shape[-1]
    r_idx = int(np.clip(bandas_rgb[0] - 1, 0, n_bands - 1))
    g_idx = int(np.clip(bandas_rgb[1] - 1, 0, n_bands - 1))
    b_idx = int(np.clip(bandas_rgb[2] - 1, 0, n_bands - 1))

    rgb = np.stack([img_hsi[:, :, r_idx], img_hsi[:, :, g_idx], img_hsi[:, :, b_idx]], axis=2)
    return _normalizar_0_255(rgb).astype(np.uint8)


def detectar_circulo_e_mascara_roi(
    rgb_uint8: np.ndarray,
    dp: float = 1.2,
    min_dist: int = 40,
    param1: float = 120,
    param2: float = 22,
    min_radius: int | None = None,
    max_radius: int | None = None,
    shrink_radius_px: int = 0,
) -> tuple[np.ndarray, tuple[int, int, int] | None, np.ndarray]:
    """Detecta círculo com HoughCircles e cria máscara booleana da ROI encolhida."""
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"Esperado RGB uint8 com shape (H, W, 3), recebido {rgb_uint8.shape}.")

    h, w = rgb_uint8.shape[:2]
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    gray_blur = cv2.GaussianBlur(gray_eq, (7, 7), 1.5)

    min_radius_i = min_radius if min_radius is not None else int(min(h, w) * 0.20)
    max_radius_i = max_radius if max_radius is not None else int(min(h, w) * 0.48)
    min_radius_i = max(2, min_radius_i)
    max_radius_i = max(min_radius_i + 1, max_radius_i)

    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
        param1=param1, param2=param2, minRadius=min_radius_i, maxRadius=max_radius_i
    )

    circulo_central: tuple[int, int, int] | None = None
    if circles is not None and circles.shape[1] > 0:
        candidatos = np.round(circles[0]).astype(int)
        cx_img = w / 2.0
        cy_img = h / 2.0
        candidatos = sorted(candidatos, key=lambda c: ((c[0] - cx_img) ** 2 + (c[1] - cy_img) ** 2, -c[2]))
        x, y, r = candidatos[0]
        circulo_central = (int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1)), int(max(1, r)))

    yy, xx = np.ogrid[:h, :w]
    if circulo_central is None:
        mascara_roi = np.ones((h, w), dtype=bool)
    else:
        x, y, r = circulo_central
        r_roi = int(r) - int(shrink_radius_px)
        if r_roi <= 0:
            r_roi = int(r)
        mascara_roi = (xx - x) ** 2 + (yy - y) ** 2 <= r_roi * r_roi

    return gray_blur, circulo_central, mascara_roi


def aplicar_cht_e_preparar_visualizacoes(
    imagens: dict[str, np.ndarray],
    bandas_rgb: tuple[int, int, int] = (180, 120, 60),
    dp: float = 1.2,
    min_dist: int = 40,
    param1: float = 120,
    param2: float = 22,
    min_radius: int | None = None,
    max_radius: int | None = None,
    shrink_radius_px: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Executa pipeline CHT e prepara imagens demarcadas e recortadas."""
    imagens_demarcadas: dict[str, np.ndarray] = {}
    imagens_recortadas: dict[str, np.ndarray] = {}

    for nome, img_hsi in imagens.items():
        rgb = gerar_rgb_sintetico(img_hsi, bandas_rgb=bandas_rgb)
        gray_pre, circulo, mascara_roi = detectar_circulo_e_mascara_roi(
            rgb_uint8=rgb, dp=dp, min_dist=min_dist, param1=param1, param2=param2,
            min_radius=min_radius, max_radius=max_radius, shrink_radius_px=shrink_radius_px
        )

        demarcada = gray_pre.copy()
        if circulo is not None:
            x, y, r = circulo
            cv2.circle(demarcada, (x, y), r, 255, thickness=2)

        recortada = gray_pre.astype(np.float32) / 255.0
        recortada = np.where(mascara_roi, recortada, np.nan)

        imagens_demarcadas[nome] = demarcada.astype(np.float32) / 255.0
        imagens_recortadas[nome] = recortada.astype(np.float32)

    return imagens_demarcadas, imagens_recortadas


def printar_mosaico_2d(imagens_2d: dict[str, np.ndarray], cols: int = 4, cmap: str = "jet", titulo: str | None = None) -> None:
    """Plota um mosaico (2D) com todas as imagens no dicionário."""
    if not imagens_2d:
        raise ValueError("Nenhuma imagem 2D para plotar.")

    n = len(imagens_2d)
    cols = max(1, cols)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.atleast_1d(axes).flatten()

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")

    for i, (chave, img) in enumerate(imagens_2d.items()):
        ax = axes[i]
        img_2d = np.asarray(img)
        img_plot = np.ma.masked_invalid(img_2d)
        ax.imshow(img_plot, cmap=cmap_obj, vmin=0, vmax=1)
        ax.set_title(str(chave)[:18], fontsize=8)
        ax.axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if titulo:
        fig.suptitle(titulo, y=0.98)
    plt.tight_layout()
    plt.show()


def hsi_3d_para_2d(img_hsi: np.ndarray, mascara_roi: np.ndarray) -> np.ndarray:
    """Converte cubo 3D para matriz 2D mantendo os pixels da ROI."""
    img_hsi = np.asarray(img_hsi)
    mascara_roi = np.asarray(mascara_roi, dtype=bool)

    if img_hsi.ndim != 3:
        raise ValueError(f"Esperado cubo 3D, recebido shape={img_hsi.shape}.")
    if mascara_roi.shape != img_hsi.shape[:2]:
        raise ValueError(f"Máscara ROI shape={mascara_roi.shape} incompatível com imagem shape={img_hsi.shape[:2]}.")

    return img_hsi[mascara_roi]


def salvar_espectros_csv(imagens_hsi: dict[str, np.ndarray], mascaras_roi: dict[str, np.ndarray], pasta_saida: str = CAMINHO_IMGS_PROCESSADAS, verbose: bool = True) -> dict[str, str]:
    """Salva os espectros em CSV e a máscara ROI correspondente em .npy."""
    caminhos: dict[str, str] = {}

    for nome, img_hsi in imagens_hsi.items():
        if nome not in mascaras_roi:
            raise KeyError(f"Máscara ROI não encontrada para '{nome}'.")

        mascara = mascaras_roi[nome]
        espectros_2d = hsi_3d_para_2d(img_hsi, mascara)

        nome_base = os.path.splitext(nome)[0]
        pasta_img = os.path.join(pasta_saida, nome_base)
        os.makedirs(pasta_img, exist_ok=True)

        caminho_csv = os.path.join(pasta_img, "espectros.csv")
        n_bandas = espectros_2d.shape[1]
        cabecalho = [f"banda_{i + 1}" for i in range(n_bandas)]

        with open(caminho_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(cabecalho)
            writer.writerows(espectros_2d.tolist())

        caminho_mascara = os.path.join(pasta_img, "mascara_roi.npy")
        np.save(caminho_mascara, mascara)
        caminhos[nome] = caminho_csv

        if verbose:
            print(f"[CSV] {nome}: {espectros_2d.shape[0]} pixels × {n_bandas} bandas → {caminho_csv}")

    return caminhos


if __name__ == "__main__":
    print("[PIPELINE] Iniciando leitura e calibração das imagens originais.")
    imagens_calibradas = ler_imagens_hyperspectrais_e_processadas(pasta_base=CAMINHO_IMGS_ORIGINAL)

    # REFINAMENTO: Alterado o corte para 25 bandas iniciais e 25 bandas finais
    n_primeiras = 25
    n_ultimas = 25
    imagens_normais = remover_bandas_ruidosas_refletancia(
        imagens_calibradas=imagens_calibradas,
        n_primeiras=n_primeiras,
        n_ultimas=n_ultimas,
        mostrar_shapes=True,
    )

    cht_params = dict(CHT_PARAMS)
    cht_params["bandas_rgb"] = tuple(max(1, b - n_primeiras) for b in cht_params["bandas_rgb"])

    print("\n[PIPELINE] Gerando visualizações com máscaras encolhidas (shrink_radius_px=20).")
    imagens_demarcadas, imagens_recortadas = aplicar_cht_e_preparar_visualizacoes(
        imagens=imagens_normais,
        **cht_params,
    )

    printar_mosaico_2d(
        imagens_demarcadas,
        cols=4,
        cmap="gray",
        titulo="Mosaico - todas as imagens com círculo demarcado (CHT)",
    )

    printar_mosaico_2d(
        imagens_recortadas,
        cols=4,
        cmap="gray",
        titulo="Mosaico - imagens recortadas pela ROI circular",
    )

    # Reconstruir e guardar dicionário de máscaras finais
    mascaras_roi: dict[str, np.ndarray] = {}
    for nome, img_hsi in imagens_normais.items():
        rgb = gerar_rgb_sintetico(img_hsi, bandas_rgb=cht_params["bandas_rgb"])
        _, _, mascara = detectar_circulo_e_mascara_roi(
            rgb_uint8=rgb,
            dp=cht_params["dp"],
            min_dist=cht_params["min_dist"],
            param1=cht_params["param1"],
            param2=cht_params["param2"],
            min_radius=cht_params["min_radius"],
            max_radius=cht_params["max_radius"],
            shrink_radius_px=cht_params["shrink_radius_px"],
        )
        mascaras_roi[nome] = mascara

    print("\n[PIPELINE] Exportando espectros limpos e máscaras compactas para arquivos.")
    salvar_espectros_csv(
        imagens_hsi=imagens_normais,
        mascaras_roi=mascaras_roi,
        pasta_saida=CAMINHO_IMGS_PROCESSADAS,
    )
    print("[PIPELINE] Processamento concluído com sucesso!")
