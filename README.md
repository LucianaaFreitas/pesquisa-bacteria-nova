## 🧬 Estrutura e Funcionamento do Pipeline Hiperespectral

O projeto está dividido em dois scripts principais que operam de forma isolada e sequencial. O objetivo é transformar cubos hiperespectrais brutos em um relatório consolidado de agrupamento (Clusterização Global) para identificar perfis de sensibilidade e resistência bacteriana.

---

### 🎛️ Script 1: `processamento.py` (Preparação e Extração dos Dados)

Este script lida com a limpeza física, calibração óptica e isolamento da região de interesse biológico, removendo a interferência do ambiente e do meio de cultura.

1. **Varredura e Carga dos Dados:** O script busca no diretório `dados/img_original` por subpastas de captura. Ele localiza o arquivo hiperespectral principal (`.hdr`/`.raw`) e seus respectivos arquivos de calibração de referência (*Dark* e *White*).
2. **Calibração Radiométrica (Reflectância):** Para neutralizar variações na intensidade da luz do scanner entre diferentes dias de coleta, a imagem bruta é calibrada. O script calcula a média das imagens escuras (*DarkRef*) e claras (*WhiteRef*) e aplica a fórmula de broadcasting:  
   $$\text{Reflectância} = \frac{\text{Imagem} - \text{Média\_Escura}}{\text{Média\_Clara} - \text{Média\_Escura}}$$
3. **Corte de Bandas Ruidosas:** Sensores hiperespectrais geram ruído eletrônico severo nas extremidades do espectro (UV e Infravermelho Próximo). O script **remove as 25 primeiras e as 25 últimas bandas**, preservando as 176 bandas centrais estáveis para análise.
4. **Detecção da Colônia (Transformada de Hough):** É gerada uma imagem RGB sintética para guiar o algoritmo. A imagem é convertida para escala de cinza, passa por equalização de histograma e suavização Gaussiana. A função `HoughCircles` do OpenCV localiza o formato circular da colônia bacteriana.
5. **Foco no Miolo Biológico (Máscara ROI Recuada):** Para garantir que o meio de cultura (ágar) e os efeitos físicos de borda da colônia não interfiram no modelo, o raio do círculo detectado é **encolhido em 20 pixels** (`shrink_radius_px=20`). Apenas os pixels deste miolo puro tornam-se a máscara ROI ativa.
6. **Exportação Matricial:** O cubo 3D é fatiado com a máscara, gerando uma tabela 2D onde cada linha representa um pixel e cada coluna uma banda espectral. O script exporta `espectros.csv` e `mascara_roi.npy` na pasta `dados/img_processadas/<amostra>`.

---

### 🧠 Script 2: `max_min_train.py` (Inteligência Artificial e Aprendizado Global)

Este script unifica todas as amostras em uma matriz única para que o modelo crie um padrão de comparação global estável entre pixels semelhantes.

1. **Leitura e Validação Sincronizada:** O script lê os arquivos `espectros.csv` e `mascara_roi.npy` de todas as amostras processadas, validando matematicamente se o número de linhas do CSV condiz com os pixels ativos da máscara.
2. **Normalização Min-Max Global:** Todos os espectros das 19 imagens (16 desconhecidas + 3 resistentes de controle) são temporariamente empilhados para calcular o mínimo e máximo absoluto de cada uma das 176 bandas. Cada pixel de cada imagem é então escalado estritamente entre `[0, 1]`. Isso remove variações de amplitude e torna a reflectância de placas diferentes comparável.
3. **Concatenação Global de Matrizes:** O script une os pixels normalizados de todo o experimento em um tabelão único chamado `X_total`. Paralelamente, gera vetores de rastreamento para saber a origem de cada pixel (`nomes_pixels`) e os rótulos de controle (`rotulos_pixels`).
4. **PCA Global (Redução de Dimensionalidade Pré-Treino):** O `X_total` entra no algoritmo **PCA**, que comprime as 176 bandas hiperespectrais em componentes principais que retêm 95% da variância da informação. Isso limpa dados redundantes, protege o modelo contra a *maldição da dimensionalidade* e gera a matriz otimizada `Z_total`.
5. **Escolha de $k$ (Cotovelo + Silhouette):** Antes do clustering, o script varre $k \in [2, 10]$ em uma subamostra, gera gráficos de **inércia (cotovelo)** e **silhouette score**, e define $k$ final pelo critério configurado (padrão: maior silhouette).
6. **Treinamento do K-Means Global:** O **K-Means** é treinado sobre o espaço reduzido do PCA (`Z_total`) com o $k$ escolhido, agrupando todos os pixels por distância euclidiana espectral.
7. **Propagação Reversa e Cálculo de Centroides:** O array de clusters globais é fatiado de volta para o formato de cada imagem individual usando mapeamento de memória por fatias (`slices`). Para cada imagem, reconstrói-se o mapa espacial 2D do cluster e calcula-se a curva média (centroide) de reflectância voltando para o espaço original de bandas (e não no PCA), gerando dados interpretáveis fisicamente.
8. **Inferência por Imagens de Controle:** O script analisa os resultados das 3 imagens sabidamente **Resistentes (ATCC)**. Ele verifica qual ID de cluster (0 ou 1) dominou a massa biológica dessas 3 amostras e define a regra de votação (ex: se as ATCC foram majoritariamente Cluster 0, o Cluster 0 é carimbado como o perfil de Resistência).
9. **Relatórios e Gráficos:** O pipeline se encerra gerando gráficos de dispersão **PCA 2D e t-SNE 2D**, cálculo de **Pureza** (para checar se a colônia foi dividida internamente) e o **Mosaico Geral de Clusters**, permitindo visualizar imediatamente quais das 16 imagens desconhecidas se comportaram de forma idêntica ao grupo resistente conhecido.
