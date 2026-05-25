import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
} from 'cursor/canvas';

const clusterRows = [
  ['3491B_240506-155911', 'desconhecido', '15.373', '0,0%', '100,0%', '100,0%'],
  ['351_1_240506-160920', 'desconhecido', '15.813', '0,4%', '99,6%', '99,6%'],
  ['420_240506-160205', 'desconhecido', '13.273', '0,0%', '100,0%', '100,0%'],
  ['426_1_240506-160058', 'desconhecido', '13.673', '0,3%', '99,7%', '99,7%'],
  ['491_240506-160846', 'desconhecido', '15.373', '0,0%', '100,0%', '100,0%'],
  ['503B_240506-160006', 'desconhecido', '14.949', '52,4%', '47,6%', '52,4%'],
  ['508_240506-160326', 'desconhecido', '17.193', '0,3%', '99,7%', '99,7%'],
  ['534_240506-160510', 'desconhecido', '16.729', '4,4%', '95,6%', '95,6%'],
  ['537_240506-160651', 'desconhecido', '15.373', '0,2%', '99,8%', '99,8%'],
  ['539_240506-160131', 'desconhecido', '15.373', '0,5%', '99,5%', '99,5%'],
  ['548A_240506-160359', 'desconhecido', '13.273', '0,0%', '100,0%', '100,0%'],
  ['549_240506-160758', 'desconhecido', '14.949', '0,0%', '100,0%', '100,0%'],
  ['553_240506-160724', 'desconhecido', '14.073', '0,0%', '100,0%', '100,0%'],
  ['554A_240506-160433', 'desconhecido', '16.729', '90,0%', '10,0%', '90,0%'],
  ['559_240506-160951', 'desconhecido', '13.673', '80,4%', '19,6%', '80,4%'],
  ['560B_240506-160251', 'desconhecido', '15.373', '0,0%', '100,0%', '100,0%'],
  ['ATCC13_240506-161053', 'resistente', '15.373', '16,7%', '83,3%', '83,3%'],
  ['ATCC16_240506-161158', 'resistente', '16.241', '0,0%', '100,0%', '100,0%'],
  ['ATCC27_240506-161129', 'resistente', '15.373', '0,8%', '99,2%', '99,2%'],
];

const clusterTone = clusterRows.map((row) => {
  if (row[1] === 'resistente') return 'success' as const;
  if (row[0] === '503B_240506-160006') return 'warning' as const;
  if (row[0] === '554A_240506-160433' || row[0] === '559_240506-160951') return 'info' as const;
  return undefined;
});

const kCategories = ['2', '3', '4', '5', '6', '7', '8', '9', '10'];
const silhouette = [0.5687, 0.5071, 0.5267, 0.5305, 0.5428, 0.5571, 0.5381, 0.4622, 0.4733];
const inertia = [1420.4, 951.0, 783.6, 616.7, 536.6, 458.4, 390.3, 338.5, 302.2];

const flowRows = [
  ['1', 'processamento.py', 'Lê cubos ENVI, DarkRef e WhiteRef; calibra reflectância.'],
  ['2', 'processamento.py', 'Remove bandas ruidosas e detecta ROI circular com Hough.'],
  ['3', 'dados/img_processadas', 'Salva espectros 2D por pixel e máscara ROI por imagem.'],
  ['4', 'max_min_train.py', 'Normaliza Min-Max global, concatena pixels e roda PCA.'],
  ['5', 'max_min_train.py', 'Escolhe k, treina K-Means global e reconstrói labels por imagem.'],
  ['6', 'dados/figuras e CSVs', 'Gera métricas, gráficos e centroides por amostra.'],
];

const riskRows = [
  ['Alta', 'Validação supervisionada fraca', 'Só há 3 controles resistentes; as 16 demais imagens estão como desconhecido. ARI/NMI globais ficam 0 e não sustentam conclusão de sensibilidade.'],
  ['Alta', 'Documentação divergente', 'README descreve 176 bandas, mas o pipeline atual gera 206 bandas após remover 25 iniciais e 25 finais de 256.'],
  ['Média', 'Fallback de ROI perigoso', 'Se HoughCircles falhar, a ROI vira a imagem inteira, incluindo fundo e ágar sem registrar falha forte.'],
  ['Média', 'Modelos não persistidos', 'Scaler Min-Max, PCA e K-Means não são salvos; fica difícil reproduzir inferência em novas amostras.'],
  ['Média', 'Padronização PCA inconsistente', 'PCA_USAR_PADRONIZACAO só afeta a projeção 2D, não o PCA usado no treino global.'],
  ['Baixa', 'Dependências sem versão', 'requirements.txt não fixa versões, abrindo risco de resultado diferente entre ambientes.'],
];

const riskTone = ['danger', 'danger', 'warning', 'warning', 'warning', 'neutral'] as const;

export default function AnaliseProjetoHiperespectral() {
  return (
    <Stack gap={20}>
      <Stack gap={8}>
        <H1>Análise do Projeto Hiperespectral</H1>
        <Text tone="secondary">
          Pipeline Python para processar cubos hiperespectrais bacterianos, extrair ROI, normalizar espectros, reduzir dimensionalidade e agrupar pixels via K-Means global.
        </Text>
        <Row gap={8} wrap>
          <Pill tone="info" active>Python</Pill>
          <Pill tone="info" active>OpenCV</Pill>
          <Pill tone="info" active>scikit-learn</Pill>
          <Pill tone="info" active>Spectral Python</Pill>
        </Row>
      </Stack>

      <Grid columns={4} gap={16}>
        <Stat value="19" label="amostras originais e processadas" tone="info" />
        <Stat value="288.179" label="pixels dentro das ROIs" />
        <Stat value="206" label="bandas por espectro processado" tone="warning" />
        <Stat value="49" label="figuras geradas" />
      </Grid>

      <Callout tone="warning" title="Leitura principal">
        O projeto já executou o pipeline completo, mas a interpretação biológica ainda é frágil: o cluster de resistência foi inferido a partir de apenas três ATCC resistentes, sem controles sensíveis rotulados para comparação.
      </Callout>

      <H2>Fluxo Do Pipeline</H2>
      <Table
        headers={['Etapa', 'Área', 'Função no projeto']}
        rows={flowRows}
        columnAlign={['center', 'left', 'left']}
      />

      <Grid columns="1fr 1fr" gap={16}>
        <Card>
          <CardHeader trailing={<Pill tone="success" size="sm" active>k final 2</Pill>}>
            Escolha de k por silhouette
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <BarChart
                categories={kCategories}
                series={[{ name: 'Silhouette médio', data: silhouette, tone: 'success' }]}
                height={220}
              />
              <Text size="small" tone="secondary">
                Eixo X: número de clusters k. Eixo Y: silhouette médio sem unidade. Fonte: `dados/escolha_k.csv`, subamostra de 10.000 pixels.
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={<Pill tone="info" size="sm" active>cotovelo 5</Pill>}>
            Inércia por k
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <BarChart
                categories={kCategories}
                series={[{ name: 'Inércia WCSS', data: inertia, tone: 'info' }]}
                height={220}
              />
              <Text size="small" tone="secondary">
                Eixo X: número de clusters k. Eixo Y: inércia WCSS. Fonte: `dados/escolha_k.csv`; o cotovelo sugeriu k=5, mas o critério final usou silhouette.
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <H2>Distribuição Dos Clusters</H2>
      <Text>
        As ATCC resistentes são majoritariamente cluster 1, então o script classifica esse cluster como perfil de resistência. A maior parte das amostras desconhecidas também cai em cluster 1; `554A` e `559` se destacam no cluster 0, e `503B` fica dividido.
      </Text>
      <Table
        headers={['Imagem', 'Rótulo', 'Pixels ROI', 'Cluster 0', 'Cluster 1', 'Pureza']}
        rows={clusterRows}
        rowTone={clusterTone}
        columnAlign={['left', 'left', 'right', 'right', 'right', 'right']}
        striped
      />

      <Divider />

      <H2>Riscos E Lacunas</H2>
      <Table
        headers={['Severidade', 'Achado', 'Impacto']}
        rows={riskRows}
        rowTone={riskTone}
        columnAlign={['left', 'left', 'left']}
      />

      <Grid columns="1fr 1fr" gap={16}>
        <Stack gap={8}>
          <H3>O Que Está Bom</H3>
          <Text>Separação clara entre processamento físico da imagem e treinamento global.</Text>
          <Text>Validação de consistência entre `espectros.csv` e `mascara_roi.npy` antes do treino.</Text>
          <Text>Uso de PCA antes do K-Means reduz dimensionalidade e torna o agrupamento mais estável.</Text>
          <Text>Artefatos intermediários ficam salvos por amostra, facilitando auditoria visual.</Text>
        </Stack>

        <Stack gap={8}>
          <H3>Próximas Prioridades</H3>
          <Text>Corrigir README para 206 bandas e alinhar descrição com o código atual.</Text>
          <Text>Adicionar controles sensíveis rotulados antes de usar ARI/NMI como evidência.</Text>
          <Text>Salvar Min-Max, PCA e K-Means para reaplicar exatamente em novas amostras.</Text>
          <Text>Transformar falhas de ROI em alerta/relatório por imagem, não fallback silencioso.</Text>
        </Stack>
      </Grid>

      <Text size="small" tone="tertiary">
        Fonte: leitura dos scripts `processamento.py`, `max_min_train.py`, `README.md`, `requirements.txt` e artefatos locais em `dados/`, analisados em 25/05/2026.
      </Text>
    </Stack>
  );
}
