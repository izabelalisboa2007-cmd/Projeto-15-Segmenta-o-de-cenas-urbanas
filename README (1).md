# Segmentação Semântica de Cenas Urbanas com DeepLabV3-ResNet50

Este repositório contém uma implementação robusta em PyTorch para **Segmentação Semântica** em imagens de trânsito e cenas urbanas. O projeto utiliza a renomada arquitetura **DeepLabV3** acoplada ao backbone **ResNet-50** pré-treinado, realizando um processo de *Transfer Learning* e *Fine-Tuning* para mapear com precisão 12 classes de objetos presentes em vias públicas.

O projeto foi validado por meio de uma **Investigação Experimental de Resoluções**, analisando o impacto do tamanho das imagens de entrada no desempenho de convergência da rede e nas métricas de acurácia.

---

## 🚀 Funcionalidades do Projeto

- **Transfer Learning Adaptativo:** Carga do modelo `deeplabv3_resnet50` pré-treinado no ImageNet com reestruturação dinâmica das camadas de classificação (`classifier` e `aux_classifier`) para 12 classes exclusivas.
- **Dataloader para Cenários Urbanos:** Customização da classe `CityscapesDataset` adaptada para o consumo nativo e direto do dataset **CamVid** (Cambridge-driving Labeled Video Database).
- **Pipeline de Data Augmentation:** Implementação de transformações estocásticas (como espelhamento horizontal aleatório) e interpolações diferenciadas para imagens (`BILINEAR`) e máscaras (`NEAREST`).
- **Métricas Avançadas de Avaliação:** Computação estrita de Intersection over Union (IoU) por classe, mIoU (Mean Intersection over Union), Acurácia Global e Acurácia por classe, desconsiderando pixels de borda indesejados (`ignore_index=255`).
- **Validação Experimental:** Loop comparativo completo de treinamento executado sob duas resoluções distintas:
  - **Experimento 1:** 128x256 pixels
  - **Experimento 2:** 256x512 pixels

---

## 📦 Dataset Utilizado

Para viabilizar a execução fluida e imediata em ambientes de nuvem (como o Google Colab), o pipeline realiza automaticamente o clone e extração do **CamVid Dataset**, estruturado no repositório público do *SegNet-Tutorial*. 

### Divisão dos Dados:
- **Imagens de Treino:** 367
- **Imagens de Validação:** 101
- **Imagens de Teste:** 233
- **Total de classes:** 12 (11 categorias semânticas + 1 classe de mapeamento nulo/*void*).

---

## 🛠️ Pré-requisitos e Instalação

As principais dependências de bibliotecas para rodar o arquivo são:
- Python 3.8+
- PyTorch (com suporte a CUDA para aceleração por GPU T4)
- Torchvision
- NumPy
- Matplotlib
- Pillow (PIL)
- tqdm

Para preparar o seu ambiente local, execute:
```bash
pip install torch torchvision numpy matplotlib pillow tqdm
```

---

## 💻 Estrutura do Pipeline de Código

O código está modularizado e documentado nas seguintes etapas principais:

1. **Configuração de Ambiente e Reprodutibilidade:** Definição manual de sementes aleatórias (`seed=42`) para travar o comportamento estocástico do PyTorch, NumPy e CUDA.
2. **Modelagem:** Customização da rede convolucional profunda através da substituição do cabeçalho original por blocos `nn.Conv2d` com `kernel_size=1`.
3. **Data Pipeline:** Criação das resoluções alvo e normalização baseada nos parâmetros estatísticos do ImageNet:
   $$\text{Mean} = [0.485, 0.456, 0.406], \quad \text{Std} = [0.229, 0.224, 0.225]$$
4. **Calculador de Métricas:** Isolamento matricial via tensores de predição e alvos para extração do cálculo matemático do IoU:
   $$\text{IoU} = \frac{\text{Interseção}}{\text{União}}$$
5. **Loops de Otimização:** Funções isoladas de `train_one_epoch` e `validate_model` gerenciando passos de Forward, Backward e atualização de pesos pelo otimizador utilizando a perda de Entropia Cruzada (`CrossEntropyLoss`).

---

## 📈 Resultados da Investigação Experimental

Durante a execução padrão de 5 épocas, pôde-se notar um ganho perceptivo e métrico ao elevar a fidelidade dimensional do processamento da imagem de entrada:

| Métrica Obtida (Época 5) | Experimento 1 (128x256) | Experimento 2 (256x512) |
| :--- | :---: | :---: |
| **Train Loss** | 0.6275 | **0.6111** |
| **Val Loss** | 0.7071 | **0.5448** |
| **Val mIoU** | 0.3128 | **0.3643** |

### Gráficos de Aprendizado
Ao final da execução, a aplicação gera automaticamente curvas comparativas utilizando `matplotlib.pyplot` expondo o decaimento da perda (*Loss*) e a subida da acurácia média de segmentação por pixel.

---

## 👨‍💻 Como Executar

Se estiver utilizando o ambiente do Jupyter ou Google Colab, basta abrir o arquivo `.ipynb` e executar sequencialmente as células. O download do dataset CamVid será engatilhado via comando de terminal incorporado `git clone --depth 1` logo na célula de estruturação de dados.

Para rodar como script Python puro:
```bash
python segmentacao_deeplab.py
```
