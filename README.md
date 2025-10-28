# Neural Graph Collaborative Filtering (NGCF) - Implementação Didática

Este repositório contém uma implementação didática em Jupyter Notebook do artigo **Neural Graph Collaborative Filtering (NGCF)**, apresentado na SIGIR 2019. O objetivo principal é servir como um guia passo a passo, conectando a teoria do *paper* diretamente com o código prático. Este notebook **não é** uma implementação otimizada para produção, mas sim um recurso de aprendizado para entender os mecanismos internos do NGCF.

**Referência do Artigo Original:**
> Wang, X., He, X., Wang, M., Feng, F., & Chua, T. S. (2019). *Neural Graph Collaborative Filtering*. Em anais da 42ª Conferência Internacional ACM SIGIR sobre Pesquisa e Desenvolvimento em Recuperação de Informação (SIGIR '19).

---

## 1. Descrição do Projeto

O **Neural Graph Collaborative Filtering (NGCF)** é um modelo de recomendação *state-of-the-art* que utiliza Redes Neurais de Grafo (GNNs) para modelar as interações usuário-item. Ao contrário dos métodos tradicionais de Filtragem Colaborativa (como Matrix Factorization) que dependem apenas das interações diretas, o NGCF explora a estrutura do grafo bipartido usuário-item para capturar explicitamente **sinais de conectividade de alta ordem**. Isso significa que o modelo aprende embeddings propagando informações através dos vizinhos do grafo (itens que um usuário interagiu, usuários que interagiram com um item, etc.).

Este notebook implementa o NGCF usando o dataset **MovieLens 100k** e abrange:
* **Pré-processamento:** Carregamento dos dados e divisão em treino/validação/teste usando a estratégia *Leave-One-Out (LOO)*.
* **Construção do Grafo:** Criação da matriz de adjacência bipartida e da **normalização Laplaciana** ($L+I$), que é o núcleo da propagação de embeddings do NGCF.
* **Implementação do Modelo:** Uma classe PyTorch que implementa a Equação (7) do paper (propagação de embeddings) e a Equação (9) (concatenação de camadas).
* **Treinamento:** Otimização do modelo usando a perda **BPR (Bayesian Personalized Ranking)**.
* **Avaliação:** Cálculo de métricas Top-K, incluindo Recall@K, Precision@K, NDCG@K e MRR@K.
* **Visualização:** Gráficos que demonstram a vizinhança do grafo e os resultados da recomendação.

## 2. Tecnologias Utilizadas

* **Python 3.10+**
* **PyTorch:** Para a implementação do modelo de rede neural.
* **Pandas & NumPy:** Para manipulação e processamento dos dados.
* **SciPy:** Utilizado especificamente para a criação de matrizes esparsas (Laplaciana).
* **Matplotlib & NetworkX:** Para a visualização dos grafos e dos resultados do treinamento.
* **Requests:** Para o download automático do dataset MovieLens.
* **Jupyter Notebook:** Para a execução interativa e didática do código.

---

## 3. Configuração do Ambiente e Execução

Para executar este notebook, é altamente recomendado criar um ambiente virtual isolado para gerenciar as dependências.

### Opção 1: Usando `venv` (Padrão Python)

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/ethoshomo/SR-NGCF.git](https://github.com/ethoshomo/SR-NGCF.git)
    cd SR-NGCF
    ```

2.  **Crie um ambiente virtual:**
    ```bash
    python3 -m venv .venv
    ```

3.  **Ative o ambiente virtual:**
    * No macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```
    * No Windows (PowerShell):
        ```bash
        .\.venv\Scripts\Activate.ps1
        ```
    * No Windows (CMD):
        ```bash
        .\.venv\Scripts\activate.bat
        ```

4.  **Instale as dependências:**

    ```bash
    pip install torch pandas numpy scipy matplotlib networkx requests jupyter
    ```

### Opção 2: Usando `conda` 

O `conda` facilita o gerenciamento de dependências complexas, especialmente do PyTorch (incluindo o CUDA, se você tiver uma GPU NVIDIA).

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/ethoshomo/SR-NGCF.git](https://github.com/ethoshomo/SR-NGCF.git)
    cd SR-NGCF
    ```

2.  **Crie um ambiente conda:**
    ```bash
    conda create -n ngcf_env python=3.10
    ```

3.  **Ative o ambiente:**
    ```bash
    conda activate ngcf_env
    ```

4.  **Instale as dependências (instalando o PyTorch primeiro):**
    *Visite [https://pytorch.org/](https://pytorch.org/) para obter o comando de instalação correto para seu sistema (CPU ou versão específica do CUDA).*

    ```bash
    # Exemplo para CPU:
    conda install pytorch torchvision torchaudio -c pytorch
    
    # Exemplo para CUDA 12.1:
    # conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    
    # Instalar o restante
    conda install pandas numpy scipy matplotlib networkx requests jupyter
    ```

### Executando o Notebook

Após ativar o ambiente e instalar as dependências:

1. **Inicie o Jupyter:**
    ```bash
    jupyter notebook
    ```
    (Ou `jupyter lab` se preferir)

2. **Abra o arquivo:**
    No seu navegador, abra o arquivo `NGCF.ipynb`.

3. **Execute o código:**
    Você pode executar todas as células de uma vez (no menu, `Cell` > `Run All`) ou executar cada célula individualmente (usando `Shift + Enter`) para acompanhar o tutorial passo a passo. O notebook foi projetado para ser executado sequencialmente.

---

## 4. Estrutura do Notebook

O notebook está dividido nas seguintes seções para facilitar o aprendizado:

* **Seção 1: Configurações Iniciais:** Importação de bibliotecas e definição de hiperparâmetros (como `EMBED_DIM`, `NUM_LAYERS`, `BATCH_SIZE`).
* **Seção 2: Carregamento e Pré-processamento:** Download do dataset MovieLens 100k, mapeamento de IDs e divisão dos dados com a estratégia *Leave-One-Out (LOO)*.
* **Seção 3: Construção do Grafo (NGCF):** Foco na implementação da normalização Laplaciana ($L = D^{-1/2} A D^{-1/2}$) e $L+I$, conectando o código com as equações do *paper*.
* **Seção 4: Visualização de Vizinhança:** Plotagem de exemplo de como o NGCF "enxerga" as vizinhanças em diferentes camadas (L=1, 2, 3).
* **Seção 5: Preparação do Dataset (BPR):** Implementação de um `Dataset` customizado do PyTorch para amostragem de tripletas (usuário, item positivo, item negativo) para a BPR loss.
* **Seção 6: Implementação do NGCF:** Onde a classe `NGCF` é definida em PyTorch, detalhando a implementação das equações de propagação (Eq. 7) e predição (Eq. 10).
* **Seção 7: Treinamento:** O loop de treinamento completo, incluindo o cálculo da BPR loss + L2, e a avaliação por época no conjunto de validação.
* **Seção 8: Avaliação Final:** Plotagem das curvas de loss/métricas, avaliação final no conjunto de teste e geração de um gráfico de exemplo das recomendações.