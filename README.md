# Neural Graph Collaborative Filtering (NGCF) - Implementação Didática

Este repositório contém uma implementação didática em Jupyter Notebook do artigo **Neural Graph Collaborative Filtering (NGCF)**, apresentado na SIGIR 2019. O objetivo principal é servir como um guia passo a passo, conectando a teoria do *paper* diretamente com o código prático. Este notebook **não é** uma implementação otimizada para produção, mas sim um recurso de aprendizado para entender os mecanismos internos do NGCF.

**Referência do Artigo Original:**
> Wang, X., He, X., Wang, M., Feng, F., & Chua, T. S. (2019). *Neural Graph Collaborative Filtering*. Em anais da 42ª Conferência Internacional ACM SIGIR sobre Pesquisa e Desenvolvimento em Recuperação de Informação (SIGIR '19).

---

## 1. Descrição do Projeto

O **Neural Graph Collaborative Filtering (NGCF)** é um modelo de recomendação *state-of-the-art* que utiliza Redes Neurais de Grafo (GNNs) para modelar as interações usuário-item. Ao contrário dos métodos tradicionais de Filtragem Colaborativa (como Matrix Factorization) que dependem apenas das interações diretas, o NGCF explora a estrutura do grafo bipartido usuário-item para capturar explicitamente **sinais de conectividade de alta ordem**. Isso significa que o modelo aprende embeddings propagando informações através dos vizinhos do grafo (itens que um usuário interagiu, usuários que interagiram com um item, etc.).

O notebook deste repositório implementa o NGCF usando o dataset **MovieLens 100k** e abrange:
* **Pré-processamento:** Carregamento dos dados e divisão em treino/validação/teste usando a estratégia *Leave-One-Out (LOO)*.
* **Construção do Grafo:** Criação da matriz de adjacência bipartida e da **normalização Laplaciana** ($L+I$), que é o núcleo da propagação de embeddings do NGCF.
* **Implementação do Modelo:** Uma classe PyTorch que implementa a Equação (7) do paper (propagação de embeddings) e a Equação (9) (concatenação de camadas).
* **Treinamento:** Otimização do modelo usando a perda **BPR (Bayesian Personalized Ranking)**.
* **Avaliação e Visualização:** Cálculo de métricas Top-K e gráficos que demonstram a vizinhança do grafo.
* **Benchmarking:** Comparação do desempenho do NGCF contra baselines de mercado e algoritmos clássicos.

---

## 2. Configuração do Ambiente e Execução

Para garantir a reprodutibilidade e evitar conflitos de versões (especialmente com bibliotecas científicas), utilize o procedimento abaixo com `venv` e o arquivo de requisitos fornecido.

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/ethoshomo/SR-NGCF.git
    cd SR-NGCF
    ```

2.  **Crie o ambiente virtual (.venv):**
    ```bash
    python3 -m venv .venv
    ```

3.  **Ative o ambiente virtual:**
    * **Linux/macOS:**
        ```bash
        source .venv/bin/activate
        ```
    * **Windows (PowerShell):**
        ```bash
        .\.venv\Scripts\Activate.ps1
        ```

4.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 3. Estrutura do Notebook

O notebook está organizado sequencialmente para facilitar o aprendizado teórico e prático:

* **Seção 1: Configurações Iniciais:** Importação de bibliotecas e definição de hiperparâmetros (como `EMBED_DIM`, `NUM_LAYERS`, `BATCH_SIZE`).
* **Seção 2: Carregamento e Pré-processamento:** Download do dataset MovieLens 100k, mapeamento de IDs e divisão dos dados com a estratégia *Leave-One-Out (LOO)*.
* **Seção 3: Construção do Grafo (NGCF):** Foco na implementação da normalização Laplaciana ($L = D^{-1/2} A D^{-1/2}$) e $L+I$, conectando o código com as equações do *paper*.
* **Seção 4: Visualização de Vizinhança:** Plotagem de exemplo de como o NGCF "enxerga" as vizinhanças em diferentes camadas (L=1, 2, 3).
* **Seção 5: Preparação do Dataset (BPR):** Implementação de um `Dataset` customizado do PyTorch para amostragem de tripletas (usuário, item positivo, item negativo) para a BPR loss.
* **Seção 6: Implementação do NGCF:** Definição da classe `NGCF` em PyTorch, detalhando a implementação das equações de propagação (Eq. 7) e predição (Eq. 10).
* **Seção 7: Treinamento:** O loop de treinamento completo, incluindo o cálculo da BPR loss + L2, e a avaliação por época no conjunto de validação.
* **Seção 8: Avaliação Final:** Plotagem das curvas de aprendizado, avaliação final no conjunto de teste e visualização gráfica das recomendações geradas.
* **Seção 9: Benchmarking:** Comparação robusta do NGCF implementado contra outros métodos usando as mesmas divisões de dados:
    * **Baselines:** Recomendação por Popularidade e Aleatória.
    * **FAISS:** Algoritmos de vizinhança (KNN) de alta performance (Item-KNN e User-KNN).
    * **Scikit-Surprise:** Algoritmos clássicos como SVD, SVD++, NMF e CoClustering.
    * **Resultados:** Tabela comparativa e gráfico de barras ordenado por métricas (Precision, Recall, NDCG, MRR).