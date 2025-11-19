# Tutorial de Uso do Neural Graph Collaborative Filtering (NGCF)

## Integrantes do Projeto Acadêmico

- Carlos Filipe de Castro Lemos
- Kaito Hayashi

## Contextualização do Projeto Acadêmico
Este material foi desenvolvido como parte das atividades da disciplina **SCC0284 - Sistemas de Recomendação** (2º Semestre/2025), oferecida pelo **Instituto de Ciências Matemáticas e de Computação (ICMC)** da **Universidade de São Paulo (USP)**, sob orientação do **Prof. Dr. Marcelo G. Manzato**.

O projeto se enquadra na modalidade de **Projeto Extensionista** via elaboração de material didático. O objetivo principal é produzir conteúdo técnico acessível para a comunidade externa à universidade (pesquisadores, estudantes e profissionais de tecnologia), abordando tópicos avançados de Sistemas de Recomendação — especificamente a implementação de algoritmos e escalabilidade — que expandem o conteúdo programático tradicional visto em sala de aula. As descrições detalhadas do projeto podem ser encontradas na pasta `projeto`.

---

## Materiais de Apoio e Avaliação
Para enriquecer o aprendizado e facilitar a reprodução deste tutorial, disponibilizamos duas videoaulas complementares e um canal para feedback:

* 🎥 **Vídeo 1: Fundamentos Teóricos** – Uma explicação visual sobre o funcionamento do NGCF e a intuição por trás de Grafos em Sistemas de Recomendação.
    * [Neural Graph Collaborative Filtering (NGCF) - parte 1 (teórica)](https://www.youtube.com/watch?v=6CXg-Mc3Kgo&list=PLcEF5dvxyCWc_2ymEzeYPabFhJYn9WCdR&index=1)
* 💻 **Vídeo 2: Tutorial Prático (Code Walkthrough)** – Um guia passo a passo rodando o notebook, explicando a implementação das classes e o treinamento do modelo.
    * [Neural Graph Collaborative Filtering (NGCF) - parte 2 (prática)](https://www.youtube.com/watch?v=x5j7wG1bCN4&list=PLcEF5dvxyCWc_2ymEzeYPabFhJYn9WCdR&index=2)

📝 **Avaliação pela Comunidade Externa:**
Se você **não possui vínculo com a USP** (é estudante de outra instituição, pesquisador ou profissional de mercado), sua avaliação é essencial para validarmos este projeto de extensão. Por favor, preencha o formulário abaixo para nos dar seu feedback sobre a qualidade e utilidade deste material didático:
* 👉 **[Formulário de Avaliação do Projeto](https://docs.google.com/forms/d/e/1FAIpQLSdX2u1OLRcy7wm1JixkeqyZUWe0TcUg2gmw3tyIGFDdB5wldw/viewform?usp=sharing&ouid=113581016036571253565)**

---

## Descrição do Algoritmo Neural Graph Collaborative Filtering (NGCF)

O **Neural Graph Collaborative Filtering (NGCF)** é um modelo de recomendação *state-of-the-art* que utiliza Redes Neurais de Grafo (GNNs) para modelar as interações usuário-item. Ao contrário dos métodos tradicionais de Filtragem Colaborativa (como Matrix Factorization) que dependem apenas das interações diretas, o NGCF explora a estrutura do grafo bipartido usuário-item para capturar explicitamente **sinais de conectividade de alta ordem**. Isso significa que o modelo aprende embeddings propagando informações através dos vizinhos do grafo (itens que um usuário interagiu, usuários que interagiram com um item, etc.).

Este repositório contém uma implementação didática em Jupyter Notebook do artigo **Neural Graph Collaborative Filtering (NGCF)**, apresentado na SIGIR 2019. O objetivo principal é servir como um guia passo a passo, conectando a teoria do *paper* diretamente com o código prático. Este notebook **não é** uma implementação otimizada para produção, mas sim um recurso de aprendizado para entender os mecanismos internos do NGCF.

**Referência do Artigo Original:**
> Wang, X., He, X., Wang, M., Feng, F., & Chua, T. S. (2019). *Neural Graph Collaborative Filtering*. Em anais da 42ª Conferência Internacional ACM SIGIR sobre Pesquisa e Desenvolvimento em Recuperação de Informação (SIGIR '19).

O notebook deste repositório implementa o NGCF usando o dataset **MovieLens 100k** e abrange:
* **Pré-processamento:** Carregamento dos dados e divisão em treino/validação/teste usando a estratégia *Leave-One-Out (LOO)*.
* **Construção do Grafo:** Criação da matriz de adjacência bipartida e da **normalização Laplaciana** ($L+I$), que é o núcleo da propagação de embeddings do NGCF.
* **Implementação do Modelo:** Uma classe PyTorch que implementa a Equação (7) do paper (propagação de embeddings) e a Equação (9) (concatenação de camadas).
* **Treinamento:** Otimização do modelo usando a perda **BPR (Bayesian Personalized Ranking)**.
* **Avaliação e Visualização:** Cálculo de métricas Top-K e gráficos que demonstram a vizinhança do grafo.
* **Benchmarking:** Comparação do desempenho do NGCF contra baselines de mercado e algoritmos clássicos.

---

## Configuração do Ambiente e Execução

Para garantir a reprodutibilidade e evitar conflitos de versões (especialmente com bibliotecas científicas), utilize o procedimento abaixo com `venv` e o arquivo de requisitos fornecido.

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/ethoshomo/SR-NGCF.git](https://github.com/ethoshomo/SR-NGCF.git)
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
    
5. **Execução:** abrir o código no jupiter notebook (disponibilizado no diretório `codigo`).

> Caso haja dificuldade, favor entrar em contato com os integrantes do projeto.
