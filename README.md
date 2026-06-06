# Projeto Final RecSys
## Visão Geral

Este repositório fornece um pequeno framework para comparar algoritmos de
recomendação. Inclui um runner de experimentos, implementações de exemplo
(baseadas em memória e em modelo), um registro para registrar algoritmos e
utilitários para avaliar e salvar resultados para análise e plotagem.


## Métricas de Avaliação

- Previsão de avaliações (rating): RMSE e MAE. RMSE penaliza erros maiores;
  MAE fornece um erro médio absoluto mais interpretável. Usar ambos dá uma
  visão mais completa da qualidade das previsões.
- Qualidade de recomendações (top-K): Precision@K, Recall@K, NDCG@K, MAP@K.
  Precision e recall avaliam acurácia e cobertura; NDCG e MAP consideram a
  ordenação dos itens — importantes quando a posição importa.


## Setup

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Baixe o dataset do MovieLens 100k:
```bash
./movielens_install.sh
```

3. Execute todos os recomendadores registrados:

```bash
python run.py
```

Para executar um algoritmo específico:

```bash
python run.py simple_memory
```

Saídas em `results/`:
- `summary.csv` — uma linha por algoritmo com métricas agregadas
- `predictions_<algo>.csv` — previsões verdadeiras e previstas por par usuário-item
- `recommendations_<algo>.csv` — recomendações ranqueadas por usuário com indicador de acerto

Gerar gráficos
---------------

Para gerar scatter plots, curvas ROC e matrizes de confusão a partir das
previsões salvas, use o script `charts.py`. Os arquivos PDF são salvos em
`results/charts/<algo>/`.

Exemplos:

```bash
# gerar gráficos para todos os algoritmos presentes em results/
python charts.py

# gerar apenas para um algoritmo registrado (ex.: simple_memory)
python charts.py simple_memory
```


## Algoritmos incluídos

- `memoryBased/simple_memory.py`: baselines simples (popularidade, médias por usuário/item).
- `memoryBased/user_based.py`: kNN baseado em usuários (cosine sobre ratings centrados).
- `memoryBased/item_based.py`: kNN baseado em itens (adjusted-cosine / similaridade por itens).
- `memoryBased/slope_one.py`: algoritmo Slope One com suavização por contagem (shrinkage).
- `modelBased/simple_model.py`: baseline com biases globais/usuário/item.
- `modelBased/regularized_svd.py`: SVD regularizado (SGD, fatores latentes, biases).
- `modelBased/bpr.py`: BPR (pairwise ranking) para feedback implícito.
- `modelBased/fm.py`: Factorization Machines (one-hot user/item features, extensível).