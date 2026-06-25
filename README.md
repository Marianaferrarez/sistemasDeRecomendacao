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

Hiperparametrização
-------------------

Para buscar os melhores hiperparâmetros via grid search com validação cruzada
de 5 folds (splits `u1`–`u5` do MovieLens):

```bash
# todos os algoritmos
python hyperparameter_tuning.py

# apenas algoritmos específicos
python hyperparameter_tuning.py svd fm
```

Saídas em `results/`:
- `hyperparameter_tuning.csv` — todas as combinações testadas com RMSE médio e desvio padrão
- `best_hyperparameters.csv` — resumo dos melhores parâmetros por algoritmo

Execução com melhores hiperparâmetros
--------------------------------------

Para re-executar a avaliação completa usando os melhores hiperparâmetros
encontrados no tuning:

```bash
python run_best.py
```

Sobrescreve os arquivos em `results/` com os resultados otimizados.

Análise de sensibilidade
-------------------------

Para investigar o impacto de um hiperparâmetro-chave no desempenho de cada
algoritmo e gerar gráficos:

```bash
# todos os algoritmos
python sensitivity_analysis.py

# apenas algoritmos específicos
python sensitivity_analysis.py svd user_based
```

Saídas em `results/sensitivity/`:
- `sensitivity_results.csv` — RMSE por valor de parâmetro por algoritmo
- `<algo>_<param>.pdf` — gráfico de sensibilidade por algoritmo

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

Relatório
----------

O relatório final está em `relatorio.tex` (template ACM). Para compilar:

```bash
pdflatex relatorio.tex
pdflatex relatorio.tex
```

O PDF gerado inclui tabelas de resultados, gráficos de visualização e análise
de sensibilidade.


## Algoritmos incluídos

- `memoryBased/simple_memory.py`: baselines simples (popularidade, médias por usuário/item).
- `memoryBased/user_based.py`: kNN baseado em usuários (cosine sobre ratings centrados).
- `memoryBased/item_based.py`: kNN baseado em itens (adjusted-cosine / similaridade por itens).
- `memoryBased/slope_one.py`: algoritmo Slope One com suavização por contagem (shrinkage).
- `modelBased/simple_model.py`: baseline com biases globais/usuário/item.
- `modelBased/regularized_svd.py`: SVD regularizado (SGD, fatores latentes, biases).
- `modelBased/bpr.py`: BPR (pairwise ranking) para feedback implícito.
- `modelBased/fm.py`: Factorization Machines (one-hot user/item features, extensível).