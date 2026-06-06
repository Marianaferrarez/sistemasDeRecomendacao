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

1. Instale as dependências (ver `requirements.txt`).
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