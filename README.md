# Análise de Dados de Imóveis e Regressão - Preço de Casas em Londres

Este projeto visa realizar uma análise exploratória e desenvolver um modelo preditivo para o preço de imóveis em Londres. O processo inclui a análise dos dados, a visualização de distribuições e a criação de um modelo de regressão, com otimização de hiperparâmetros, para prever o preço das casas. As etapas envolvem desde o pré-processamento dos dados até a avaliação do modelo.

## Bibliotecas Utilizadas

- **Pandas**: Para manipulação e análise de dados.
- **Seaborn**: Para visualização de dados, como gráficos de dispersão, boxplots e histograma.
- **Matplotlib**: Para personalização e controle dos gráficos.
- **Plotly**: Para visualizações interativas.
- **Numpy**: Para manipulação de dados numéricos.
- **Scikit-learn**: Para a criação de modelos de machine learning e otimização de hiperparâmetros.
- **Joblib**: Para salvar o modelo treinado.

## Objetivos do Projeto

1. **Análise Exploratória de Dados (EDA)**:
   - Realizar a exploração inicial do dataset, verificando as colunas, dados ausentes e outliers.
   - Analisar as colunas numéricas com métricas estatísticas (média, desvio padrão, skew, etc.).
   - Verificar a distribuição e frequência das colunas categóricas.

2. **Visualizações**:
   - Visualizar a distribuição das variáveis numéricas e compará-las com variáveis categóricas, usando gráficos como boxplots, scatter plots e histograma.
   - Explorar a relação entre o preço e características dos imóveis como tamanho, número de quartos, etc.

3. **Modelo de Regressão**:
   - Criar um modelo preditivo utilizando `GradientBoostingRegressor` para prever os preços das casas.
   - Utilizar técnicas de otimização de hiperparâmetros para encontrar os melhores parâmetros para o modelo.

4. **Avaliação do Modelo**:
   - Avaliar o desempenho do modelo com métricas de regressão como MSE, MAE, R² e RMSE.
   - Aplicar validação cruzada para garantir a generalização do modelo.

---

## Etapas do Projeto

### 1. **Carregamento e Preparação dos Dados**
Projeto foi iniciado com o download de um dataset do site kaggle e a transformação do mesmo, traduzindo-o para pt-br e baixando-o localmente.

### 2. **Análise Exploratória de Dados (EDA)**
Durante a análise exploratória, as colunas numéricas foram analisadas com métricas estatísticas, incluindo:
    - Média, valor mínimo, valor máximo
    - Desvio padrão
    - Skew (assimetria) e kurtosis (curtose)
    - Teste de normalidade (Shapiro-Wilk)

Além disso, as colunas categóricas foram analisadas para verificar a quantidade de valores únicos e a frequência acumulada.

### 3. **Modelo de Regressão com Otimização de Hiperparâmetros**
Para prever o preço das casas, foi criado um modelo de regressão utilizando o GradientBoostingRegressor. A otimização dos hiperparâmetros foi realizada utilizando GridSearchCV para encontrar os melhores valores para:
    - Número de estimadores (n_estimators)
    - Taxa de aprendizado (learning_rate)
    - Profundidade máxima das árvores (max_depth)

Pipeline de Pré-processamento e Modelo
O MaxAbsScaler foi utilizado para normalizar as variáveis numéricas antes de alimentar o modelo. O Pipeline foi utilizado para garantir uma execução eficiente de todo o processo.

### 4. **Avaliação do Modelo**
O modelo foi avaliado utilizando as seguintes métricas de regressão:
    - MSE (Erro Quadrático Médio)
    - MAE (Erro Absoluto Médio)
    - R² (Coeficiente de Determinação)
    - RMSE (Raiz do Erro Quadrático Médio)

Além disso, foi realizada validação cruzada para avaliar a generalização do modelo.

### 5. **Salvamento do Modelo**
Após a otimização, o modelo final foi salvo utilizando o joblib, permitindo sua reutilização em novos dados.

### 6. **Visualização de Dados**
Como parte da análise exploratória de dados, foram criados gráficos para entender melhor as correlações entre as variáveis e explorar a distribuição dos dados.

**Heatmap de Correlação**
Foi gerado um **heatmap** para analisar a correlação entre as variáveis numéricas. A partir dessa visualização, foi possível identificar que a variável alvo, **Preço**, tem uma **alta correlação positiva** com as seguintes colunas:
- **Bairro**
- **Metros Quadrados**
- **Tipo do Imóvel**

O heatmap ajudou a entender como essas variáveis se comportam em relação ao preço, indicando quais colunas podem ter maior impacto na previsão do preço dos imóveis.

**Distribuição das Variáveis**:
Utilizando a biblioteca Plotly, foram criados gráficos interativos para visualizar a distribuição das colunas numéricas e categóricas. Isso facilitou a análise visual das distribuições e ajudou a identificar padrões importantes nos dados.

**Gráficos de Relações entre Colunas com Alta Correlação**:
Além do heatmap, outros gráficos foram criados para representar as relações de alta correlação entre as variáveis. Esses gráficos permitiram uma melhor compreensão de como as variáveis se comportam juntas, ajudando a identificar quais características são mais relevantes para prever o preço dos imóveis. Essas visualizações foram fundamentais para tomar decisões durante a modelagem e análise dos dados, além de fornecer insights valiosos sobre as características que influenciam o preço dos imóveis.
