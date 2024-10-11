# Regressão Logística - Previsão de Diabetes

Este projeto aplica a técnica de **Regressão Logística** para prever a presença de diabetes em pacientes com base em dados médicos, como níveis de glicose, pressão sanguínea, índice de massa corporal (IMC), entre outros.

## Dataset

O dataset utilizado está disponível no [Kaggle](https://www.kaggle.com/datasets/saurabh00007/diabetescsv) e contém informações de 768 pacientes, incluindo 8 características principais e a variável alvo que indica se o paciente é diabético ou não.

## Bibliotecas Utilizadas

As seguintes bibliotecas foram utilizadas no projeto:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
```

## Etapas do Projeto
### * 1. Importação dos Dados:
Carregamos os dados do dataset em um DataFrame do pandas e realizamos uma análise exploratória para entender melhor as características dos dados.
```python
df = pd.read_csv("Caminho_do_arquivo/diabetes.csv")
df.head()
```
### * 2. Pré-processamento:
A variável Outcome foi mapeada para valores categóricos ('Diabético' e 'Não Diabético') e, em seguida, transformada em binário para facilitar o treinamento do modelo.
```python
df['Outcome'] = df['Outcome'].map({1: 'Diabético', 0: 'Não Diabético'})
dfCat = pd.get_dummies(df['Outcome'].values)
df['DiseaseCat'] = dfCat['Diabético'].values
```
### * 3. Divisão dos Dados:
Os dados foram divididos em variáveis independentes (X) e dependente (Y), e posteriormente em conjuntos de treino e teste.
```python
X = df.drop('DiseaseCat', axis=1)
Y = df['DiseaseCat']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)
```
### * 4. Escalonamento:
Utilizamos a técnica de escalonamento dos dados com StandardScaler para padronizar as características e reduzir possíveis erros devido à escala.
```python
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
### * 5. Ajustes de Hiperparâmetros:
Utilizamos validação cruzada para encontrar os melhores hiperparâmetros do modelo de Regressão Logística.
```python
params = {"penalty": ["l1", "l2"], "C": [0.1, 0.5, 1, 10]}
grid_search = GridSearchCV(LogisticRegression(), params, cv=10)
grid_search.fit(X_train, Y_train)
```
### * 6. Treinamento e Avaliação:
Treinamos o modelo com os melhores hiperparâmetros encontrados e avaliamos sua performance usando uma matriz de confusão e relatórios de classificação.
```python
instance = LogisticRegression(penalty='l2', C=0.9)
instance.fit(X_train, Y_train)
predict = instance.predict(X_test)
```
### A precisão do modelo foi de aproximadamente 74%.
```python
print(classification_report(Y_test, predict))
print(confusion_matrix(Y_test, predict))
```
## Resultados:
O modelo alcançou uma precisão de 74%, com a seguinte matriz de confusão:
```txt
[[86 15]
 [25 28]]
```
## Visualizações:
Gráficos foram gerados para melhor compreensão dos dados e da performance do modelo, como gráficos de pares (pairplot) e uma matriz de correlação (heatmap). Eles podem ser acessados no notebook completo.
## Conclusão
A regressão logística foi eficaz para prever a presença de diabetes, embora ajustes adicionais e o uso de técnicas mais avançadas possam melhorar a performance do modelo.
## Como Executar:
1. Clone este repositório:
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
```
2. Instale as dependências:
```bash
pip install -r requirements.txt
```
3. Execute o notebook no Jupyter:
```bash
jupyter notebook Logistic_Regression_Diabetes.ipynb
```
