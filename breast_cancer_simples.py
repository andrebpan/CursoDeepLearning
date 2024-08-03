import pandas as pd

previsores = pd.read_csv('datasets/entradas_breast.csv')
classe = pd.read_csv('datasets/saidas_breast.csv')

print(previsores.head())
print(classe.head())

from sklearn.model_selection import train_test_split

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

print("\nDistribuição dos dados apos o hold-out: ")

print(previsores_treinamento.shape)
print(previsores_teste.shape)
print(classe_treinamento.shape)
print(classe_teste.shape)

