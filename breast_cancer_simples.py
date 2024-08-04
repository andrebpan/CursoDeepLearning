import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense


previsores = pd.read_csv('datasets/entradas_breast.csv')
classe = pd.read_csv('datasets/saidas_breast.csv')

print(previsores.head())
print(classe.head())

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

print("\nDistribuição dos dados apos o hold-out: ")

print(previsores_treinamento.shape)
print(previsores_teste.shape)
print(classe_treinamento.shape)
print(classe_teste.shape)

clf = Sequential()
#qtdeOculta = (30 + 1) / 2
clf.add(Dense(units = 16, activation = 'relu'))