# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 10:14:50 2021

@author: James Bond
"""

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

base = pd.read_csv('upsampled_binded.csv')
base = base.dropna()

base = pd.DataFrame(base.iloc[92192:93792,:].values)

base_treinamento = base.iloc[:,1:2].values

normalizador = MinMaxScaler(feature_range = (0, 1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

previsores = []
preco_real = []
for i in range(4, 1600):
    previsores.append(base_treinamento_normalizada[i-4:i, 0])
    preco_real.append(base_treinamento_normalizada[i, 0])
previsores, preco_real = np.array(previsores), np.array(preco_real)
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

regressor = Sequential()
#input_shape = dados de entrada, 1 pois só tem 1 classe de previsor
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (previsores.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 20, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 20, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 20))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1, activation = 'linear'))

regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', 
                  metrics = ['mean_absolute_error'])

es = EarlyStopping(monitor = 'loss', min_delta = 1e-10, patience = 10, verbose = 1)
rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 5, verbose = 1)
mcp = ModelCheckpoint(filepath = 'pesos.h5', monitor = 'loss', 
                      save_best_only = True, verbose = 1)
regressor.fit(previsores, preco_real, epochs = 50, batch_size = 32)

base_teste = base.tail(160)
valencia_real_teste = base_teste.iloc[:, 1:2].values

base_treinamento2 = pd.DataFrame(base.iloc[92192:93792, 1:2].values)

base_completa = pd.concat((base_treinamento2[0], base_teste['0']), axis = 0)
entradas = base_completa[len(base_completa) - len(base_teste) - 4:].values
entradas = entradas.reshape(-1,1)
entradas = normalizador.transform(entradas)

X_teste = []
for i in range(4, 164):
    X_teste.append(entradas[i-4:i, 0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))
previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)

base_teste.columns
base_teste = base.head(160)
base_teste = normalizador.transform(base_teste)
base_teste = base_teste[:,0]

base_teste = list(base_teste)

t = []
for i in range(len(base_teste)-4):
    t.append(base_teste[i:i+4])

real = []
for i in range(5,len(base_teste)):
    real.append(base_teste[i])
oi = np.array([t])
oi = np.reshape(t, (156, 4, 1))
previsoes = regressor.predict(oi)

previsoes.mean()
valencia_real_teste.mean()

plt.plot(valencia_real_teste, color = 'red', label = 'Preço real')
plt.plot(previsoes, color = 'blue', label = 'Previsões')
plt.title('Previsão da Valência')
plt.xlabel('Musicas')
plt.ylabel('Valencia')
plt.legend()
plt.show()
