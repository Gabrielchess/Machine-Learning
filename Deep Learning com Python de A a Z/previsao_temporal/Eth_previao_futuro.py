#%%
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

######### Lendo dados
base = pd.read_csv('eth_price.csv')
base = base.dropna()

######### Pegando preços de entrada
base = base['open'].values.reshape(-1, 1)

######### Normalizando
normalizador = MinMaxScaler(feature_range = (0, 1))
base = normalizador.fit_transform(base)

#%%

# Recebe array, sample size e tamanho do output
# Retorna 1) lista de listas com tamanho do sample_size
# e lista de listas com tamanho do output 
# lista não ultrapassa o tamanho do array dado

def sample_maker(x, sample_size, out):
    if (sample_size+out) > len(x):
        raise Exception("Tamanhos incompatíveis. Ajuste o input e/ou o output")
    entrada = np.array([x[i:i+sample_size] for i in range(len(x)-(sample_size+out-1))]) 
    saida   = np.array([x[i+sample_size:i+sample_size+out] for i in range(len(x)-(sample_size+out-1))])
    return (entrada, saida)

# Teste: 
#sample = 4, output = 1

oi, tchau = sample_maker([5, 4, 3, 66, 3, 2, 3, 6, 3, 5, 7, 5, 4, 2, 1, 8], 4, 1)
len(oi) == len(tchau)

#sample=1, output =10
oi, tchau = sample_maker([5, 4, 3, 66, 3, 2, 3, 6, 3, 5, 7, 5, 4, 2, 1, 8], 10, 2)








#%%
######## Treino e teste
tamanho_treino = int(0.8*len(base)) # oitenta pcento dos dados
entrada_treino = base[0:tamanho_treino]
entrada_teste  = base[tamanho_treino:] # o resto

######## Verificando
len(entrada_teste)+len(entrada_treino) == len(base) # OK!








#%%


############## Separando em samples
entrada_treino, saida_treino = sample_maker(entrada_treino, sample_size=45, out=10)
entrada_teste, saida_teste = sample_maker(entrada_teste, sample_size=45, out=10)


#%%
entrada_treino.shape






#%%
regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (entrada_treino.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = len(saida_treino[0]), activation = 'linear'))

regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', 
                  metrics = ['mean_absolute_error'])

es = EarlyStopping(monitor = 'loss', min_delta = 1e-10, patience = 10, verbose = 1)
rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 5, verbose = 1)
mcp = ModelCheckpoint(filepath = 'pesos.h5', monitor = 'loss', 
                      save_best_only = True, verbose = 1)



#%%


regressor.fit(entrada_treino, saida_treino, epochs = 50, batch_size = 32,
              callbacks = [es, rlr, mcp])



#%%
prev = regressor.predict(entrada_teste)


real = [float(k) for l in saida_teste for k in l]
prev = [float(k) for l in prev for k in l]


# previsoes.mean()

plt.plot(prev, color = 'blue', label = 'Previsões')
plt.plot(real, color = 'red', label = 'real')
plt.title('Previsão preço')
plt.xlabel('Dia')
plt.ylabel('Valor Eth')
plt.legend()
plt.show()

# %%
