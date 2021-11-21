#!/usr/bin/env python
# coding: utf-8

# In[99]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# In[100]:


base = pd.read_csv('/home/pasoneto/Documents/github/doc_suomi/data/treated_data/upsampled_binded.csv')
base = base.dropna()

nalbums_treino = 3000
nalbums_teste = (len(base)/16)-nalbums_treino

treino = np.array(base["0"][0:(nalbums_treino*16)]).reshape(1, -1)
treino = treino[0]
teste = np.array(base["0"][len(treino):]).reshape(1, -1)
teste = teste[0]
# Verificando

len(treino)+len(teste) == len(base['0'])


# In[101]:


# Sampling
previsores_treino = [treino[i:i+4] for i in range(len(treino)-4)]
previsores_teste = [teste[i:i+4] for i in range(len(teste)-4)]

real_treino = [treino[i] for i in range(4,len(treino))]
real_teste =  [teste[i]  for i in range(4,len(teste))]


# In[119]:


list(range(0, len(previsores_treino), 16))[-1]
previsores_treino[47984:47984+12]
previsores_treino[47984+13]


# In[120]:


############## TREINO
p_treino = []
for i in range(0, len(previsores_treino), 16):
    p_treino.append(previsores_treino[i:i+12])

p_treino = np.concatenate(p_treino).ravel().tolist()
p_treino = np.reshape(p_treino, (int(len(p_treino)/4), 4, 1))

############## TESTE
p_teste = []
for i in range(0, len(previsores_teste), 16):
    p_teste.append(previsores_teste[i:i+12])

p_teste = np.concatenate(p_teste).ravel().tolist()
p_teste = np.reshape(p_teste, (int(len(p_teste)/4), 4, 1))


############## REAL TREINO
r_treino = []
for i in range(0, len(real_treino), 16):
    r_treino.append(real_treino[i:i+12])

r_treino = np.concatenate(r_treino).ravel().tolist()
r_treino = np.reshape(r_treino, (len(r_treino), 1, 1))


# ############## REAL TESTE
r_teste = []
for i in range(0, len(real_teste), 16):
    r_teste.append(real_teste[i:i+12])

r_teste = np.concatenate(r_teste).ravel().tolist()
r_teste = np.reshape(r_teste, (len(r_teste), 1, 1))


# In[121]:


# Verificando
for i in range(10):
        print(previsores_treino[i], real_treino[i])


# In[122]:


# Definindo modelo
regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (p_treino.shape[1], 1)))
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


# In[ ]:


# Fitando modelo
regressor.fit(p_treino, r_treino, epochs = 100, batch_size = 32)


# In[ ]:


previsoes = regressor.predict(p_teste)


# In[ ]:


for i in range(0, 120, 12):
    real = np.concatenate(r_teste).ravel().tolist()[i:i+12]
    previsao = previsoes[i:i+12]
    baseline = [np.mean(real) for i in range(12)]
    plt.plot(real, color = 'red', label = 'Valencia real')
    plt.plot(previsao, color = 'blue', label = 'Previsões')
    plt.plot(baseline, 'r--', color = 'green', label = 'Baseline')
    plt.title('Previsão da Valência')
    plt.xlabel('Musicas')
    plt.ylabel('Valencia')
    print("RMSE model:"   , mean_squared_error(real, previsao))
    print("RMSE baseline:", mean_squared_error(real, baseline))
    plt.legend()
    plt.show()


# In[ ]:


mean_squared_error(np.concatenate(r_teste).ravel().tolist(), previsoes)


# In[ ]:


# serialize model to JSON
model_json = regressor.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
regressor.save_weights("model.h5")
print("Saved model to disk")


# # Loading model
# 
# Loading model's weights.

# In[74]:


# load json and create model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


# In[75]:


# evaluate loaded model on test data
loaded_model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', 
                  metrics = ['mean_absolute_error'])


# In[76]:


previsoes = loaded_model.predict(p_teste)


# In[43]:


rmse_model = []
rmse_baseline = []
for i in range(0, 120-12, 12):
    real = np.concatenate(r_teste).ravel().tolist()[i:i+12]
    previsao = previsoes[i:i+12]
    baseline = [np.mean(real) for i in range(12)]
#     plt.plot(real, color = 'red', label = 'Valencia real')
#     plt.plot(previsao, color = 'blue', label = 'Previsões')
#     plt.plot(baseline, 'r--', color = 'green', label = 'Baseline')
#     plt.title('Previsão da Valência')
#     plt.xlabel('Musicas')
#     plt.ylabel('Valencia')
#     print("RMSE model:"   , mean_squared_error(real, previsao))
#     print("RMSE baseline:", mean_squared_error(real, baseline))
#    plt.legend()
#    plt.show()
    rmse_model.append(mean_squared_error(real, previsao))
    rmse_baseline.append(mean_squared_error(real, baseline))
        


# In[67]:


import seaborn as sns

df = {'rmse_model': rmse_model, 'rmse_baseline': rmse_baseline}
df = pd.DataFrame(data=df)

# plot of 2 variables
p1=sns.kdeplot(df["rmse_model"], shade=True, color="r")
p1=sns.kdeplot(df["rmse_baseline"], shade=True, color="b")


# In[ ]:




