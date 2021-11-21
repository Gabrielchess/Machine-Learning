# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:12:07 2020

@author: Hacker James Bond
"""

import pandas as pd

dados = pd.read_csv('mercado.csv', header = None)
transacoes = []
for i in range(0, 10):
    transacoes.append([str(dados.values[i, j]) for j in range(0,4)])
    
from apyori import apriori
regras = apriori(transacoes, min_suppot = 0.3, min_confidence = 0.8, min_lift = 2, min_lenght = 2)

resultados = list(regras)

resultados2 = [list(x) for x in resultados]
resultados2
resultadosFormatado = []
for j in range(0, 3):
    resultadosFormatado.append([list(x) for x in resultados2[j][2]])
resultadosFormatado