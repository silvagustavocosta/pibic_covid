from cmath import nan
from distutils.command.build_scripts import first_line_re
from re import X
import string
import time
from tracemalloc import start
from turtle import clear
from matplotlib import markers
from matplotlib.lines import lineStyles
import numpy as np
from sklearn.preprocessing import StandardScaler
import csv
import json
import pandas as pd
import glob
import os
from os import listdir, system
from os.path import isfile, join
from os.path import exists
import matplotlib
import matplotlib.pyplot as plt
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import Pre_Processing
from sklearn.ensemble import IsolationForest
import Anomaly_Detection


def day_time_separation(df, size):
    """
        Construção e separação do dataframe em diferentes vetores que vão possuir origin diferentes (com intervalos de 1 dia)
        size = tamanho de cada um dos vetores (7, 14, 21 ou 28)
    """

    init_date = df.index[0]
    final_date = df.index[-1]

    size = datetime.timedelta(days=size)
    separation = datetime.timedelta(days=1)

    # criar lista com os índices de todos as datas de início e fim dos vetores
    initIndexes = []
    endIndexes = []
    while (init_date + size) < final_date:
        initIndexes.append(init_date)
        end = init_date + size
        endIndexes.append(end)
        init_date = init_date + separation

    # associar os valores de initIndexes e endIndexes aos valores mais próximos do índice do dataframe:
    listIndex = df.index.tolist()
    teste = initIndexes[0]

    # idx = df.index[df.index.get_loc(teste, method='nearest')]

    initIndexes = df.index[df.index.get_indexer(initIndexes, method="nearest")]
    endIndexes = df.index[df.index.get_indexer(endIndexes, method="nearest")]

    vetores = []
    for i in range(len(initIndexes)):
        vetor = df[initIndexes[i]:endIndexes[i]]
        vetores.append(vetor)

    return vetores


def visualizationVetores(vetores, init, end):
    """
        Printa e visualiza o que nós encontramos dentro dos vetores, é passado a lista dos vetores e o intervalo de vetores que 
        queremos visualizar. 
    """

    while init < end:
        Pre_Processing.plot(vetores[init], 1, "line")
        print(len(vetores[init]))
        init += 1


def plotAnomalyVetores(vetores, init, end):

    while init < end:
        df = vetores[init]

        fig, ax = plt.subplots()

        # ax.scatter(df.index, df['heartrate'], label='rhr',
        #            marker='.', c=df['scores'], cmap='winter_r')

        ax.scatter(df.index, df['heartrate'], label='RHR', zorder=0, s=5)

        x = df.loc[df['anomaly'] == -1, 'heartrate']
        ax.scatter(x.index, x, c='r', marker='.', label='Anomaly', zorder=5)

        plt.gcf().set_size_inches(12, 10)

        ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

        plt.gcf().autofmt_xdate()
        plt.tight_layout()

        plt.show()

        init += 1


def plotFullAnalysis(origDF, anomalyDF):
    """
        Plotar o gráfico de número de anomalias (dos vetores) VS os dias da amostra
        No próprio gráfico levar em consideração a qualidade do vetor para o dia (utilizar um esquema de cores), 
        um vetor (representado pelo dia) que tem uma boa qualidade 
        Plotar também as datas que determinam o período pré_sintomatico, sintomas, covid e recuperação
    """

    fig, axs = plt.subplots(2)

    axs[0].plot(origDF.index, origDF["heartrate"], label='rhr', marker='.')
    axs[1].scatter(anomalyDF.index,
                   anomalyDF["countAnomaly"], label='Anomaly Count', color="r")

    plt.gcf().set_size_inches(14, 10)

    plt.title("Número de Anomalias vs Dias de amostra")
    plt.gcf().autofmt_xdate()
    axs[0].legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
