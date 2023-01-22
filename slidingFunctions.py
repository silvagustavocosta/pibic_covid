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


def plotFullAnalysis(origDF, anomalyDF, pre_symptom_date, symptom_date, covid_date, recovery_date):
    """
        Plotar o gráfico de número de anomalias (dos vetores) VS os dias da amostra
        No próprio gráfico levar em consideração a qualidade do vetor para o dia (utilizar um esquema de cores), 
        um vetor (representado pelo dia) que tem uma boa qualidade 
        Plotar também as datas que determinam o período pré_sintomatico, sintomas, covid e recuperação
    """

    fig, axs = plt.subplots(2)
    plot_min0 = origDF['heartrate'].min()
    plot_max0 = origDF['heartrate'].max()
    plot_min1 = anomalyDF['countAnomaly'].min()
    plot_max1 = anomalyDF['countAnomaly'].max()

    axs[0].plot(origDF.index, origDF["heartrate"], label='rhr', marker='.')
    axs[1].scatter(anomalyDF.index,
                   anomalyDF["countAnomaly"], label='Anomaly Count', color="r")

    plt.gcf().set_size_inches(14, 10)

    if symptom_date:
        axs[0].vlines(x=symptom_date, ymin=plot_min0, ymax=plot_max0, color='y',
                      label='symptom date')
        axs[1].vlines(x=symptom_date, ymin=plot_min1, ymax=plot_max1, color='y',
                      label='symptom date')
    if covid_date:
        axs[0].vlines(x=covid_date, ymin=plot_min0, ymax=plot_max0, color='r',
                      label='covid_date')
        axs[1].vlines(x=covid_date, ymin=plot_min1, ymax=plot_max1, color='r',
                      label='covid_date')
    if recovery_date:
        axs[0].vlines(x=recovery_date, ymin=plot_min0, ymax=plot_max0, color='g',
                      label='recovery_date')
        axs[1].vlines(x=recovery_date, ymin=plot_min1, ymax=plot_max1, color='g',
                      label='recovery_date')
    if pre_symptom_date:
        axs[0].vlines(x=pre_symptom_date, ymin=plot_min0, ymax=plot_max0, color='m',
                      label='pre_symptom date')
        axs[1].vlines(x=pre_symptom_date, ymin=plot_min1, ymax=plot_max1, color='m',
                      label='pre_symptom date')

    plt.title("Número de Anomalias vs Dias de amostra")
    plt.gcf().autofmt_xdate()
    axs[0].legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def qualityHR(hr_data):
    """
        Qualidade da hora analisada, quantas amostras no HR (dados não processados), 
        definir porcentagem (quantidade de minutos com dados/60)*100
    """

    hr_data = hr_data.set_index("datetime")
    hr_data.index.name = None
    hr_data.index = pd.to_datetime(hr_data.index)

    # resampling the data to 1min, deixa o dataframe completo com valores nulos = nan
    hr_data = hr_data.resample("1min").mean()
    hr_data["heartrate"] = hr_data["heartrate"].fillna(0)

    # associar o minuto que não possui dados, ["heartrate"] == 0, com valor de 0. Minuto que possui dado com valor de 1
    hr_data["Qmin_HR"] = np.where(hr_data["heartrate"] == 0, 0, 1)

    # para cada hora de análise, realizar a operação: (somatório de Qmin_HR/60)*100
    hr_dataHour = pd.DataFrame()
    hr_dataHour = hr_data.resample("1H").sum()
    hr_dataHour["Qmin_HR"] = (hr_dataHour["Qmin_HR"]/60)*100
    hr_dataHour = hr_dataHour.drop(columns=["heartrate"])

    # qualidade da amostra para um dia
    hr_dataDay = pd.DataFrame()
    hr_dataDay = hr_data.resample("1D").sum()
    hr_dataDay["Qday_HR"] = (hr_dataDay["Qmin_HR"]/1440)*100
    hr_dataDay = hr_dataDay.drop(columns=["heartrate", "Qmin_HR"])

    # qualidade de amostras para a semana
    hr_dataWeek = pd.DataFrame()
    hr_dataWeek = hr_data.resample("1W").sum()
    hr_dataWeek["Qweek_HR"] = (hr_dataWeek["Qmin_HR"]/10080)*100
    hr_dataWeek = hr_dataWeek.drop(columns=["heartrate", "Qmin_HR"])

    # ploting quality data for HR
    # Pre_Processing.plot(hr_dataHour, 1, "scatter")
    # Pre_Processing.plot(hr_dataDay, 1, "scatter")
    # Pre_Processing.plot(hr_dataWeek, 1, "scatter")

    return hr_dataHour, hr_dataDay, hr_dataWeek
