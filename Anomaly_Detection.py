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


def time_separation(df):
    # dividindo o dataframe em diferentes listas de vetores que começam em lugares distintos
    vetor0 = [g for n, g in df.groupby(pd.Grouper(
        freq='H', origin='start'))]
    vetor1 = [g for n, g in df.groupby(pd.Grouper(
        freq='H', origin='start', offset=datetime.timedelta(minutes=10)))]
    vetor2 = [g for n, g in df.groupby(pd.Grouper(
        freq='H', origin='start', offset=datetime.timedelta(minutes=20)))]
    vetor3 = [g for n, g in df.groupby(pd.Grouper(
        freq='H', origin='start', offset=datetime.timedelta(minutes=30)))]
    vetor4 = [g for n, g in df.groupby(pd.Grouper(
        freq='H', origin='start', offset=datetime.timedelta(minutes=40)))]
    vetor5 = [g for n, g in df.groupby(pd.Grouper(
        freq='H', origin='start', offset=datetime.timedelta(minutes=50)))]

    vetores = []
    # concatenar os 6 vetores para formar lista de vetores completa
    for i, vetor in enumerate(vetor0):
        if i == 0:
            continue
        else:
            vetores.append(vetor1[i])
            vetores.append(vetor2[i])
            vetores.append(vetor3[i])
            vetores.append(vetor4[i])
            vetores.append(vetor5[i])
            vetores.append(vetor0[i])

    return vetores


def quality(vetores):
    """
        Conta quantas amostras cada vetor possui, apaga vetores com menos de 
        40 amostras
    """
    qualidade = []

    for index, vetor in sorted(enumerate(vetores), reverse=True):
        if len(vetor) < 45:
            del vetores[index]
        else:
            qualidade.insert(0, len(vetor))

    return vetores, qualidade


def get_date(symptom_date, covid_date, recovery_date):
    """
        Separa symptom_date, covid_date, recovery_date no formato de datas, se alguma desses valores não 
        existirem passa uma data vazia
    """

    if len(symptom_date) != 0:
        symptom_date = datetime.datetime.strptime(
            symptom_date[0], '%Y-%m-%d %H:%M:%S')
    else:
        symptom_date = None
    if len(covid_date) != 0:
        covid_date = datetime.datetime.strptime(
            covid_date[0], '%Y-%m-%d %H:%M:%S')
    else:
        covid_date = None
    if len(recovery_date) != 0:
        recovery_date = datetime.datetime.strptime(
            recovery_date[0], '%Y-%m-%d %H:%M:%S')
    else:
        recovery_date = None

    return(symptom_date, covid_date, recovery_date)


def sick_min(df_sick, vetores, participant):
    """
        Relacionar os vetores com os períodos onde o participant está saudável (0), 
        com sintomas (1), doente (2)
    """

    symptom_date, covid_date, recovery_date = Pre_Processing.get_sick_time(
        df_sick, participant)

    symptom_date, covid_date, recovery_date = get_date(
        symptom_date, covid_date, recovery_date)

    # TODO Revisar esse for loop, nem sempre as datas estão separadas corretamente dessa forma
    # montar lista das datas
    # for data in lista_datas:
    #     for vetor em vetores:
    #         if vetor.index[0] > data:
    #             vetor segue a característica dessa data
    #         elif vetor.index[atual] > data.proxima:
    #             break

    sick_id = []
    for vetor in vetores:
        if vetor.index[-1] >= symptom_date and vetor.index[-1] <= covid_date:
            sick_id.append(1)
        elif vetor.index[-1] >= covid_date and vetor.index[-1] <= recovery_date:
            sick_id.append(2)
        else:
            sick_id.append(0)

    return sick_id


def sick_hour(df_sick, scRHR, participant):
    """
        Relaciona os períodos em que o participante esteve saudável (0), com sintomas (1) e 
        doente (2) com os dados de RHR pre_processados e analisados por hora (scRHR) 
    """

    symptom_date, covid_date, recovery_date = Pre_Processing.get_sick_time(
        df_sick, participant)

    symptom_date, covid_date, recovery_date = get_date(
        symptom_date, covid_date, recovery_date)

    symptomI = scRHR.index.get_loc(symptom_date, method='nearest')
    covidI = scRHR.index.get_loc(covid_date, method='nearest')
    recoveryI = scRHR.index.get_loc(recovery_date, method='nearest')

    scRHR = scRHR.reset_index()
    scRHR = scRHR.rename(columns={'index': 'datetime'})

    scRHR.loc[:symptomI, 'sickID'] = 0
    scRHR.loc[symptomI:covidI, 'sickID'] = 1
    scRHR.loc[covidI:recoveryI, 'sickID'] = 2
    scRHR.loc[recoveryI:, 'sickID'] = 0

    scRHR = scRHR.set_index('datetime')
    scRHR.index.name = None

    return scRHR


def isolation_forestHOUR(df):
    """
        Aplica o algorítimo de Isolation Forest no rhr do dataframe df (scRHR), adiciona
        coluna de anomalias no dataframe
    """

    model = IsolationForest(n_estimators=100, max_samples='auto',
                            contamination=float(0.08), max_features=1.0)

    model.fit(df[['heartrate']])

    df['scores'] = model.decision_function(df[['heartrate']])
    df['anomaly'] = model.predict(df[['heartrate']])

    return df


def plot_anomaly(df, symptom_date, covid_date, recovery_date, title):
    """
        Traça os gráficos do rhr levando em consideração os tempos de doenças e 
    """

    fig, ax = plt.subplots()
    plot_min = df['heartrate'].min()
    plot_max = df['heartrate'].max()

    # ax.scatter(df.index, df['heartrate'], label='rhr',
    #            marker='.', c=df['scores'], cmap='winter_r')

    ax.plot(df.index, df['heartrate'], label='RHR', zorder=0)
    x = df.loc[df['anomaly'] == -1, 'heartrate']

    ax.scatter(x.index, x, c='r', marker='.', label='Anomaly', zorder=5)

    plt.gcf().set_size_inches(8, 6)

    ax.vlines(x=symptom_date, ymin=plot_min, ymax=plot_max, color='y',
              label='symptom date')
    ax.vlines(x=covid_date, ymin=plot_min, ymax=plot_max, color='r',
              label='covid_date')
    ax.vlines(x=recovery_date, ymin=plot_min, ymax=plot_max, color='g',
              label='recovery_date')

    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.title(title)

    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.show()


def input_data(time_min_vetores, df_size):
    """
        Encontra missing values nos indíces dos vetores, adiciona índices que faltam 
        utilizando index. Interpolação
    """

    minute = datetime.timedelta(minutes=1)
    time_min_inp = []

    for vetor in time_min_vetores:
        first_index = vetor.index[0]

        # construir novo index:
        new_index = []
        for i in range(df_size):
            new_index.append(first_index)
            first_index = first_index + minute

        # mean = vetor.mean()
        vetor = vetor.reindex(new_index)
        # vetor = vetor.fillna(value=mean)
        vetor = vetor.interpolate(method='linear')
        time_min_inp.append(vetor)

    return(time_min_inp)


def organize_data(vetores):
    """
        Organiza todos os vetores de time_min_vetores em um único dataframe, indíces são os primeiros
        horários de cada vetores, os dados são arrays dos valores
    """

    df_vet = pd.DataFrame()
    listIndex = []
    listValues = []

    for vetor in vetores:
        listIndex.append(vetor.index[0])
        values = vetor['heartrate'].to_numpy()
        listValues.append(values)

    df_vet['index'] = listIndex
    df_vet = df_vet.set_index('index')
    df_vet.index.name = None

    df_vet['heartrate'] = listValues

    return df_vet


def isolation_forestMin(df):
    """
        Aplica Isolation Forest no rhr do dataframe df (minutesRHR), adiciona
        coluna de anomalias no dataframe
    """

    # expand the x array to columns, getting an array of shape n_samples and n_features
    # TODO expand the x array to columns, each column is the n-date of the n-element on the arrray
    df_expanded = pd.DataFrame(df['heartrate'].tolist())

    model = IsolationForest(n_estimators=100, max_samples='auto',
                            contamination=float(0.11), max_features=1.0)

    model.fit(df_expanded)

    df['scores'] = model.decision_function(df_expanded)
    df['anomaly'] = model.predict(df_expanded)

    return df
