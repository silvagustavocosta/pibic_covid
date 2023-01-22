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
        45 amostras
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
        symptom = []
        for date in symptom_date:
            x = datetime.datetime.strptime(
                date, '%Y-%m-%d %H:%M:%S')
            symptom.append(x)
    else:
        symptom = None
    if len(covid_date) != 0:
        covid = []
        for date in covid_date:
            x = datetime.datetime.strptime(
                date, '%Y-%m-%d %H:%M:%S')
            covid.append(x)
    else:
        covid = None
    if len(recovery_date) != 0:
        recovery = []
        for date in recovery_date:
            x = datetime.datetime.strptime(
                date, '%Y-%m-%d %H:%M:%S')
            recovery.append(x)
    else:
        recovery = None

    return(symptom, covid, recovery)


def sort_datas(symptom_date, covid_date, recovery_date, pre_symptom_date):
    """
        Constroi a partir dos valores das datas de sintomas, covid e recueperação uma lista de 
        dicionários em ordem temporal. Os dicionários estão no formato data : situação do paciente
    """

    # criar a lista de dicionários de sintomas, covid, recuperação e datas pré-sintomáticas:
    dateList = []
    if pre_symptom_date:
        for data in pre_symptom_date:
            symptomDict = {}
            symptomDict["date"] = data
            symptomDict["status"] = 3
            dateList.append(symptomDict)
    if symptom_date:
        for data in symptom_date:
            symptomDict = {}
            symptomDict["date"] = data
            symptomDict["status"] = 1
            dateList.append(symptomDict)
    if covid_date:
        for data in covid_date:
            symptomDict = {}
            symptomDict["date"] = data
            symptomDict["status"] = 2
            dateList.append(symptomDict)
    if recovery_date:
        for data in recovery_date:
            symptomDict = {}
            symptomDict["date"] = data
            symptomDict["status"] = 0
            dateList.append(symptomDict)

    # sort the list according to the date
    dateList = sorted(dateList, key=lambda x: x['date'], reverse=False)

    return(dateList)


def covid_period(dateList):
    """
        Caso período entre doença e recuperação da pessoar for muito curto (menor que 5 dias), adicionamos mais
        7 dias na análise para termos resultados mais favoráveis
    """
    # TODO
    print(dateList)
    for count, data in enumerate(dateList):
        if count == 0:
            continue
        else:
            if data["status"] == 2 and dateList[count-1]["status"] == 1:
                if data["date"] - dateList[count-1]["date"] < datetime.timedelta(days=5):
                    # adicionamos mais 5 dias no péríodo de doença
                    data["date"] = data["date"] + datetime.timedelta(days=1)
                    dateList[count+1]["date"] = dateList[count +
                                                         1]["date"] + datetime.timedelta(days=5)
                else:
                    continue

    return(dateList)


def sick_min(df_sick, vetores, participant):
    """
        Relacionar os vetores com os períodos onde o participant está saudável (0), 
        com sintomas (1), doente (2)
    """

    symptom_date, covid_date, recovery_date = Pre_Processing.get_sick_time(
        df_sick, participant)

    symptom_date, covid_date, recovery_date = get_date(
        symptom_date, covid_date, recovery_date)

    # Período pré-sintomático: 14 dias antes do início das anomalias
    if symptom_date:
        pre_symptom_date = []
        pre_symptom_date.append(symptom_date[0] - datetime.timedelta(days=14))
    else:
        pre_symptom_date = None

    dateList = sort_datas(symptom_date, covid_date,
                          recovery_date, pre_symptom_date)

    # se o período entre doença e recuperação da pessoa for menor que 7 dias, adicionar mais 7 dias para esse período
    # dateList = covid_period(dateList)

    # sick_id vai carregar com 0, 1 e 2 se o vetor é de uma época saudável, com sintomas ou doente
    # os seguintes foor loops separam todos os vetores nessas categorias
    sick_id = []
    if len(dateList) != 0:
        for vetor in vetores:
            if vetor.index[0] < dateList[0]["date"]:
                sick_id.append(0)
    if len(dateList) != 0:
        for count, lDict in enumerate(dateList):
            if count < (len(dateList)-1):
                next_date = dateList[count + 1]["date"]
                for vetor in vetores:
                    if vetor.index[0] >= lDict["date"] and vetor.index[0] < next_date:
                        sick_id.append(lDict["status"])
            else:
                for vetor in vetores:
                    if vetor.index[0] >= lDict["date"]:
                        sick_id.append(lDict["status"])
    else:
        for vetor in vetores:
            sick_id.append(0)

    return sick_id, dateList


def sick_hour(df_sick, scRHR, participant):
    """
        Relaciona os períodos em que o participante esteve saudável (0), com sintomas (1) e 
        doente (2) com os dados de RHR pre_processados e analisados por hora (scRHR) 
    """

    symptom_date, covid_date, recovery_date = Pre_Processing.get_sick_time(
        df_sick, participant)

    symptom_date, covid_date, recovery_date = get_date(
        symptom_date, covid_date, recovery_date)

    # Período pré-sintomático: 14 dias antes do início das anomalias
    if symptom_date:
        pre_symptom_date = []
        pre_symptom_date.append(symptom_date[0] - datetime.timedelta(days=14))
    else:
        pre_symptom_date = None

    dateList = sort_datas(symptom_date, covid_date,
                          recovery_date, pre_symptom_date)

    # dateList = covid_period(dateList)

    # sick_HID vai carregar com 0, 1, 2  e 3 se o vetor é de uma época saudável, com sintomas ou doente
    # os seguintes foor loops separam todos as horas nessas categorias
    # TODO
    sick_HID = []
    if len(dateList) != 0:
        for idx in scRHR.index:
            if idx < dateList[0]["date"]:
                sick_HID.append(0)
    if len(dateList) != 0:
        for count, lDict in enumerate(dateList):
            if count < (len(dateList)-1):
                next_date = dateList[count + 1]["date"]
                for idx in scRHR.index:
                    if idx >= lDict["date"] and idx < next_date:
                        sick_HID.append(lDict["status"])
            else:
                for idx in scRHR.index:
                    if idx >= lDict["date"]:
                        sick_HID.append(lDict["status"])
    else:
        for idx in scRHR.index:
            sick_HID.append(0)

    return sick_HID


def isolation_forestHOUR(df):
    """
        Aplica o algorítimo de Isolation Forest no rhr do dataframe df (scRHR), adiciona
        coluna de anomalias no dataframe
    """

    model = IsolationForest(n_estimators=100, max_samples='auto',
                            contamination=float(0.02), max_features=1.0)

    model.fit(df[['heartrate']])

    df['scores'] = model.decision_function(df[['heartrate']])
    df['anomaly'] = model.predict(df[['heartrate']])

    return df


def plot_anomaly(df, symptom_date, covid_date, recovery_date, pre_symptom_date, title, save_mode, participant):
    """
        Traça os gráficos do rhr levando em consideração os tempos de doenças 
    """

    fig, ax = plt.subplots()
    plot_min = df['heartrate'].min()
    plot_max = df['heartrate'].max()

    # ax.scatter(df.index, df['heartrate'], label='rhr',
    #            marker='.', c=df['scores'], cmap='winter_r')

    ax.scatter(df.index, df['heartrate'], label='RHR', zorder=0, s=5)

    x = df.loc[df['anomaly'] == -1, 'heartrate']
    ax.scatter(x.index, x, c='r', marker='.', label='Anomaly', zorder=5)

    plt.gcf().set_size_inches(12, 10)

    if symptom_date:
        ax.vlines(x=symptom_date, ymin=plot_min, ymax=plot_max, color='y',
                  label='symptom date')
    if covid_date:
        ax.vlines(x=covid_date, ymin=plot_min, ymax=plot_max, color='r',
                  label='covid_date')
    if recovery_date:
        ax.vlines(x=recovery_date, ymin=plot_min, ymax=plot_max, color='g',
                  label='recovery_date')
    if pre_symptom_date:
        ax.vlines(x=pre_symptom_date, ymin=plot_min, ymax=plot_max, color='m',
                  label='pre_symptom date')

    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.title(title)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    if save_mode == "on":
        base_path = "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Data"
        figName = title + ".jpg"
        dir_path = os.path.join(base_path, participant, figName)
        plt.savefig(dir_path)

    plt.show()


def input_data(time_min_vetores, df_size):
    """
        Encontra missing values nos indíces dos vetores, adiciona índices que faltam 
        utilizando index. Interpolação dos valores que os índices representam.
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

        vetor = vetor.reindex(new_index)

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


def isolation_forestMin(df, contamination):
    """
        Aplica Isolation Forest no rhr do dataframe df (minutesRHR), adiciona
        coluna de anomalias no dataframe
    """

    # expand the x array to columns, getting an array of shape n_samples and n_features
    df_expanded = pd.DataFrame(df['heartrate'].tolist())

    model = IsolationForest(n_estimators=100, max_samples='auto',
                            contamination=float(contamination), max_features=1.0)

    model.fit(df_expanded)

    df['scores'] = model.decision_function(df_expanded)
    df['anomaly'] = model.predict(df_expanded)

    anomaly_count = len(df.loc[df['anomaly'] == -1])

    return df, anomaly_count


def number_of_inputs(df):
    """
        Carrega quantos dados vão ser inputados para que o dataframe fique completo, inputa os dados no dataframe
    """

    firstIndex = df.index[0]
    lastIndex = df.index[-1]

    dates = pd.date_range(firstIndex, lastIndex, freq="1min")
    totalLen = len(dates)
    dfLen = len(df)

    df = df.reindex(dates)

    longest_na_gaps, lengths_consecutive_na = can_be_inputed(df)

    # Last Carried Observation
    # df = df.fillna(method="ffill")

    # Inputing Base Data
    # df = df.fillna(60)

    # Interpolating data
    df = df.interpolate(method="linear")

    return df, totalLen, dfLen, lengths_consecutive_na


def zoomdf(df, dateInit, dateEnd):
    """
        Diminui o tamanho de um dataframe para conseguir plotar em uma imagem maior só o intervalo desejado
        entre a dateInit e a dateEnd
    """

    df = df[dateInit:dateEnd]
    return df


def can_be_inputed(df):
    """
        Detecta arquivos que podemos inputar dados e arquivos que não é viável inputar dados
    """

    na_groups = df["heartrate"].notna().cumsum()[df["heartrate"].isna()]
    lengths_consecutive_na = na_groups.groupby(na_groups).agg(len)
    longest_na_gap = lengths_consecutive_na.max()

    lengths_consecutive_na = lengths_consecutive_na.sort_values(
        ascending=False)

    # print(lengths_consecutive_na)

    return longest_na_gap, lengths_consecutive_na


def ploting(df, pre_symptom_date, symptom_date, covid_date, recovery_date, title, column, save_mode, participant):
    """
        Plotagem de gráficos
    """

    fig, ax = plt.subplots()
    plot_min = df['heartrate'].min()
    plot_max = df['heartrate'].max()

    ax.scatter(df.index,
               df[column], label="vetoresRHR", marker=".")

    if symptom_date:
        ax.vlines(x=symptom_date, ymin=plot_min, ymax=plot_max, color='y',
                  label='symptom date')
    if covid_date:
        ax.vlines(x=covid_date, ymin=plot_min, ymax=plot_max, color='r',
                  label='covid_date')
    if recovery_date:
        ax.vlines(x=recovery_date, ymin=plot_min, ymax=plot_max, color='g',
                  label='recovery_date')
    if pre_symptom_date:
        ax.vlines(x=pre_symptom_date, ymin=plot_min, ymax=plot_max, color='m',
                  label='pre_symptom date')

    plt.gcf().set_size_inches(12, 10)
    plt.title(title)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.legend()

    if save_mode == "on":
        base_path = "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Data"
        figName = title + ".jpg"
        dir_path = os.path.join(base_path, participant, figName)
        plt.savefig(dir_path)

    plt.show()
