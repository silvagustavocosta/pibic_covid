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


def get_contamination(start, end, spacement):
    """ 
        Cria lista que começa em float(start) e termina em float(end) com 
        espaçamento de cada valor de float(spacement)
    """

    start = float(start)
    end = float(end)
    spacement = float(spacement)

    contamination = []
    while True:
        contamination.append(start)
        start = start + spacement
        if start > end:
            break

    real_cont = []
    for numero in contamination:
        numero = round(numero, 3)
        real_cont.append(numero)

    return real_cont


def simple_plot(x, y, title, xlabel, ylabel):
    """
        Plota um gráfico simples a partir dos eixos x e y fornecidos
    """

    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gcf().set_size_inches(8, 6)
    plt.title(title)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()


def var_contamination(df, cont_para, sick_id):
    """
        Varia a contaminação conforme os parâmetros passados, faz as análises de isolation forest no código, 
        retorna os valores de anomalias totais, corretas e a porcentagem dessas anomalias
    """
    total_anomaly = []  # carrega numa lista quantas anomlias totais certa contaminação traz
    # carrega numa lsita quantas anomalias verdadeiras (em periodos de sintoma ou doença) certa contaminação traz
    true_anomaly = []
    # carrega numa lista a porcentagem de anomalias corretas de certa contaminaçã
    porc_anomaly = []

    for contamination in cont_para:
        df, n_anomaly = Anomaly_Detection.isolation_forestMin(
            df, contamination)

        time_min_mean = df.copy()
        time_min_mean['heartrate'] = time_min_mean['heartrate'].apply(np.mean)

        time_min_mean['sick_ID'] = sick_id

        # localiza o tamanho das respostas corretas de anomalia que o isolation forest achou
        # com esses resultados temos n de anomalias, n de anomalias corretas, n de anomalias erradas

        nAno = time_min_mean.loc[(time_min_mean["anomaly"] == -1)]
        nSic = time_min_mean.loc[(((time_min_mean["anomaly"] == -1) & (time_min_mean["sick_ID"] == 1)) |
                                  ((time_min_mean["anomaly"] == -1) & (time_min_mean["sick_ID"] == 2)) | ((time_min_mean["anomaly"] == -1) & (time_min_mean["sick_ID"] == 3)))]
        por = (len(nSic)/len(nAno))*100

        total_anomaly.append(len(nAno))
        true_anomaly.append(len(nSic))
        porc_anomaly.append(round(por, 4))

    return total_anomaly, true_anomaly, porc_anomaly


def df_stepDivision(df, symptom_date):
    """
        Separa o dataframe de steps somente em 21 dias antes do inicio dos sintomas e 7 dias após
        o início dos sintomas
    """

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index(['datetime'])

    if len(symptom_date) != 0:
        symptomLimits = {}
        limitsList = []
        for data in symptom_date:
            init = data - datetime.timedelta(days=21)
            end = data + datetime.timedelta(days=7)
            symptomLimits["init"] = init
            symptomLimits["end"] = end
            limitsList.append(symptomLimits)

        for limits in limitsList:
            df = df.loc[limits["init"]:limits["end"]]

    return df

# TODO


def stepPor(df):
    """
        Remove individuals with more than 50% of steps or sleep (each individually) data missing in a window of 21 d before symptom onset and 7 d after.
    """

    del df['user']
    resample_time = "1Min"
    df = df.resample(resample_time).mean()

    print(df)


# Remove individuals with more than 50% of steps or sleep (each individually) data missing in a window of 21 d before symptom onset and 7 d after:
mode = "solo"

Supplementary_Table = pd.read_csv(
    "/home/gustavo/PibicData1/Sick_Values_01.txt")
if mode == "solo":
    subjects = []
    subjects.append("AF3J1YC")
elif mode == "full":
    subjects = Supplementary_Table.ParticipantID.values.tolist()

df_sick = pd.read_csv("/home/gustavo/PibicData1/Sick_Values_01.txt")

for subject in subjects:
    participant = subject
    print(participant)

    # importar os arquivos
    hr_data = pd.read_csv(
        "/home/gustavo/PibicData1/COVID-19-Wearables/" + participant + "_hr.csv")
    steps_data = pd.read_csv(
        "/home/gustavo/PibicData1/COVID-19-Wearables/" + participant + "_steps.csv")

    symptom_date, covid_date, recovery_date = Pre_Processing.get_sick_time(
        df_sick, participant)

    symptom_date, covid_date, recovery_date = Anomaly_Detection.get_date(
        symptom_date, covid_date, recovery_date)

    df_steps = df_stepDivision(steps_data, symptom_date)

    step_percentage = stepPor(df_steps)
