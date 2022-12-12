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
