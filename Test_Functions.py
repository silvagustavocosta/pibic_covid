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
