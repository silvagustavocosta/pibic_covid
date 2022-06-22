import string
from tarfile import ExFileObject
from tkinter import Y
from turtle import clear
import numpy as np
import csv
import pandas as pd
import glob
import os
from os import listdir, system
from os.path import isfile, join
from os.path import exists


# função que separa quais são as pessoas que estão em determinado documento
def get_names(list_of_documents):
    list_of_names = []
    for document_name in list_of_documents:
        temp_name = document_name.rsplit("_")
        list_of_names.append(temp_name[0])
    # retira nomes duplicados da lista de nomes
    list_of_names = list(set(list_of_names))
    return(list_of_names)


# pega um pedaço do path e retorna o path inteiro para o documento
def get_path(part_of_path, termination):
    path_geral = "/home/gustavo/PibicData1/COVID-19-Wearables/"
    if termination == "hr":
        path = (path_geral + part_of_path + "_hr.csv")
    elif termination == "steps":
        path = (path_geral + part_of_path + "_steps.csv")
    elif termination == "hrlong":
        path = (path_geral + part_of_path + "_hr_longterm.csv")
    elif termination == "stepslong":
        path = (path_geral + part_of_path + "_steps_longterm.csv")
    return path


def outliers(df):  # retira os outliers do hr (heart rate > 200 e heart rate < 30). Duplicates in hr and steps são removidos
    print(df["heartrate"])


# consegue o nome dos arquivos dentro do diretório
path_to_data = "/home/gustavo/PibicData1/COVID-19-Wearables/"
list_of_documents = [f for f in listdir(
    path_to_data) if isfile(join(path_to_data, f))]

test_names = ["A0KX894"]
dataframes_test = {}

# ler os arquivos e armazena no dataframe_test que estamos utilizando:
for i in range(len(test_names)):
    if exists(get_path(test_names[i], "hr")):  # confere que o path existe
        temp_df_hr = pd.read_csv(get_path(test_names[i], "hr"))
    else:
        temp_df_hr = pd.DataFrame()
    if exists(get_path(test_names[i], "steps")):
        temp_df_steps = pd.read_csv(get_path(test_names[i], "steps"))
    else:
        temp_df_steps = pd.DataFrame()
    if exists(get_path(test_names[i], "hrlong")):
        temp_df_hr_longterm = pd.read_csv(get_path(test_names[i], "hrlong"))
    else:
        temp_df_hr_longterm = pd.DataFrame()
    if exists(get_path(test_names[i], "stepslong")):
        temp_df_steps_longterm = pd.read_csv(
            get_path(test_names[i], "stepslong"))
    else:
        temp_df_steps_longterm = pd.DataFrame()
    frames = [temp_df_hr, temp_df_steps,
              temp_df_hr_longterm, temp_df_steps_longterm]
    temp_concat = pd.concat(frames)
    dataframes_test[test_names[i]] = temp_concat


x = dataframes_test.values()  # pega os valores guardados no dicionário
# dessa forma eu posso caminhar pelos arquivos, transforma um dictvalues em uma lista
x = list(x)
df = x[0]

# pre-processing:
outliers(df)
