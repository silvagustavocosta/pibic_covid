import string
from turtle import clear
import numpy as np
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
from datetime import datetime

def hr_outliers(df): #retira outliers da base de dados de hr. Outliers são hr < 30, hr > 200
    indice = (df.loc[df["heartrate"] > 200]).index #acha o indice dos outliers 
    for i in indice:
        df = df.drop(labels=i, axis=0)
    indice = (df.loc[df["heartrate"] < 30]).index 
    for i in indice:
        df = df.drop(labels=i, axis=0)
    return(df)

def resting_heart_rate(df_hr, df_steps):
    """
        This function uses heart rate and steps data to infer restign heart rate.
        It filters the heart rate with steps that are zero and also 12 minutes ahead.
    """
    #heartrate data:
    df_hr = df_hr.set_index("datetime")   #seta o índice como datetime
    df_hr.index.name = None  #retira o título do indice
    df_hr.index = pd.to_datetime(df_hr.index) #ter certeza que todos os índices foram convertidos para datetime corretamente

    #steps data:
    df_steps = df_steps.set_index("datetime")
    df_steps.index.name = None
    df_steps.index = pd.to_datetime(df_steps.index) #standard time stamp for steps

    #merge dataframes (método da NATURE), se eu não me engano perde uma grande quantidade de dados do hr:
    df1 = pd.merge(df_hr, df_steps, left_index=True, right_index=True) #merge somente quando ambos os horários forem iguais, se perde muita informação dos passos 
    df1 = df1.resample('1min').mean() #resample -> as amostras vão aparecer de 1 em 1 minuto e serão substituidas pela média
    df1 = df1.dropna() #remove missing values

    #merge dataframes (meu método), feito para não perder os dados de hr
    """
    df_hr = df_hr.resample('1min').mean()
    df1 = pd.merge(df_hr, df_steps, left_index=True, right_index=True)
    df1 = df1.resample('1min').mean()
    df1 = df1.dropna()
    print(df1)
    """

    # filters resting heart rate:
    df1['steps_window_12'] = df1['steps'].rolling(12).sum() #create new column (sum of steps in a 12 minute window), used to calculate rhr
    df1 = df1.loc[(df1['steps_window_12'] == 0)]  #separa uma base de dados que vai levar em consideração somente quando steps_window_12 for 0 (leva então o rhr)
    return(df1)

def pre_processing(df_hr):
    """
        This function takes resting heart rate data and applies moving averages to smooth the data and 
        downsamples to one hour by taking the avegare values
    """
    # smooth data:
    df_nonas = df_hr.dropna()  #retira valores nulos
    df1_rom = df_nonas.rolling(400).mean()  #o novo dataframe vai possuir as médias dos próximos (400?) valores de rhr e steps, isso é feito para deixar os valores mais uniformes
    
    #resample:
    df1_resmp = df1_rom.resample('1H').mean()  #ressample os valores para 1 hora, pegando as médias
    df2 = df1_resmp.drop(['steps'], axis=1)  #retira a coluna steps
    df2 = df2.dropna() #retira valores nulos
    return df2 #retorna datafram de rhr com valores já uniformes e com amostras de 1 em 1 hora

def plot(df): 
    """
        Plota os gráficos dos dataframes. 
    """
    df = df.reset_index()    #reseta o indice para eu poder utilizar posteriormente para plotar os g

    df.plot(kind='line',x='index',y='heartrate')
    plt.show()

def get_sick_time(df, participant): #guarda os valores dos timestamps onde o paciente esteve doente para futuramente plotar o gráfico com mais valores
    
    df = df.set_index("ParticipantID") #seta o indice como ParticipantID
    df.index.name = None #retira o título do indice
    
    symptom_date = df["Symptom_dates"].loc[participant] 
    covid_date = df["covid_diagnosis_dates"].loc[participant]
    recovery_date = df["recovery_dates"].loc[participant]
    
    #Essa próxima parte necessita de modificações, é uma limpeza dos dados que vai quebrar quando essas datas possuirem mais de uma chamada (pessoa ficou doente mais vezes no intervalo)
    #Pensar em outra alternativa
    symptom_date = symptom_date.rsplit("'") 
    symptom_date = symptom_date[1] 
    covid_date = covid_date.rsplit("'") 
    covid_date = covid_date[1] 
    recovery_date = recovery_date.rsplit("'") 
    recovery_date = recovery_date[1] 

    return(symptom_date, covid_date, recovery_date)

def plot_limitations(df, symptom_date, covid_date, recovery_date):
    """
        Plota os gráficos dos dataframes. Plota com as limitações de linhas para definir os momentos
        de sintomas, covid e recuperação
    """
    df = df.reset_index()    #reseta o indice para eu poder utilizar posteriormente para plotar os g

    ax = df.plot(kind='line',x='index',y='heartrate') #plota o gráfico do hr

    ax.vlines(x=symptom_date,ymin=60, ymax=70, color='y', label='symptom date')  #plota linha vertical do symptom_date
    ax.vlines(x=covid_date,ymin=60, ymax=70, color='r', label='covid_date')  #plota linha vertical do covid_date
    ax.vlines(x=recovery_date,ymin=60, ymax=70, color='g', label='recovery_date')  #plota linha vertical do recovery_date

    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()

#importar os dois arquivos
participant = "AFPB8J2"
hr_data = pd.read_csv("/home/gustavo/PibicData1/COVID-19-Wearables/AFPB8J2_hr.csv")
steps_data = pd.read_csv("/home/gustavo/PibicData1/COVID-19-Wearables/AFPB8J2_steps.csv")

#importar arquivo das pessoas doentes
df_sick = pd.read_csv("/home/gustavo/PibicData1/Sick_Values_01.txt")

#Pre-processing
hr_data = hr_outliers(hr_data)

hr_data = hr_data.drop_duplicates() #remove the duplicates
steps_data = steps_data.drop_duplicates() #remove the duplicates

hr_data = hr_data.drop(columns=["user"]) #retira a coluna de user
steps_data = steps_data.drop(columns=["user"]) #retira a coluna de user

df_rhr = resting_heart_rate(hr_data, steps_data)
df_rhr_processed = pre_processing(df_rhr)

symptom_date, covid_date, recovery_date = get_sick_time(df_sick, participant)

plot_limitations(df_rhr_processed, symptom_date, covid_date, recovery_date)







