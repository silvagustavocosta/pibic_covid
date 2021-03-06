from cmath import nan
from re import X
import string
from tracemalloc import start
from turtle import clear
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


def control_basedata(hr_data, steps_data, controle):
    hr_data = hr_data.dropna(subset=['heartrate'])
    controle["Raw_heartrate"] = len(hr_data)

    steps_data = steps_data.dropna(subset=['steps'])
    controle["Raw_steps"] = len(steps_data)


def hr_outliers(df):  # retira outliers da base de dados de hr. Outliers são hr < 30, hr > 200
    # acha o indice dos outliers
    indice = (df.loc[df["heartrate"] > 200]).index
    for i in indice:
        df = df.drop(labels=i, axis=0)
    indice = (df.loc[df["heartrate"] < 30]).index
    for i in indice:
        df = df.drop(labels=i, axis=0)
    return(df)


def qmin(df):
    """
        calcula a qualidade de cada amostra de 1 minuto de acordo com a quantidade de dados coletados naquele minuto
    """
    df = df.reset_index()
    df = df.rename(columns={'index': 'datetime'})

    first_date = df['datetime'][0]
    start_date = first_date.to_pydatetime()
    delta = datetime.timedelta(minutes=1)
    start_date = start_date.replace(second=0)
    inter_date = start_date + delta

    data = []
    qmin = []

    count = 0
    for linha in df.itertuples():
        if linha.datetime >= start_date and linha.datetime < inter_date:
            count = count + 1
        else:
            data.append(start_date)
            qmin.append(count)
            start_date = start_date + delta
            inter_date = inter_date + delta
            count = 1

    dict = {"datetime": data, "qmin": qmin}
    minq = pd.DataFrame(dict)

    minq = minq.set_index("datetime")
    return minq


def organize_dataframe(df):
    """
        Essa função vai organizar o dataframe. Seta o index como o "datetime". Colocar os valores de tempo para pandas datetime:
        Usada no resting_heart_rate() e no qmin()
    """

    df = df.set_index("datetime")
    df.index.name = None
    df.index = pd.to_datetime(df.index)

    return df


def resting_heart_rate(df_hr, df_steps, controle):
    """
        This function uses heart rate and steps data to infer restign heart rate.
        It filters the heart rate with steps that are zero and also 12 minutes ahead.
    """
    resample_time = "1min"
    controle['Merge_time'] = resample_time

    # heartrate data:
    df_hr = organize_dataframe(df_hr)

    # carregar os dados do hr data como constituintes da média das amostras por minuto
    df_hr = df_hr.resample(resample_time).mean()

    # steps data:
    df_steps = organize_dataframe(df_steps)

    # merge dataframes de heartrate e passos, deixa os dados mais uniformes quando ocorre um resample desses:
    df1 = pd.merge(df_hr, df_steps, left_index=True, right_index=True)
    df1 = df1.resample(resample_time).mean()
    df1 = df1.dropna()

    # filters resting heart rate:
    window_step = 12
    df1['steps_window_12'] = df1['steps'].rolling(window_step).sum()
    df1 = df1.loc[(df1['steps_window_12'] == 0)]

    return(df1)


def qmax(df):
    """
        Calcula a qualidade de cada amostra de uma hora de acordo com a qualidade das amostras de um minuto que constituem aquela hora
    """
    df = df.reset_index()
    df = df.rename(columns={'index': 'datetime'})

    first_date = df['datetime'][0]
    start_date = first_date.to_pydatetime()
    delta = datetime.timedelta(hours=1)
    start_date = start_date.replace(minute=0)
    inter_date = start_date + delta

    data = []
    qmax = []

    count = 0
    for linha in df.itertuples():
        if linha.datetime >= start_date and linha.datetime < inter_date:
            count = count + linha.qmin
        else:
            data.append(start_date)
            qmax.append(count)
            start_date = start_date + delta
            inter_date = inter_date + delta
            count = 0

    dict = {"datetime": data, "qmax": qmax}
    maxq = pd.DataFrame(dict)
    maxq = maxq.set_index("datetime")

    return maxq


def pre_processing(df_hr, controle):
    """
        This function takes resting heart rate data and applies moving averages to smooth the data and
        downsamples to one hour by taking the avegare values
    """
    avarage_sample = 400
    controle["Smooth_data_sample"] = avarage_sample

    # smooth data:
    df_nonas = df_hr.dropna()
    df1_rom = df_nonas.rolling(avarage_sample).mean()

    resample_time = '1H'
    controle["Resample_time"] = resample_time

    df1_resmp = df1_rom.resample(resample_time).mean()
    df2 = df1_resmp.drop(['steps'], axis=1)
    df2 = df2.dropna()

    return df2


def size(df, controle):
    """
        Vai pegar a amostra já trabalhada e descobrir quantos dias de amostragem foram feitos, carregar
        esse dado como um indicativo de qualidade para cada amostra
    """

    first_sample = df.first_valid_index()
    last_sample = df.last_valid_index()
    sample_time = last_sample - first_sample
    controle["Sample_size_days"] = sample_time.days


def arranje_sick_time(time_data):
    """
        Segunda parte da função get_sick_time, iremos pegar a lista de intervalos de sintomas, doença e recuperação e transformar em uma lista com os horários bem separados
    """

    clean_data = []
    time_data = time_data.rsplit("'")
    for idx, time in enumerate(time_data):
        if idx % 2 == 0:
            continue
        else:
            clean_data.append(time)

    return clean_data


def get_sick_time(df, participant):
    """
        Guarda os valores de quando a pessoa esteve com sintomas, doentes, em recuperação para plotar os gráficos futuramente
    """
    df = df.set_index("ParticipantID")  # seta o indice como ParticipantID
    df.index.name = None  # retira o título do indice

    # separa as datas de sintoma, doença e recuperação de acordo com o participante:
    symptom_date = df["Symptom_dates"].loc[participant]
    covid_date = df["covid_diagnosis_dates"].loc[participant]
    recovery_date = df["recovery_dates"].loc[participant]

    symptom_date = arranje_sick_time(symptom_date)
    covid_date = arranje_sick_time(covid_date)
    recovery_date = arranje_sick_time(recovery_date)

    return(symptom_date, covid_date, recovery_date)


def plot_limitations(df, symptom_date, covid_date, recovery_date, base_rhr):
    """
        Plota os gráficos dos dataframes. Plota com as limitações de linhas para definir os momentos
        de sintomas, covid e recuperação
    """
    kind = 'line'

    # reseta o indice para eu poder utilizar posteriormente para plotar os g
    df = df.reset_index()

    # plota o gráfico do hr
    ax = df.plot(kind=kind, x='index', y='heartrate')

    plot_min = df['heartrate'].min()
    plot_max = df['heartrate'].max()

    ax.vlines(x=symptom_date, ymin=plot_min, ymax=plot_max, color='y',
              label='symptom date')  # plota linha vertical do symptom_date
    ax.vlines(x=covid_date, ymin=plot_min, ymax=plot_max, color='r',
              label='covid_date')  # plota linha vertical do covid_date
    ax.vlines(x=recovery_date, ymin=plot_min, ymax=plot_max, color='g',
              label='recovery_date')  # plota linha vertical do recovery_date

    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()


def plot(df, resultado, tipo):
    """
        Plota o gráfico do dataframe sem as limitações
    """
    df = df.reset_index()
    header_name = df.columns.values[resultado]
    ax = df.plot(kind=tipo, x='index', y=header_name,)
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()


def get_base_rhr(df, controle):
    """
        Pega o valor médio do batimento cardíaco, vai ser utilizado para plotar gráficos, o valor é dado com duas casas decimais
    """

    base_rhr = df['heartrate'].mean()
    base_rhr = round(base_rhr, 2)
    controle["Base_rhr"] = base_rhr
    return base_rhr


def seasonal_vizualisation(trend, seasonal, resid):
    print("Trend:")
    plot(trend)
    print("Seasonal:")
    plot(seasonal)
    print("Residual:")
    plot(resid)


def seasonality_correction(df_rhr_processed, controle):
    """
        Aplica os métodos de correção sazonal na base de dados 
    """

    rhr = df_rhr_processed[['heartrate']]
    period = 168
    controle['Sasonal_period'] = period

    rhr_decomposition = seasonal_decompose(
        rhr, model='additive', period=period)

    rhr_trend = rhr_decomposition.trend
    rhr_seasonal = rhr_decomposition.seasonal
    rhr_resid = rhr_decomposition.resid

    # seasonal_vizualisation(rhr_trend, rhr_seasonal, rhr_resid) #visualização da correção sazonal:

    scRHR = pd.DataFrame(rhr_trend + rhr_resid)
    scRHR.rename(
        columns={scRHR.columns[0]: 'heartrate'}, inplace=True)

    return scRHR


def vector_distribution(df, n):
    """
        Separa o df em dataframes menores que vão analiser pequenos intervalos desses vetores,
        em média intervalos de 5 minutos, construir distribuição de probabilidades desses vetores
    """

    tamanho_df = len(df)
    z = tamanho_df/n

    list_df = np.array_split(df, z)

    for key, value in enumerate(list_df):
        plot(list_df[key], 1, "line")
        probability_distribution(list_df[key])
        if key > 10:
            break


def probability_distribution(df):
    """
        Plotar gráficos da função densidade de probabilidades. Histograma com essa distribuição
        e a Kernel Density Estimate
    """
    # Kernel Density Estimate:
    df['heartrate'].plot.kde(bw_method=None, color='darkorange')

    # Histogram:
    q25, q75 = np.percentile(df['heartrate'], [25, 75])
    bin_width = 2 * (q75 - q25) * len(df['heartrate']) ** (-1/3)
    # Freedman–Diaconis
    bins = round((df['heartrate'].max() - df['heartrate'].min()) / bin_width)
    plt.hist(df['heartrate'], density=True, color="darkturquoise", bins=bins)
    plt.ylabel('Probability')
    plt.xlabel('Resting Heart Rate')
    plt.show()


def standardization(df):
    """
        Standardize the data with zero meann and unit variance (Z-score).
    """

    data_scaled = StandardScaler().fit_transform(df.values)
    data_scaled_features = pd.DataFrame(
        data_scaled, index=df.index, columns=df.columns)
    data_df = pd.DataFrame(data_scaled_features)
    data = pd.DataFrame(data_df).fillna(0)
    return data


# importar os dois arquivos
participant = "A0VFT1N"
hr_data = pd.read_csv(
    "/home/gustavo/PibicData1/COVID-19-Wearables/A0VFT1N_hr.csv")
steps_data = pd.read_csv(
    "/home/gustavo/PibicData1/COVID-19-Wearables/A0VFT1N_steps.csv")

# dicionário que vai armazenar todos os parâmetros (controle de qualidade da amostra) para utilizar depois:
controle = {}

# importar arquivo das pessoas doentes
df_sick = pd.read_csv("/home/gustavo/PibicData1/Sick_Values_01.txt")

# controle de dados, pegar Batimentos Cardíacos Iniciais e passos iniciais
control_basedata(hr_data, steps_data, controle)

# Pre-processing
hr_data = hr_outliers(hr_data)

hr_data = hr_data.drop_duplicates()  # remove the duplicates
steps_data = steps_data.drop_duplicates()  # remove the duplicates

hr_data = hr_data.drop(columns=["user"])  # retira a coluna de user
steps_data = steps_data.drop(columns=["user"])  # retira a coluna de user

df_rhr = resting_heart_rate(hr_data, steps_data, controle)

minq = qmin(organize_dataframe(hr_data))

maxq = qmax(minq)

df_rhr_processed = pre_processing(df_rhr, controle)

size(df_rhr_processed, controle)

symptom_date, covid_date, recovery_date = get_sick_time(df_sick, participant)

base_rhr = get_base_rhr(df_rhr_processed, controle)

scRHR = seasonality_correction(df_rhr_processed, controle)

scRHR = scRHR.dropna()

scRHR += 0.1

stdRHR = standardization(scRHR)

hr_data_org = organize_dataframe(hr_data)
# vector_distribution(hr_data_org, 60)
# vector_distribution(df_rhr, 60)

scRHR_1 = pd.merge(scRHR, maxq,
                   left_index=True, right_index=True)

scRHR_2 = pd.merge(stdRHR, maxq,
                   left_index=True, right_index=True)

plot_limitations(scRHR_1, symptom_date,
                 covid_date, recovery_date, base_rhr)

probability_distribution(scRHR_1)

# plot_limitations(scRHR_2, symptom_date,
#                  covid_date, recovery_date, base_rhr)

# probability_distribution(scRHR_2)

# plot(scRHR_1, 2, "scatter")

# print(controle)
