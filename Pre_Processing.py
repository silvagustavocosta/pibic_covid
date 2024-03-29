from cmath import nan
from re import X
import string
from tracemalloc import start
from turtle import clear
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from os import listdir, system
from os.path import isfile, join
from os.path import exists
import matplotlib
import matplotlib.pyplot as plt
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.pyplot import figure


def control_basedata(hr_data, steps_data, controle, cRaw_heartrate, cRaw_steps):
    hr_data = hr_data.dropna(subset=['heartrate'])
    cRaw_heartrate.append(len(hr_data))
    controle["Raw_heartrate"] = cRaw_heartrate

    steps_data = steps_data.dropna(subset=['steps'])
    cRaw_steps.append(len(steps_data))
    controle["Raw_steps"] = cRaw_steps


def hr_outliers(df):
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


def resting_heart_rate(df_hr, df_steps, controle, cMerge_time):
    """
        This function uses heart rate and steps data to infer restign heart rate.
        It filters the heart rate with steps that are zero and also 12 minutes ahead.
    """
    resample_time = "1min"
    cMerge_time.append(resample_time)
    controle['Merge_time'] = cMerge_time

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


def moving_avarage(df_hr, controle, avarage_sample, cSmooth_data_sample):
    """
        This function takes resting heart rate data and applies moving averages to smooth the data
    """
    cSmooth_data_sample.append(avarage_sample)
    controle["Smooth_data_sample"] = cSmooth_data_sample

    df_nonas = df_hr.dropna()
    df1_rom = df_nonas.rolling(avarage_sample).mean()

    return df1_rom


def pre_processing(df1_rom):
    """
        This function downsamples to one hour by taking the avegare values from that hour
    """
    resample_time = '1H'

    df1_resmp = df1_rom.resample(resample_time).mean()
    df2 = df1_resmp.drop(['steps'], axis=1)
    df2 = df2.dropna()

    return df2


def arranje_sick_time(time_data):
    """
        Segunda parte da função get_sick_time, iremos pegar a lista de intervalos de sintomas, 
        doença e recuperação e transformar em uma lista com os horários bem separados
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


def plot_limitations(df, coluna, symptom_date, covid_date, recovery_date, title, saveMode, participant):

    fig, ax = plt.subplots()
    plot_min = df['heartrate'].min()
    plot_max = df['heartrate'].max()

    ax.plot(df.index, df[coluna], label='RHR', marker='.')

    plt.gcf().set_size_inches(12, 6)

    ax.vlines(x=symptom_date, ymin=plot_min, ymax=plot_max, color='y',
              label='symptom date')
    ax.vlines(x=covid_date, ymin=plot_min, ymax=plot_max, color='r',
              label='covid_date')
    ax.vlines(x=recovery_date, ymin=plot_min, ymax=plot_max, color='g',
              label='recovery_date')

    plt.xlabel('Data')
    plt.ylabel('Frequência cardíaca em repouso')

    plt.title(title)
    plt.gcf().autofmt_xdate()
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    if saveMode == "on":
        base_path = "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Data"
        figName = title + ".jpg"
        dir_path = os.path.join(base_path, participant, figName)
        plt.savefig(dir_path)


def plot(df, resultado, tipo):
    """
        Plota o gráfico do dataframe sem as limitações
    """

    df = df.reset_index()
    header_name = df.columns.values[resultado]
    ax = df.plot(kind=tipo, x='index', y=header_name,
                 label="Resting Heart Rate", marker=",")

    plt.gcf().set_size_inches(10, 4)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()


def plot_quality(df, coluna, title, saveMode, participant):
    """
        Plota um gráfico tipo scatter para mostrar a qualidade dos dados de amostra 
    """

    fig, ax = plt.subplots()

    plt.scatter(df.index, df[coluna], c=df[coluna],
                cmap='Blues')
    plt.gcf().set_size_inches(12, 6)

    cbar = plt.colorbar()
    cbar.set_label('Qualidade')

    plt.title(title)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    if saveMode == "on":
        base_path = "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Data"
        figName = title + ".jpg"
        dir_path = os.path.join(base_path, participant, figName)
        plt.savefig(dir_path)

    plt.show()


def get_base_rhr(df, controle, cBase_rhr):
    """
        Pega o valor médio do batimento cardíaco, vai ser utilizado para plotar gráficos, o valor é dado com duas casas decimais
    """

    base_rhr = df['heartrate'].mean()
    base_rhr = round(base_rhr, 2)
    cBase_rhr.append(base_rhr)
    controle["Base_rhr"] = cBase_rhr
    return base_rhr


def seasonal_vizualisation(trend, seasonal, resid):
    print("Trend:")
    plot(trend, 1, "line")
    print("Seasonal:")
    plot(seasonal, 1, "line")
    print("Residual:")
    plot(resid, 1, "line")


def seasonality_correction(df_rhr_processed, controle, period, cSasonal_period):
    """
        Aplica os métodos de correção sazonal na base de dados
    """

    rhr = df_rhr_processed[['heartrate']]

    # caso precise passar o sasonal_period como parâmetro para acompanhar no dicionário
    # cSasonal_period.append(period)
    # controle['Sasonal_period'] = cSasonal_period

    rhr_decomposition = seasonal_decompose(
        rhr, model='additive', period=period)

    rhr_trend = rhr_decomposition.trend
    rhr_seasonal = rhr_decomposition.seasonal
    rhr_resid = rhr_decomposition.resid

    # visualização da correção sazonal:
    # seasonal_vizualisation(rhr_trend, rhr_seasonal, rhr_resid)

    scRHR = pd.DataFrame(rhr_trend + rhr_resid)
    scRHR.rename(
        columns={scRHR.columns[0]: 'heartrate'}, inplace=True)

    return scRHR


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
    if bin_width == 0:
        bin_width = 0.1

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


def saving_df(archive, dir_path, name):
    """
        Salva um arquivo específico no local definido para funcionar como
    """
    file_path = os.path.join(dir_path, name)
    archive.to_csv(file_path)


def seasonal_period(df, mode):
    """
        Encontra o melhor período da amostra para ser utilizado no seasonal_decompose(),
        de acordo com o tipo, tamanho e tempo de amostragem
        Modes = hour, minute
    """

    # TODO se o tamanho do dataset é menor que o dobro de 168 (para horas) e 10080 (para minutos)
    # utilziar period calculado pela seguinte formula:

    if mode == 'hour' and len(df) >= 336:
        period = 168
    elif mode == 'minute' and len(df) >= 20160:
        period = 10080
    else:
        dfSize = len(df)
        dfTime = (df.index[-1] - df.index[0])/np.timedelta64(1, 'W')

        period = (dfSize/dfTime)
        period = int(period)

    return period


def fill_missing_dates(df):
    """
    Takes a DataFrame with a datetime index and a numeric column, and resamples the DataFrame 
    to fill missing values with zeros.
    """
    df.set_index('datetime', inplace=True)
    # Resample the DataFrame with a minute frequency and fill missing values with zeros
    df_resampled = df.resample('T').sum().fillna(0)
    df_resampled.reset_index(inplace=True)

    return df_resampled


def final_processing(hr_data, steps_data, participant, cParticipantID, cRaw_heartrate, cRaw_steps, cMerge_time, cSmooth_data_sample, cBase_rhr, cSasonal_period):
    """
        Makes all the processes of the data from one person, returning the 3 dataframes and the control dataset 
    """
    # dicionário que vai armazenar todos os parâmetros (controle de qualidade da amostra) para utilizar depois:
    controle = {}
    cParticipantID.append(participant)
    controle["ParticipantID"] = cParticipantID

    # controle de dados, pegar Batimentos Cardíacos Iniciais e passos iniciais
    control_basedata(hr_data, steps_data, controle, cRaw_heartrate, cRaw_steps)

    # Pre-processing
    hr_data = hr_outliers(hr_data)

    hr_data = hr_data.drop_duplicates()  # remove the duplicates
    steps_data = steps_data.drop_duplicates()  # remove the duplicates

    hr_data = hr_data.drop(columns=["user"])  # retira a coluna de user
    steps_data = steps_data.drop(columns=["user"])  # retira a coluna de user

    # fill steps_data with 0 when there is no date
    steps_data['datetime'] = pd.to_datetime(
        steps_data['datetime'], format='%Y-%m-%d %H:%M:%S')
    steps_data = fill_missing_dates(steps_data)

    df_rhr = resting_heart_rate(
        hr_data, steps_data, controle, cMerge_time)

    minq = qmin(organize_dataframe(hr_data))

    maxq = qmax(minq)

    # TODO forma melhor de escrever essa próxima parte do código
    df_rhr_avarage = moving_avarage(df_rhr, controle, 400, cSmooth_data_sample)

    df_rhr_avarage1 = df_rhr_avarage.drop(['steps'], axis=1)
    df_rhr_avarage2 = df_rhr_avarage1.dropna()

    df_rhr_processed = pre_processing(df_rhr_avarage)

    base_rhr = get_base_rhr(df_rhr_processed, controle, cBase_rhr)

    period = seasonal_period(df_rhr_processed, 'hour')
    # period = 168
    scRHR = seasonality_correction(
        df_rhr_processed, controle, period, cSasonal_period)

    period = seasonal_period(df_rhr_avarage2, 'minute')
    # period = 10080
    minutes_rhr = seasonality_correction(
        df_rhr_avarage2, controle, period, cSasonal_period)
    minutes_rhr = minutes_rhr.dropna()

    scRHR = scRHR.dropna()

    scRHR += 0.1

    stdRHR = standardization(scRHR)

    scRHR_1 = pd.merge(scRHR, maxq,
                       left_index=True, right_index=True)

    stdRHR_1 = pd.merge(stdRHR, maxq,
                        left_index=True, right_index=True)

    return(scRHR_1, stdRHR_1, minutes_rhr, base_rhr, controle)


def save_files(participant, scRHR, stdRHR, minutesRHR, base_rhr, symptom_date, covid_date, recovery_date):
    """
        Saves the dataframes and the images files on the directory, based on the name of the participant
    """

    base_path = "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Data"
    dir_path = os.path.join(base_path, participant)
    os.mkdir(dir_path)

    saving_df(scRHR, dir_path, "scRHR")

    saving_df(stdRHR, dir_path, "stdRHR")

    saving_df(minutesRHR, dir_path, "minutesRHR")


def main():
    # for pessoa in set:
    #     pre_processed(data)
    #     save_data()

    mode = "solo"
    save_mode = "off"

    Supplementary_Table = pd.read_csv(
        "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Input/Sick_Values_01.txt")
    # Supplementary_Table = pd.read_csv(
    #     "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Input/One.csv")
    if mode == "solo":
        subjects = []
        subjects.append("ASFODQR")
    elif mode == "full":
        subjects = Supplementary_Table.ParticipantID.values.tolist()
        # test subjects
        # subjects = ["AFPB8J2", "APGIB2T", "A0NVTRV", "A4G0044",
        #             "AS2MVDL", "ASFODQR", "AYWIEKR", "AJMQUVV"]

    df_sick = pd.read_csv(
        "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Input/Sick_Values_01.txt")
    # df_sick = pd.read_csv(
    #     "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Input/One.csv")

    # criar os parametros do dicionario controle em formato de listas (carregam os valores)
    # para todos os participantes
    cParticipantID = []
    cRaw_heartrate = []
    cRaw_steps = []
    cMerge_time = []
    cSmooth_data_sample = []
    cBase_rhr = []
    cSasonal_period = []

    for subject in subjects:
        # importar os arquivos
        participant = subject
        print(participant)

        hr_data = pd.read_csv(
            "/home/gustavo/PibicData1/COVID-19-Wearables/" + participant + "_hr.csv")
        steps_data = pd.read_csv(
            "/home/gustavo/PibicData1/COVID-19-Wearables/" + participant + "_steps.csv")
        # hr_data = pd.read_csv(
        #     "/mnt/d/PIBIC/Data Final/" + participant + "_hr.csv")
        # steps_data = pd.read_csv(
        #     "/mnt/d/PIBIC/Data Final/" + participant + "_steps.csv")

        symptom_date, covid_date, recovery_date = get_sick_time(
            df_sick, participant)

        # em final_processing estamos passando as listas que vão formar o dicionário final total
        scRHR, stdRHR, minutesRHR, base_rhr, controle = final_processing(
            hr_data, steps_data, participant, cParticipantID, cRaw_heartrate, cRaw_steps, cMerge_time, cSmooth_data_sample, cBase_rhr, cSasonal_period)

        # ploting:
        plot_limitations(scRHR, 'heartrate', symptom_date,
                         covid_date, recovery_date, "Batimentos cardíacos em repouso divididos em horas", save_mode, participant)

        plot_limitations(minutesRHR, 'heartrate', symptom_date,
                         covid_date, recovery_date, "Batimentos cardíacos em repouso divididos em minutos", save_mode, participant)

        plot_quality(
            scRHR, 'qmax', "Número de Amostras por Hora dos Dados", save_mode, participant)

        if save_mode == "on":
            save_files(participant, scRHR, stdRHR, minutesRHR,
                       base_rhr, symptom_date, covid_date, recovery_date)
        else:
            continue

        # probability_distribution(scRHR)

    controle = pd.DataFrame.from_dict(controle)
    # print(controle)
    if save_mode == "on":
        base_path = "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Data"
        saving_df(controle, base_path, "controle")


if __name__ == '__main__':
    main()
