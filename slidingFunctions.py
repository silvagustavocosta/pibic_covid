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
import slidingwindows


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

    initIndeesC = df.index[df.index.get_indexer(initIndexes, method="nearest")]
    endIndexesC = df.index[df.index.get_indexer(endIndexes, method="nearest")]

    vetores = []
    for i in range(len(initIndeesC)):
        vetor = df[initIndeesC[i]:endIndexesC[i]]
        vetores.append(vetor)

    return vetores, initIndexes, endIndexes


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


def vector_association(vetoresMin, dateList):
    """
        Associar cada vetor como pré-sintomático, sintomático, covid e recuperação
    """

    sick_id = []
    if len(dateList) != 0:
        for vetor in vetoresMin:
            if vetor.index[0] < dateList[0]["date"]:
                sick_id.append(0)
    if len(dateList) != 0:
        for count, lDict in enumerate(dateList):
            if count < (len(dateList)-1):
                next_date = dateList[count + 1]["date"]
                for vetor in vetoresMin:
                    if vetor.index[0] >= lDict["date"] and vetor.index[0] < next_date:
                        sick_id.append(lDict["status"])
            else:
                for vetor in vetoresMin:
                    if vetor.index[0] >= lDict["date"]:
                        sick_id.append(lDict["status"])
    else:
        for vetor in vetoresMin:
            sick_id.append(0)

    return sick_id


def detectionWindowAssociation(vetoresMin, detectionWindow):
    """
        Associar os vetores com o período marcado pela detection window
    """

    detec_id = []
    if not detectionWindow:
        return

    for vetor in vetoresMin:
        x = 0
        first_date = vetor.index[0]
        for count in range(0, len(detectionWindow), 2):
            start_date = detectionWindow[count]
            end_date = detectionWindow[count + 1]
            if start_date <= first_date <= end_date:
                detec_id.append(1)
                x = 1
                break
        if x == 0:
            detec_id.append(0)
        else:
            x = 1

    return detec_id


def vector_qualityHR(vetoresMin, vector_lengthDays, hr_dataDay):
    """
        Associar o valor de qualidade para o tamanho do vetor utilizando os dados de hr_dataDay. Se o vetor for do tamanho de 7 dias,
        associar os 7 dias para esse vetor 
    """

    vetoresHR = []
    for vetor in vetoresMin:
        firstDay = vetor.index[0]
        lastDay = vetor.index[-1]

        daysDF = pd.date_range(start=firstDay,
                               end=lastDay, freq='1D')
        daysList = []
        for day in daysDF:
            day = day.replace(hour=0, minute=0, second=0, microsecond=0)
            daysList.append(day)

        soma = 0
        for day in daysList:
            x = hr_dataDay._get_value(day, "Qday_HR", takeable=False)
            soma = soma + x
        mean = soma/(vector_lengthDays+1)

        vetoresHR.append(mean)

    return vetoresHR


def ContVarSd(vetoresMin):
    """
        Calcular a Variância e Desvio Padrão de cada vetor
    """

    varList = []
    sdList = []
    for vetor in vetoresMin:
        var = vetor.var()
        sd = vetor.std()
        varList.append(var)
        sdList.append(sd)

    return varList, sdList


def input_data(vetoresMin, vector_lengthDays, initIndexes, endIndexes):
    """
        Garantir que todos os vetores tenham o mesmo tamanho (mesmo número de dados por análise). Inputar os dados na análise.[
        Associar o lengths_consecutive_na a cada vetor, definir o treshhold e aplicar ao vetor, carregar esse valor como qualidade
        para o vetor.
    """
    vetoresMinInp = []
    qualityConsecNa = []

    count = 0

    for i, vetor in enumerate(vetoresMin):

        vetor, totalLen, dfLen, lengths_consecutive_na = Anomaly_Detection.number_of_inputs(
            vetor, "on", initIndexes[i], endIndexes[i])

        # os NaN valores que sobraram são substituídos pela média dos dados de rhr
        columnavarage = vetor["heartrate"].mean()
        vetor = vetor.fillna(columnavarage)

        vetoresMinInp.append(vetor)

        # Associar intervalos consecutivos sem dados para cada vetor (definir separadamente um treshhold e aplicar no vetor)
        lengths_consecutive_na = lengths_consecutive_na.to_frame()
        lengths_consecutive_na.set_axis(
            ["nullSize"], axis="columns", inplace=True)

        dataTresh = lengths_consecutive_na.loc[lengths_consecutive_na["nullSize"] > 240]
        somaLen = dataTresh.sum()
        qualityConsecNaValue = (somaLen/len(vetor))*100

        qualityConsecNa.append(float(qualityConsecNaValue))

    return vetoresMinInp, qualityConsecNa


def consecutivesNa(hr_data, vector_lengthDays, initIndexes, endIndexes):
    """
        Calcular o qualityConsecNa para os dados brutos de HR
    """

    hr_data = hr_data.set_index("datetime")
    hr_data.index.name = None
    hr_data.index = pd.to_datetime(hr_data.index)

    hr_data = hr_data.reset_index().drop_duplicates(
        subset='index', keep='first').set_index('index')

    originalIndexes = hr_data.index
    originalIndexes = originalIndexes.drop_duplicates(keep=False)

    originalIndexes = sorted(originalIndexes)
    endIndexes = sorted(endIndexes)
    originalIndexes = pd.to_datetime(originalIndexes)
    endIndexes = pd.to_datetime(endIndexes)

    initIndexesC = originalIndexes[originalIndexes.get_indexer(
        initIndexes, method="nearest")]
    endIndexesC = originalIndexes[originalIndexes.get_indexer(
        endIndexes, method="nearest")]

    # separar a data de HR nos períodos estabelecidos por initIndexesC e endIndexesC
    vetores = []
    for i in range(len(initIndexesC)):
        vetor = hr_data[initIndexesC[i]:endIndexesC[i]]
        vetores.append(vetor)

    # erro no vetores[21]
    vetoresMinInp, qualityConsecNa = input_data(
        vetores, vector_lengthDays, initIndexesC, endIndexesC)

    return qualityConsecNa


def MeanRealVectorization(vetores, symptom_date, covid_date, recovery_date, pre_symptom_date):
    """
        Calcular a média móvel dos dados de RHR nos vetores, inicialmente calcular a média para um vetor, 
        depois para dois vetores e incrementar a quantidade de vetores para cada passo dado;
    """

    # adicionar índice no dataframe
    vetores = vetores.reset_index(inplace=False)

    meanTotal = []
    for i in range(1, len(vetores)+1):
        df = vetores[0:i]

        rhrArrays = df["heartrate"]
        mergedArray = np.concatenate(rhrArrays)
        mean = mergedArray.mean()

        meanTotal.append(mean)

    # calcular os valores da distorção
    listDistortion = distortion(vetores, meanTotal)
    vetores["realDistortion"] = listDistortion

    # plotar os valores da distorção de acordo com o tamanho do vetor
    vetores = vetores.set_index("index")
    vetores.index.name = None
    vetores.index = pd.to_datetime(vetores.index)

    # Pre_Processing.plot(vetores, 4, "scatter")
    distortionPlot(vetores, symptom_date, covid_date,
                   recovery_date, pre_symptom_date)


def meanTotalVectorization(vetores, symptom_date, covid_date, recovery_date, pre_symptom_date):
    """
        Calcular a média usando todos os vetores disponíveis menos o atual;
        Calcular a distorção para cada vetor calculado usando essa média;
    """

    # adicionar índice no dataframe
    vetores = vetores.reset_index(inplace=False)

    meanTotal = []
    for i in range(len(vetores)):
        df = vetores.drop(i)

        df = df.reset_index(inplace=False)
        rhrArrays = df["heartrate"]
        mergedArray = np.concatenate(rhrArrays)

        mean = mergedArray.mean()
        meanTotal.append(mean)

    listDistortion = distortion(vetores, meanTotal)
    vetores["realDistortion"] = listDistortion

    # plotar os valores da distorção de acordo com o tamanho do vetor
    vetores = vetores.set_index("index")
    vetores.index.name = None
    vetores.index = pd.to_datetime(vetores.index)

    # Pre_Processing.plot(vetores, 4, "scatter")
    distortionPlot(vetores, symptom_date, covid_date,
                   recovery_date, pre_symptom_date)


def meanHealthyVectorization(vetores, symptom_date, covid_date, recovery_date, pre_symptom_date, sick_id):
    """
        Calcular a média apenas quando o paciente estava bem ou pré-sintomático;
    """

    vetores["sick"] = sick_id

    df = vetores.loc[(vetores['sick'] == 0) | (vetores['sick'] == 3)]
    df = df.reset_index(inplace=False)

    rhrArrays = df["heartrate"]
    mergedArray = np.concatenate(rhrArrays)

    mean = mergedArray.mean()
    meanTotal = []
    for i in range(len(vetores)):
        meanTotal.append(mean)

    listDistortion = distortion(vetores, meanTotal)
    vetores["realDistortion"] = listDistortion

    vetores = vetores.drop(columns="sick")

    # Pre_Processing.plot(vetores, 4, "scatter")
    distortionPlot(vetores, symptom_date, covid_date,
                   recovery_date, pre_symptom_date)


def distortionPlot(df, symptom_date, covid_date, recovery_date, pre_symptom_date):
    fig, ax = plt.subplots()
    plot_min = df['realDistortion'].min()
    plot_max = df['realDistortion'].max()

    ax.scatter(df.index, df['realDistortion'],
               label='realDistortion', zorder=0, s=10)

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
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()


def distortion(vetores, meanTotal):
    """
        Calcular a distorção para cada vetor calculado nessa média, dado pela diferença das médias ao longo dos vetores, comparar o dado em rhr
        com a média do vetor atual
    """

    listDistortion = []
    for i in range(1, len(vetores)+1):
        mean = meanTotal[i-1]

        df = vetores[0:i]
        rhrArrays = df["heartrate"]
        mergedArray = np.concatenate(rhrArrays)

        minor_distortion = []
        for rhr in mergedArray:
            x = (rhr - mean)**2
            minor_distortion.append(x)

        distortion = (sum(minor_distortion))/len(minor_distortion)
        listDistortion.append(distortion)

    return listDistortion


def final_input(vetoresMinInp):
    """
        Caso ainda existam vetores que são totalmente formados por NaN values, inputar dados da média total dos RHR
    """

    df_concat = pd.concat(vetoresMinInp, axis=0)
    baserhr = df_concat['heartrate'].mean()

    for vetor in vetoresMinInp:
        if vetor["heartrate"].isna().all() == True:
            vetor["heartrate"] = baserhr

    return vetoresMinInp


def por(df):
    """
        Contar porcentagem de anomalias nos períodos doentes/sintomáticos e nos períodos de detectionwindow
    """

    nAno = df.loc[(df["anomaly"] == -1)]

    nSic = df.loc[(((df["anomaly"] == -1) & (df["sick_ID"] == 1)) |
                   ((df["anomaly"] == -1) & (df["sick_ID"] == 2)) | ((df["anomaly"] == -1) & (df["sick_ID"] == 3)))]
    porcT = (len(nSic)/len(nAno))*100

    nDet = df.loc[((df["anomaly"] == -1) & (df["detection_window"] == 1))]
    porcP = (len(nDet)/len(nAno))*100

    return porcT, porcP


def finalPlot_quality(df, coluna, title, saveMode, participant, symptom_date, covid_date, detectionWindow, recovery_date, dir_path):
    """
        Plota os dados de qualidade baseados em scRHR
    """

    df.set_index(df.columns[0], inplace=True)
    new_index = pd.to_datetime(df.index)
    df.index = new_index

    fig, ax = plt.subplots()
    plot_min = df['qmax'].min()
    plot_max = df['qmax'].max()

    plt.scatter(df.index, df[coluna], color="b")
    # plt.scatter(df.index, df[coluna], c=df[coluna], cmap='Blues')
    plt.gcf().set_size_inches(12, 6)

    # cbar = plt.colorbar()
    # cbar.set_label('Qualidade')

    if symptom_date:
        ax.vlines(x=symptom_date, ymin=plot_min, ymax=plot_max, color='purple',
                  label='symptom date', linestyle='--')
    if covid_date:
        ax.vlines(x=covid_date, ymin=plot_min, ymax=plot_max, color='r',
                  label='covid_date', linestyle='--')
    if recovery_date:
        ax.vlines(x=recovery_date, ymin=plot_min, ymax=plot_max, color='g',
                  label='recovery_date', linestyle='--')
    if detectionWindow:
        for count in range(0, len(detectionWindow), 2):
            ax.plot(detectionWindow[count], plot_max,
                    color="orange", marker=">", markersize=20)
            ax.plot(detectionWindow[count+1], plot_max,
                    color="orange", marker="<", markersize=20)
            plt.legend(markerscale=0.5)

    plt.xlabel('Data')
    plt.ylabel('Número de dados de frequência cardíaca na hora')

    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.title(title)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    if saveMode == "on":
        base_path = dir_path
        figName = title + ".jpg"
        dir_path = os.path.join(base_path, figName)
        plt.savefig(dir_path)

    plt.show()


def anomaly_frequency(df):
    """
        Vai calcular a frequência de anomalias em cada uma das etapas onde os dados estão divididos, período pré-sintomático,
        período sintomático e período saudável. 
    """

    # cálculo para o período saudável
    countVetores = len(df[(df['sick_ID'] == 0)])
    countAnomalys = len(df[(df['sick_ID'] == 0) & (df['anomaly'] == -1)])
    if countVetores == 0:
        freq_health = nan
    else:
        freq_health = countAnomalys/countVetores

    # cálculo para o período pre-sintomatico
    countVetores = len(df[(df['sick_ID'] == 3)])
    countAnomalys = len(df[(df['sick_ID'] == 3) & (df['anomaly'] == -1)])
    if countVetores == 0:
        freq_pre = nan
    else:
        freq_pre = countAnomalys/countVetores

    # cálculo para o período sintomático
    countVetores = len(df[(df['sick_ID'] == 1)
                          | (df['sick_ID'] == 2)])
    countAnomalys = len(df[(df['sick_ID'] == 1) & (
        df['anomaly'] == -1)]) + len(df[(df['sick_ID'] == 2) & (df['anomaly'] == -1)])
    if countVetores == 0:
        freq_symp = nan
    else:
        freq_symp = countAnomalys/countVetores

    return freq_health, freq_pre, freq_symp


def filtragem(df):
    """
        Loop to iterate over the elements of the column and check for the specific condition of having a sequence of seven numbers with one 1 and six 0.
    """

    df = df.reset_index(drop=False)
    i = 0
    while i < len(df):
        if df.loc[i, 'anomaly'] == -1 and i <= len(df) - 7:
            if df.loc[i+1:i+6, 'anomaly'].tolist() == [1, 1, 1, 1, 1, 1] and (i == 0 or df.loc[i-1, 'anomaly'] != -1) and (i+7 == len(df) or df.loc[i+7, 'anomaly'] != -1):
                df.loc[i, 'anomaly'] = 1
                i += 7  # Skip the next 7 elements since we know they are all 1's
            else:
                i += 1
        else:
            i += 1

    df = df.set_index('index')

    return df
