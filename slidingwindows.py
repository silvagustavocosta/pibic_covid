from time import time
import Pre_Processing
import Anomaly_Detection
import Test_Functions
import slidingFunctions
import pandas as pd
from main import dateListSeparation
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.ensemble import IsolationForest


def data_org(minutesRHR, vetorsize):
    """
        Separa o minutesRHR (dataframe minutesRHR ou scRHR) e vai dividir o paciente em vetores (os vetores vão possuir 7, 14, 21 ou 28 dias), 
        vão estar separados entre si em 1 dia.
        O último dia do vetor é o que define o próprio vetor 
    """

    minutesRHR = minutesRHR.set_index("datetime")
    minutesRHR.index.name = None
    minutesRHR.index = pd.to_datetime(minutesRHR.index)

    vetoresMin = slidingFunctions.day_time_separation(minutesRHR, vetorsize)

    # construindo o dataframe que vai carregar os índices (última data dos vetores) e a quantidade de anomalias
    df = pd.DataFrame()

    vetorIdx = []
    for vetor in vetoresMin:
        idx = vetor.index[-1]
        vetorIdx.append(idx)
    df["datetime"] = vetorIdx
    df = df.set_index("datetime")
    df.index.name = None
    df.index = pd.to_datetime(df.index)

    vetorData = []
    for vetor in vetoresMin:
        vlist = vetor["heartrate"].tolist()
        vetorData.append(vlist)

    df["data"] = vetorData

    return vetoresMin, df, minutesRHR


def isolationForestVetores(vetores, df):
    """
        Utilizar a detecção de anomalias por Isolation Forest no vetor, contar o número de anomalias no 
        intervalo e associar ao último dia do vetor (aquele que determina o próprio vetor). Associar cada 
        vetor a uma data (índice) e o número de anomalias (coluna) de um dataframe 
    """

    countAnomaly = []
    for vetor in vetores:
        model = IsolationForest(n_estimators=100, max_samples='auto',
                                contamination=float(0.02), max_features=1.0)

        x_dates = vetor.index.values
        y_rhr = vetor[["heartrate"]].values

        model.fit(y_rhr)

        vetor['scores'] = model.decision_function(y_rhr)
        vetor['anomaly'] = model.predict(y_rhr)

        anomaly_count = len(vetor.loc[vetor['anomaly'] == -1])
        countAnomaly.append(anomaly_count)

    df["countAnomaly"] = countAnomaly

    return vetores, df


def time_stamps(df_sick, participant):
    """
        Faz todas as análises para separar os períodos de interesse.
        PreSymptomatic, Symptomatic, Covid, Recovery.
    """

    symptom_date, covid_date, recovery_date = Pre_Processing.get_sick_time(
        df_sick, participant)

    symptom_date, covid_date, recovery_date = Anomaly_Detection.get_date(
        symptom_date, covid_date, recovery_date)

    # # Período pré-sintomático: 14 dias antes do início das anomalias
    if symptom_date:
        pre_symptom_date = []
        pre_symptom_date.append(symptom_date[0] - datetime.timedelta(days=14))
    else:
        pre_symptom_date = None

    dateList = Anomaly_Detection.sort_datas(symptom_date, covid_date,
                                            recovery_date, pre_symptom_date)

    pre_symptom_date, symptom_date, covid_date, recovery_date = dateListSeparation(
        dateList)

    return dateList, pre_symptom_date, symptom_date, covid_date, recovery_date


def main():
    mode = "solo"

    Supplementary_Table = pd.read_csv(
        "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Input/Sick_Values_01.txt")

    if mode == "solo":
        subjects = []
        subjects.append("AR4FPCC")
    elif mode == "full":
        subjects = Supplementary_Table.ParticipantID.values.tolist()
        # test patients:
        # subjects = ["AFPB8J2", "APGIB2T", "A0NVTRV", "A4G0044",
        #             "AS2MVDL", "ASFODQR", "AYWIEKR", "AJMQUVV"]

    for participant in subjects:
        print(participant)

        minutesRHR = pd.read_csv(
            "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Data/" + participant + "/minutesRHR")
        scRHR = pd.read_csv(
            "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Data/" + participant + "/scRHR")

        # importar dados brutos para realizar a análise de qualidade
        hr_data = pd.read_csv(
            "/home/gustavo/PibicData1/COVID-19-Wearables/" + participant + "_hr.csv")
        steps_data = pd.read_csv(
            "/home/gustavo/PibicData1/COVID-19-Wearables/" + participant + "_steps.csv")

        # dataframe que contém os resultados das análises de main.py
        controleQuality = pd.read_csv(
            "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Data/controleAnalysis")
        controleQuality = controleQuality.sort_values(
            by="vetoresPorT", ascending=False)
        # print(controleQuality)
        # break

        hr_data = Pre_Processing.hr_outliers(hr_data)
        hr_data = hr_data.drop_duplicates()  # remove the duplicates
        steps_data = steps_data.drop_duplicates()  # remove the duplicates
        hr_data = hr_data.drop(columns=["user"])  # retira a coluna de user
        steps_data = steps_data.drop(
            columns=["user"])  # retira a coluna de user

        # getting timestamps from when the subject is sick:
        df_sick = pd.read_csv(
            "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Input/Sick_Values_01.txt")

        # encontrando a qualidade das amostras HR
        hr_dataHour, hr_dataDay, hr_dataWeek = slidingFunctions.qualityHR(
            hr_data)

        # encontrando a qualidade das amostras RHR
        minutesRHR.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)
        rhr_dataHour, rhr_dataDay, rhr_dataWeek = slidingFunctions.qualityHR(
            minutesRHR)

        dateList, pre_symptom_date, symptom_date, covid_date, recovery_date = time_stamps(
            df_sick, participant)

        vetoresMin, dfMin, minutesRHR = data_org(minutesRHR, 7)

        # plot dos vetores de acordo com a posição
        # slidingFunctions.visualizationVetores(vetoresMin, 0, 7)

        # Isolation Forest em cada vetor
        vetoresMin, dfMin = isolationForestVetores(vetoresMin, dfMin)

        # plot dos vetores com as anomalias
        # slidingFunctions.plotAnomalyVetores(vetoresMin, 0, 6)

        # plota o número de anomalias vs data em comparativo com as curvas do dataframe
        slidingFunctions.plotFullAnalysis(
            minutesRHR, dfMin, pre_symptom_date, symptom_date, covid_date, recovery_date)


if __name__ == '__main__':
    main()
