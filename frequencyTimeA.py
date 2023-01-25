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
import slidingwindows


def main():
    mode = "solo"

    Supplementary_Table = pd.read_csv(
        "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Input/Sick_Values_01.txt")

    if mode == "solo":
        subjects = []
        subjects.append("AS2MVDL")
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

        dateList, pre_symptom_date, symptom_date, covid_date, recovery_date = slidingwindows.time_stamps(
            df_sick, participant)

        vector_lengthDays = 6
        vetoresMin, dfMin, minutesRHR, initIndexes, endIndexes = slidingwindows.data_org(
            minutesRHR, vector_lengthDays)

        # plot dos vetores de acordo com a posição
        # slidingFunctions.visualizationVetores(vetoresMin, 33, 37)
        # idicesdeexemplo = [33, 34, 35, 36]

        # associar cada vetor como pré-sintomático, sintomático, covid e recuperação. sick_id vai carregar esses valores
        sick_id = slidingFunctions.vector_association(vetoresMin, dateList)

        # associar os parâmetros de qualidade a cada vetor
        # associar valor de qualidade para o tamanho do vetor utilizando os dados de hr_dataDay
        vetoresHR = slidingFunctions.vector_qualityHR(
            vetoresMin, vector_lengthDays, hr_dataDay)

        varList, sdList = slidingFunctions.ContVarSd(vetoresMin)

        # Associar intervalos consecutivos sem dados para cada vetor (definir separadamente um treshhold e aplicar no vetor).
        # Inputar os dados nos vetores
        # TODO utilizar os dados brutos para o cálculo de qualityConsecNa e não os dados em rhr (que vão levar em consideração os momentos de descanso)
        vetoresMinInp, qualityConsecNa = slidingFunctions.input_data(
            vetoresMin, vector_lengthDays, initIndexes, endIndexes)


if __name__ == '__main__':
    main()
