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
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import hamming
import os


def main():
    mode = "solo"
    save_mode = 'off'

    Supplementary_Table = pd.read_csv(
        "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Input/Sick_Values_01.txt")

    if mode == "solo":
        subjects = []
        subjects.append("A35BJNV")
    elif mode == "full":
        subjects = Supplementary_Table.ParticipantID.values.tolist()
        # test patients:
        subjects = ["ASFODQR", "AHYIJDV", "AX6281V", "AJMQUVV", "A4G0044",
                    "AIFDJZB", "AJWW3IY", "APGIB2T", "AS2MVDL", "AYWIEKR", "AZIK4ZA"]

    for participant in subjects:
        print(participant)

        if save_mode == "on":
            base_path = "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/DataTime"
            dir_path = os.path.join(base_path, participant)
            os.mkdir(dir_path)
        else:
            dir_path = "blank"

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
        # slidingFunctions.visualizationVetores(vetoresMin, 0, 10)

        # Cálculo Detection Window
        detectionWindow = Anomaly_Detection.detection_window(
            participant, symptom_date, covid_date)

        # associar cada vetor como pré-sintomático, sintomático, covid e recuperação. sick_id vai carregar esses valores
        sick_id = slidingFunctions.vector_association(vetoresMin, dateList)

        detec_id = slidingFunctions.detectionWindowAssociation(
            vetoresMin, detectionWindow)

        # associar os parâmetros de qualidade a cada vetor
        # associar valor de qualidade para o tamanho do vetor utilizando os dados de hr_dataDay
        QvetoresHR = slidingFunctions.vector_qualityHR(
            vetoresMin, vector_lengthDays, hr_dataDay)

        QvarList, QsdList = slidingFunctions.ContVarSd(vetoresMin)
        QvarList = pd.Series(QvarList)
        QvarList = QvarList.astype(float)
        QsdList = pd.Series(QsdList)
        QsdList = QsdList.astype(float)

        # Inputar os dados nos vetores
        vetoresMinInp, QqualityConsecNa = slidingFunctions.input_data(
            vetoresMin, vector_lengthDays, initIndexes, endIndexes)

        # calcular o QqualityConsecNa para os dados brutos de HR
        QqualityConsecNa = slidingFunctions.consecutivesNa(
            hr_data, vector_lengthDays, initIndexes, endIndexes)

        # criar dataframe com todos os parâmetros de qualidade para cada vetor
        quality = pd.DataFrame()
        quality["QHR"] = QvetoresHR
        quality["QConsecNa"] = QqualityConsecNa
        quality["Var"] = QvarList
        quality["SD"] = QsdList
        mean_A = quality['QHR'].mean()
        mean_B = quality['QConsecNa'].mean()
        mean_C = quality['Var'].mean()
        mean_D = quality['SD'].mean()

        # plot dos vetores de acordo com a posição
        # slidingFunctions.visualizationVetores(vetoresMinInp, 33, 37)

        vetoresMinInp = slidingFunctions.final_input(vetoresMinInp)

        # Isolation Forest
        # Dividir os vetores em um único vetor total com a data incial do vetor sendo o índice, a coluna rhr são os valores de rhr separados
        vetoresRHR = Anomaly_Detection.organize_data(vetoresMinInp)
        # Aplicar Isolation Forest
        vetoresRHR, n_anomaly = Anomaly_Detection.isolation_forestMin(
            vetoresRHR, 0.15)
        # Análise para o plot
        time_min_mean = vetoresRHR.copy()
        time_min_mean['heartrate'] = time_min_mean['heartrate'].apply(np.mean)
        time_min_mean['sick_ID'] = sick_id
        time_min_mean['detection_window'] = detec_id

        quality["anomaly"] = time_min_mean["anomaly"].tolist()
        mean_E = quality['anomaly'].mean()
        quality.loc['mean'] = [mean_A, mean_B, mean_C, mean_D, mean_E]
        mean_A = quality.loc[quality['anomaly'] == -1, 'QHR'].mean()
        mean_B = quality.loc[quality['anomaly'] == -1, 'QConsecNa'].mean()
        mean_C = quality.loc[quality['anomaly'] == -1, 'Var'].mean()
        mean_D = quality.loc[quality['anomaly'] == -1, 'SD'].mean()
        mean_E = quality.loc[quality['anomaly'] == -1, 'anomaly'].mean()
        quality.loc['mean_anomaly'] = [mean_A, mean_B, mean_C, mean_D, mean_E]
        print(quality)

        # contar porcentagem de anomalias nos períodos doentes/sintomáticos e nos períodos de detectionwindow
        porcT, porcP = slidingFunctions.por(time_min_mean)
        print("Porcentagem de anomalias nos intervalos de sintomas, pré-sintomas e de doença:", porcT)
        print("Porcentagem de anomalias nos intervalos de detecção:", porcP)

        if save_mode == "on":
            Pre_Processing.saving_df(
                time_min_mean, dir_path, "AnDetection VetoresSW")
            Pre_Processing.saving_df(
                quality, dir_path, "Qualidade dos Vetores")

        slidingFunctions.finalPlot_quality(
            scRHR, 'qmax', "Número de Amostras por Hora dos Dados", save_mode, participant, symptom_date, covid_date, detectionWindow, recovery_date)

        Anomaly_Detection.finalPlot(time_min_mean, symptom_date, covid_date,
                                    recovery_date, detectionWindow, "Detecção de Anomalias Vetores Sliding Windows de 7 dias", save_mode, participant, "Vetor Sliding Windows", dir_path)

        # Anomaly_Detection.plot_anomaly(time_min_mean, symptom_date, covid_date,
        #                                recovery_date, pre_symptom_date, "Detecção de Anomalias Vetores Sliding Windows de 7 dias", save_mode, participant, "Vetor Sliding Windows", dir_path)

        # Trabalhar com a distorção causada pelas médias
        # slidingFunctions.MeanRealVectorization(
        #     vetoresRHR, symptom_date, covid_date, recovery_date, pre_symptom_date)

        # slidingFunctions.meanTotalVectorization(
        #     vetoresRHR, symptom_date, covid_date, recovery_date, pre_symptom_date)

        # slidingFunctions.meanHealthyVectorization(
        #     vetoresRHR, symptom_date, covid_date, recovery_date, pre_symptom_date, sick_id)
    print("DONE")


if __name__ == '__main__':
    main()
