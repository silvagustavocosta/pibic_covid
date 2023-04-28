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

    # Supplementary_Table = pd.read_csv(
    #     "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Input/Sick_Values_01.txt")
    Supplementary_Table = pd.read_csv(
        "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Input/One.csv")

    if mode == "solo":
        subjects = []
        subjects.append("P305571")
    elif mode == "full":
        # subjects = Supplementary_Table.ParticipantID.values.tolist()
        # usuarios com dados limpos:
        subjects = ["P110465", "P111019", "P182427", "P206998", "P230742", "P249349", "P256033", "P271946", "P279697", "P292181", "P305571", "P320539", "P333074", "P355472", "P389953", "P401732", "P442730", "P469888", "P469946", "P476443", "P476514", "P500432", "P511540", "P516467",
                    "P530788", "P542912", "P543995", "P549078", "P584112", "P612886", "P625831", "P631814", "P635568", "P662021", "P662924", "P682517", "P693795", "P708653", "P723961", "P726139", "P741171", "P749288", "P754260", "P759795", "P799237", "P839431", "P851598", "P954010", "P992022"]
        # subjects = ["AFPB8J2", "APGIB2T", "AQC0L71", "AD77K91", "A3OU183", "AX6281V", "A4E0D03", "AS2MVDL", "AKXN5ZZ", "AF3J1YC", "AJWW3IY", "AAXAA7Z",
        #             "AHYIJDV", "AURCTAK", "A1K5DRI", "A7EM0B6", "ASFODQR", "AZIK4ZA", "AYWIEKR", "AIFDJZB", "A1ZJ41O", "A35BJNV", "AOGFRXL", "AFHOHOM"]
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
        # hr_data = pd.read_csv(
        #     "/home/gustavo/PibicData1/COVID-19-Wearables/" + participant + "_hr.csv")
        # steps_data = pd.read_csv(
        #     "/home/gustavo/PibicData1/COVID-19-Wearables/" + participant + "_steps.csv")
        hr_data = pd.read_csv(
            "/mnt/d/PIBIC/Data Final/" + participant + "_hr.csv")
        steps_data = pd.read_csv(
            "/mnt/d/PIBIC/Data Final/" + participant + "_steps.csv")

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

        # TODO
        dateList, pre_symptom_date, symptom_date, covid_date, recovery_date = slidingwindows.time_stamps(
            df_sick, participant)

        vector_lengthDays = 7
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

        # slidingFunctions.visualizationVetores(vetoresMinInp, 0, 10)

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

        # contar porcentagem de anomalias nos períodos doentes/sintomáticos e nos períodos de detectionwindow
        porcT, porcP = slidingFunctions.por(time_min_mean)
        quality["PorcT"] = porcT
        quality["PorcP"] = porcP

        if save_mode == "on":
            Pre_Processing.saving_df(
                time_min_mean, dir_path, "AnDetection VetoresSW")
            Pre_Processing.saving_df(
                quality, dir_path, "Qualidade dos Vetores")

        slidingFunctions.finalPlot_quality(
            scRHR, 'qmax', "Número de Amostras por Hora dos Dados", save_mode, participant, symptom_date, covid_date, detectionWindow, recovery_date, dir_path)

        Anomaly_Detection.finalPlot(time_min_mean, symptom_date, covid_date,
                                    recovery_date, detectionWindow, "Detecção de Anomalias Vetores Sliding Windows de 7 dias", save_mode, participant, "Vetor Sliding Windows", dir_path)

        # valores de frequência pré-filtrados:
        freq_health, freq_pre, freq_symp = slidingFunctions.anomaly_frequency(
            time_min_mean)

        # filtragem de anomalias que estão sozinhas em um intervalo de uma semana
        time_min_mean_filtered = slidingFunctions.filtragem(time_min_mean)

        # print os dados filtrados

        Anomaly_Detection.finalPlot(time_min_mean_filtered, symptom_date, covid_date,
                                    recovery_date, detectionWindow, "Detecção de Anomalias Vetores Sliding Windows de 7 dias Filtrados", save_mode, participant, "Vetor Sliding Windows", dir_path)

        # valores de frequência filtrados
        Ffreq_health, Ffreq_pre, Ffreq_symp = slidingFunctions.anomaly_frequency(
            time_min_mean_filtered)

        fileName = participant + ".txt"
        full_path = os.path.join(dir_path, fileName)

        if save_mode == "on":
            with open(full_path, 'w') as file:
                # write some text to the file
                file.write(str(freq_health) + '\n')
                file.write(str(Ffreq_health) + '\n')
                file.write(str(freq_pre) + '\n')
                file.write(str(Ffreq_pre) + '\n')
                file.write(str(freq_symp) + '\n')
                file.write(str(Ffreq_symp) + '\n')

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
