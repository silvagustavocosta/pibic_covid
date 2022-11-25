from time import time
import Pre_Processing
import Anomaly_Detection
import Test_Functions
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import datetime


def data_org(participant, minutesRHR, scRHR, df_sick):
    """
        Organiza os vetores minutesRHR e scRHR da forma necessária para aplicar o isolation
        forest em cada um dos casos.
        Separa minutesRHR em diferentes vetores transladados, inputa dados e retorna sua média
    """

    # organizing index of dataframe:
    minutesRHR.rename(columns={"Unnamed: 0": "datetime"}, inplace=True)
    scRHR.rename(columns={"Unnamed: 0": "datetime"}, inplace=True)
    minutesRHR = Pre_Processing.organize_dataframe(minutesRHR)
    scRHR = Pre_Processing.organize_dataframe(scRHR)

    time_min_vetores = Anomaly_Detection.time_separation(minutesRHR)

    # apagar todos os vetores de time_min_vetores que estejam duplicados
    time_min_vetores = [time_min_vetores[x] for x, _ in enumerate(
        time_min_vetores) if time_min_vetores[x].equals(time_min_vetores[x-1]) is False]

    time_min_vetores, qualidade = Anomaly_Detection.quality(time_min_vetores)

    sick_id, dateList = Anomaly_Detection.sick_min(
        df_sick, time_min_vetores, participant)

    time_min_inp = Anomaly_Detection.input_data(time_min_vetores, 60)

    time_min_org = Anomaly_Detection.organize_data(time_min_inp)

    # plot visualization vetores:
    # i = 70
    # while i < 80:
    #     Pre_Processing.plot(time_min_inp[i], 1, "scatter")
    #     print(len(time_min_inp[i]))
    #     print(sick_id[i])
    #     i += 1

    # ISOLATION FOREST scRHR:
    # TODO:
    # scRHR = Anomaly_Detection.sick_hour(df_sick, scRHR, participant)

    return time_min_org, scRHR, sick_id, dateList


def isolation_analysis(vetoresRHR, scRHR, sick_id, df_sick, participant, dateList):
    """
        Utiliza os dados trabalhados para realizar as análises do isolation forest tanto
        para vetoresRHR e scRHR
    """

    for data in dateList:
        if data["status"] == 1:
            symptom_date = data["date"]
        elif data["status"] == 2:
            covid_date = data["date"]
        elif data["status"] == 0:
            recovery_date = data["date"]
        elif data["status"] == 3:
            pre_symptom_date = data["date"]

    # ISOLATION FOREST vetores:
    vetoresRHR, n_anomaly = Anomaly_Detection.isolation_forestMin(
        vetoresRHR, 0.02)

    time_min_mean = vetoresRHR.copy()
    time_min_mean['heartrate'] = time_min_mean['heartrate'].apply(np.mean)

    time_min_mean['sick_ID'] = sick_id

    Anomaly_Detection.plot_anomaly(
        time_min_mean, symptom_date, covid_date, recovery_date, pre_symptom_date, "Vetores")

    # ISOLATION FOREST scRHR:
    # scRHR = Anomaly_Detection.isolation_forestHOUR(scRHR)

    # Anomaly_Detection.plot_anomaly(
    #     scRHR, symptom_date, covid_date, recovery_date, "RHR Hora")


def cont_var(vetoresRHR, sick_id):
    """
        Variando a contaminação de 0 até certo valor para observar como a variação desse parâmetro afeta o
        resultado esperado pelo isolation forest
    """

    # Variando o fator de contaminação para testar os efeitos sobre o isolation forest:
    cont_para = Test_Functions.get_contamination(0.005, 0.15, 0.005)
    total_anomaly, true_anomaly, porc_anomaly = Test_Functions.var_contamination(vetoresRHR,
                                                                                 cont_para, sick_id)

    return total_anomaly, true_anomaly, porc_anomaly, cont_para


def main():
    # reading pre_processed data:
    mode = "full"

    Supplementary_Table = pd.read_csv(
        "/home/gustavo/PibicData1/Sick_Values_01.txt")
    if mode == "solo":
        subjects = []
        subjects.append("APGIB2T")
    elif mode == "full":
        # subjects = Supplementary_Table.ParticipantID.values.tolist()
        subjects = ["AFPB8J2", "APGIB2T", "A0NVTRV", "A4G0044",
                    "AS2MVDL", "ASFODQR", "AYWIEKR", "AJMQUVV"]

    y = []
    z = []
    for participant in subjects:
        minutesRHR = pd.read_csv(
            "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Data/" + participant + "/minutesRHR")
        scRHR = pd.read_csv(
            "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Data/" + participant + "/scRHR")

        # getting timestamps from when the subject is sick:
        df_sick = pd.read_csv(
            "/home/gustavo/PibicData1/Sick_Values_01.txt")

        vetoresRHR, scRHR, sick_id, dateList = data_org(
            participant, minutesRHR, scRHR, df_sick)

        # Variação da contaminação:
        total_anomaly, true_anomaly, porc_anomaly, cont_para = cont_var(
            vetoresRHR, sick_id)
        y.append(porc_anomaly)
        z.append(total_anomaly)

        # isolation_analysis(vetoresRHR, scRHR, sick_id,
        #                    df_sick, participant, dateList)

    # Variação da contaminação (plots):
    fig, ax = plt.subplots()

    ax.plot(cont_para, z[0], label="AFPB8J2")
    ax.plot(cont_para, z[1], label="APGIB2T")
    ax.plot(cont_para, z[2], label="A0NVTRV")
    ax.plot(cont_para, z[3], label="A4G0044")
    ax.plot(cont_para, z[4], label="AS2MVDL")
    ax.plot(cont_para, z[5], label="ASFODQR")
    ax.plot(cont_para, z[6], label="AYWIEKR")
    ax.plot(cont_para, z[7], label="AJMQUVV")

    plt.xlabel("Contaminação")
    plt.ylabel("Número Total de Anomalias")
    plt.gcf().set_size_inches(8, 6)
    plt.title("Variação da Contaminação")
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
