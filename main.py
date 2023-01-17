from time import time
import Pre_Processing
import Anomaly_Detection
import Test_Functions
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import datetime


def input(df):
    """
        Inputa no dataframe um valor sempre que possuir um index vazio de minuto. Todos os minutos do dataframe
        serão preenchidos com valores interpolados ou calculados a partir de outras funções. Sets de dados que
        possuem muitas lacunas não devem ser trabalhadas
    """

    # minutesRHR_inp = df

    # detectar arquivos que podem ou não ser inputados:
    # inputa os dados no arquivo:
    minutesRHR_inp, totalLen, dfLen, lengths_consecutive_na = Anomaly_Detection.number_of_inputs(
        df)

    # TODO
    # Teste de qualidade utilizando lengths_consecutive_na

    # print(len(minutesRHR_inp))

    # zoom no dataframe
    # minutesRHR_inp = Anomaly_Detection.zoomdf(
    #     minutesRHR_inp, "2025-03-01 00:00:00", "2025-03-15 00:00:00")

    # ploting
    # fig, ax = plt.subplots()
    # ax.scatter(minutesRHR_inp.index,
    #            minutesRHR_inp["heartrate"], label="RHR Inputed", marker=".", s=1)
    # plt.gcf().set_size_inches(12, 10)
    # plt.title("RHR Após Inputar Missing Values")
    # plt.gcf().autofmt_xdate()
    # plt.tight_layout()
    # plt.legend()
    # plt.show()

    return minutesRHR_inp


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

    minutesRHR_inp = input(minutesRHR)

    time_min_vetores = Anomaly_Detection.time_separation(minutesRHR_inp)

    # apagar todos os vetores de time_min_vetores que estejam duplicados
    time_min_vetores = [time_min_vetores[x] for x, _ in enumerate(
        time_min_vetores) if time_min_vetores[x].equals(time_min_vetores[x-1]) is False]

    time_min_vetores, qualidade = Anomaly_Detection.quality(time_min_vetores)

    sick_id, dateList = Anomaly_Detection.sick_min(
        df_sick, time_min_vetores, participant)

    time_min_inp = Anomaly_Detection.input_data(time_min_vetores, 60)

    time_min_org = Anomaly_Detection.organize_data(time_min_inp)

    # plot visualization vetores:
    # i = 120
    # while i < 130:
    #     Pre_Processing.plot(time_min_inp[i], 1, "scatter")
    #     print(len(time_min_inp[i]))
    #     print(sick_id[i])
    #     i += 1

    # ISOLATION FOREST scRHR
    sick_HID = Anomaly_Detection.sick_hour(df_sick, scRHR, participant)

    return time_min_org, scRHR, sick_id, dateList, sick_HID


def dateListSeparation(dateList):
    """
        Separa todos os valores dentro de dateList em symptom_date, covid_date, recovery_date e
        pre_symptom_date
    """

    # TODO:
    # esse método só vai levar em consideração uma data de cada tipo de análise, só vai carregar a última das datas
    # criar método que consiga pegar o período completo, todos as datas de certa categoria, além de conseguir carregar
    # variáveis vazias caso a data em específico não exista

    # associar todos os períodos de doença com o que está guardado no dateList
    pre_symptom_date = []
    symptom_date = []
    covid_date = []
    recovery_date = []

    if len(dateList) != 0:
        for data in dateList:
            if data["status"] == 1:
                symptom_date.append(data["date"])
            elif data["status"] == 2:
                covid_date.append(data["date"])
            elif data["status"] == 0:
                recovery_date.append(data["date"])
            elif data["status"] == 3:
                pre_symptom_date.append(data["date"])

    if len(pre_symptom_date) == 0:
        pre_symptom_date = None
    if len(symptom_date) == 0:
        symptom_date = None
    if len(covid_date) == 0:
        covid_date = None
    if len(recovery_date) == 0:
        recovery_date = None

    return pre_symptom_date, symptom_date, covid_date, recovery_date


def isolation_analysis(vetoresRHR, scRHR, sick_id, df_sick, participant, dateList, sick_HID, save_mode, vetoresPorT, vetoresPorP, horasPorT, horasPorP):
    """
        Utiliza os dados trabalhados para realizar as análises do isolation forest tanto
        para vetoresRHR e scRHR
    """

    pre_symptom_date, symptom_date, covid_date, recovery_date = dateListSeparation(
        dateList)

    # ISOLATION FOREST vetores:
    vetoresRHR, n_anomaly = Anomaly_Detection.isolation_forestMin(
        vetoresRHR, 0.02)

    time_min_mean = vetoresRHR.copy()
    time_min_mean['heartrate'] = time_min_mean['heartrate'].apply(np.mean)

    time_min_mean['sick_ID'] = sick_id

    scRHR['sick_ID'] = sick_HID

    # ploting vetoresRHR, dataframe dos minutesRHR organizado em vetores
    Anomaly_Detection.ploting(time_min_mean, pre_symptom_date,
                              symptom_date, covid_date, recovery_date, "vetoresRHR", "heartrate", save_mode, participant)

    Anomaly_Detection.plot_anomaly(
        time_min_mean, symptom_date, covid_date, recovery_date, pre_symptom_date, "Vetores", save_mode, participant)

    # calcular a porcentagem das anomalias
    porcT, porcP = cont_porc(time_min_mean)
    vetoresPorT.append(porcT)
    vetoresPorP.append(porcP)
    print("Porcentagem de anomalias detectadas corretamente: ", porcT)
    print("Porcentagem de anomalias detectadas no período Pré-Sintomático: ", porcP)
    # adicionar os valores no controle

    # ISOLATION FOREST scRHR:
    scRHR = Anomaly_Detection.isolation_forestHOUR(scRHR)

    Anomaly_Detection.plot_anomaly(
        scRHR, symptom_date, covid_date, recovery_date, pre_symptom_date, "RHR Hora", save_mode, participant)

    porcT, porcP = cont_porc(scRHR)
    horasPorT.append(porcT)
    horasPorP.append(porcP)
    print("Porcentagem de anomalias detectadas corretamente: ", porcT)
    print("Porcentagem de anomalias detectadas no período Pré-Sintomático: ", porcP)

    return vetoresPorT, vetoresPorP, horasPorT, horasPorP


def cont_porc(df):
    """
        Calcula a porcentagem de anomalias detectadas no período correto de análise, além de calcular a porcentagem
        de anomalias detectadas no período pré-sintomático.
    """

    nAno = df.loc[(df["anomaly"] == -1)]
    nSic = df.loc[(((df["anomaly"] == -1) & (df["sick_ID"] == 1)) |
                   ((df["anomaly"] == -1) & (df["sick_ID"] == 2)) | ((df["anomaly"] == -1) & (df["sick_ID"] == 3)))]
    porcT = (len(nSic)/len(nAno))*100

    nPre = df.loc[((df["anomaly"] == -1) & (df["sick_ID"] == 3))]
    porcP = (len(nPre)/len(nAno))*100

    return porcT, porcP


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
    mode = "solo"
    save_mode = "off"

    Supplementary_Table = pd.read_csv(
        "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Input/Sick_Values_01.txt")

    controle = pd.read_csv(
        "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Data/controle")
    del controle["Unnamed: 0"]

    if mode == "solo":
        subjects = []
        subjects.append("AS2MVDL")
    elif mode == "full":
        subjects = Supplementary_Table.ParticipantID.values.tolist()
        # test patients:
        # subjects = ["AFPB8J2", "APGIB2T", "A0NVTRV", "A4G0044",
        #             "AS2MVDL", "ASFODQR", "AYWIEKR", "AJMQUVV"]

    # criar listas que vão acompanhar os valores do controle
    vetoresPorT = []
    vetoresPorP = []
    horasPorT = []
    horasPorP = []

    for participant in subjects:
        print(participant)

        minutesRHR = pd.read_csv(
            "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Data/" + participant + "/minutesRHR")
        scRHR = pd.read_csv(
            "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Data/" + participant + "/scRHR")

        # getting timestamps from when the subject is sick:
        df_sick = pd.read_csv(
            "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Input/Sick_Values_01.txt")

        vetoresRHR, scRHR, sick_id, dateList, sick_HID = data_org(
            participant, minutesRHR, scRHR, df_sick)

        vetoresPorT, vetoresPorP, horasPorT, horasPorP = isolation_analysis(vetoresRHR, scRHR, sick_id,
                                                                            df_sick, participant, dateList, sick_HID, save_mode, vetoresPorT, vetoresPorP, horasPorT, horasPorP)

    controle["vetoresPorT"] = vetoresPorT
    controle["vetoresPorP"] = vetoresPorP
    controle["horasPorT"] = horasPorT
    controle["horasPorP"] = horasPorP

    if save_mode == "on":
        base_path = "/mnt/c/Users/silva/Desktop/Gustavo/Pibic/Data"
        Pre_Processing.saving_df(controle, base_path, "controleAnalysis")

    print(controle)


if __name__ == '__main__':
    main()
