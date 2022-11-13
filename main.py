from time import time
import Pre_Processing
import Anomaly_Detection
import Test_Functions
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def isolation_analysis(participant, minutesRHR, scRHR, df_sick):
    symptom_date, covid_date, recovery_date = Pre_Processing.get_sick_time(
        df_sick, participant)

    # organizing index of dataframe:
    minutesRHR.rename(columns={"Unnamed: 0": "datetime"}, inplace=True)
    scRHR.rename(columns={"Unnamed: 0": "datetime"}, inplace=True)
    minutesRHR = Pre_Processing.organize_dataframe(minutesRHR)
    scRHR = Pre_Processing.organize_dataframe(scRHR)

    # ISOLATION FOREST minutesRHR:
    time_min_vetores = Anomaly_Detection.time_separation(minutesRHR)

    # apagar todos os vetores de time_min_vetores que estejam duplicados
    time_min_vetores = [time_min_vetores[x] for x, _ in enumerate(
        time_min_vetores) if time_min_vetores[x].equals(time_min_vetores[x-1]) is False]

    time_min_vetores, qualidade = Anomaly_Detection.quality(time_min_vetores)

    sick_id = Anomaly_Detection.sick_min(df_sick, time_min_vetores, participant)

    time_min_inp = Anomaly_Detection.input_data(time_min_vetores, 60)

    time_min_org = Anomaly_Detection.organize_data(time_min_inp)

    # Variando o fator de contaminação para testar os efeitos sobre o isolation forest:
    # cont_para = Test_Functions.get_contamination(0.005, 0.15, 0.005)

    # total_anomaly, true_anomaly, porc_anomaly = Test_Functions.var_contamination(time_min_org,
    #     cont_para, sick_id)

    time_min_org, n_anomaly = Anomaly_Detection.isolation_forestMin(
        time_min_org, 0.11)

    time_min_mean = time_min_org.copy()
    time_min_mean['heartrate'] = time_min_mean['heartrate'].apply(np.mean)

    time_min_mean['sick_ID'] = sick_id

    Anomaly_Detection.plot_anomaly(
         time_min_mean, symptom_date, covid_date, recovery_date, "Vetores")

    # plot visualization:
    # i = 70
    # while i < 80:
    #     Pre_Processing.plot(time_min_inp[i], 1, "scatter")
    #     print(len(time_min_inp[i]))
    #     print(sick_id[i])
    #     i += 1

    # ISOLATION FOREST scRHR:
    # TODO:
    # scRHR = Anomaly_Detection.sick_hour(df_sick, scRHR, participant)

    # scRHR = Anomaly_Detection.isolation_forestHOUR(scRHR)

    # Anomaly_Detection.plot_anomaly(
    #     scRHR, symptom_date, covid_date, recovery_date, "RHR Hora")

    return total_anomaly, true_anomaly, porc_anomaly, cont_para

def main():
    # reading pre_processed data:
    mode = "full"

    Supplementary_Table = pd.read_csv(
        "/mnt/c/Users/silva/Downloads/COVID-19-Wearables/Sick_Values_01.csv")
    if mode == "solo":
        subjects = []
        subjects.append("ALKAXMZ")
    elif mode == "full":
        #subjects = Supplementary_Table.ParticipantID.values.tolist()
        subjects = ["AFPB8J2", "APGIB2T", "A0NVTRV", "A4G0044", 
            "AS2MVDL", "ASFODQR", "AZIK4ZA", "AYWIEKR", "AJMQUVV"]

    y = []
    z = []
    for participant in subjects:
        minutesRHR = pd.read_csv(
            "/mnt/c/Users/silva/Downloads/COVID-19-Wearables/Data/" + participant + "/minutesRHR")
        scRHR = pd.read_csv(
            "/mnt/c/Users/silva/Downloads/COVID-19-Wearables/Data/" + participant + "/scRHR")

        # getting timestamps from when the subject is sick:
        df_sick = pd.read_csv("/mnt/c/Users/silva/Downloads/COVID-19-Wearables/Sick_Values_01.csv")

        total_anomaly, true_anomaly, porc_anomaly, x = isolation_analysis(participant, minutesRHR, scRHR, df_sick)
        y.append(porc_anomaly)
        z.append(total_anomaly)

    fig, ax = plt.subplots()
    ax.plot(x, z[0], label = "AFPB8J2")
    ax.plot(x, z[1], label = "A0NVTRV")
    ax.plot(x, z[2], label = "A4G0044")
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
