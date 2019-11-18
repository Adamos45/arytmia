from symbols_enums import FeatureNames
from feature_extraction import prepare_data, read_data
from train_models import train_models
from program_exploitation import program_exploitation
import pandas
import tkinter as tk
import time


def main():
    # files = [FeatureNames.mvsk, FeatureNames.db5,
    #          FeatureNames.db6, FeatureNames.diffs]
    # prepare_data(files)
    # data = read_data(files)
    # train_models(data, files)

    def get_record(record):
        if not record.endswith(".csv"):
            record += ".csv"
        try:
            return pandas.read_csv(record, usecols=[1]).values.T[0]
        except FileNotFoundError:
            return None

    window = tk.Tk()
    window.title("Wykrywanie arytmii serca")
    text_count = tk.StringVar()
    text_indx = tk.StringVar()
    label1 = tk.Label(window, textvariable=text_count).pack()
    label2 = tk.Label(window, textvariable=text_indx).pack()
    name = tk.Entry(window, width=40)
    name.pack()

    def exploitation(record):
        if record is not None:
            text_indx.set("")
            text_count.set("Proszę czekać")
            window.update()
            predictions = program_exploitation(record)
            predictions_str = "N: " + str(predictions.count(0)) + "\nS: " + str(predictions.count(1)) + "\nV: "\
                              + str(predictions.count(2)) + "\nF: " + str(predictions.count(3))
            arrythmias = [[], [], []]
            for i, p in enumerate(predictions):
                if p == 1:
                    arrythmias[0].append(i)
                elif p == 2:
                    arrythmias[1].append(i)
                elif p == 3:
                    arrythmias[2].append(i)
            indexes = ""
            for i, a in enumerate(arrythmias):
                if i == 0 and len(a):
                    indexes += "\n\nCzęstoskurcz nadkomorowy"
                elif i == 1 and len(a):
                    indexes += "\n\nCzęstoskurcz komorowy"
                elif i == 2 and len(a):
                    indexes += "\n\nPobudzenie zsumowane"
                for it, i in enumerate(a):
                    if it % 26 == 0:
                        indexes += "\n"
                    indexes += str(i+1) + " "
                indexes += "\n"
            text_indx.set(indexes)
            text_count.set(predictions_str)
        else:
            text_count.set("Nie odnaleziono rekordu")

    btn1 = tk.Button(window, text="Rozpocznij wykrywanie", command=lambda: exploitation(get_record(name.get())))
    btn1.pack()
    btn2 = tk.Button(window, text="Zakończ", command=lambda: window.quit())
    btn2.pack()
    tk.mainloop()


if __name__ == "__main__":
    main()
