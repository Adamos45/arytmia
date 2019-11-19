from symbols_enums import FeatureNames
from feature_extraction import prepare_data, read_data
from train_models import train_models
from program_exploitation import program_exploitation
import wfdb
import numpy as np
from tkinter import *
from tkinter.filedialog import askopenfilename, askopenfilenames


def main():

    gui()


def gui():
    record = []
    models = []
    window = Tk()
    window.title("Wykrywanie arytmii serca")
    text_count = StringVar()
    text_indx = StringVar()
    text_record = StringVar(value="Załadowany plik: Brak")
    Label(window, text="Program do wykrywania arytmii serca na podstawie zapisu EKG").pack(pady=30, padx=20)
    Button(window, text="Rozpocznij ewaluację", command=lambda: evaluate()).pack()
    Button(window, text="Trenuj model", command=lambda: train()).pack()
    Button(window, text="Instrukcja obługi", command=lambda: general_help()).pack()
    Button(window, text="Zakończ", command=lambda: window.destroy()).pack(pady=30)

    def eval_read_record(record):
        window.update()
        record_name = askopenfilename(title="Proszę wybrać plik z rekordem",
                                      filetypes=[("WFDB", ".dat")])
        if len(record_name) == 0:
            return
        extension_split = str.rsplit(record_name, '.', 1)
        try:
            read = wfdb.rdrecord(extension_split[0], channels=[0]).p_signal[:, 0]
        except FileNotFoundError:
            text_indx.set("Nie znaleziono pliku" + extension_split[0] + ".hea")
            return
        record.clear()
        record.extend(read)
        text_record.set("Załadowany plik: " + record_name)
        window.update()

    def train_read_records(record):
        window.update()
        record_names = askopenfilenames(title="Proszę wybrać pliki z rekordami",
                                      filetypes=[("WFDB", ".dat")])
        if len(record_names) == 0:
            return
        record.clear()
        record_names_short = []
        for record_name in record_names:
            extension_split = str.rsplit(record_name, '.', 1)
            try:
                baseline = wfdb.rdrecord(extension_split[0], channels=[0]).p_signal[:, 0]
                annotation = wfdb.rdann(extension_split[0], 'atr')
                record.append([baseline, annotation])
                record_names_short.append(str.rsplit(extension_split[0], '/', 1)[1])
            except FileNotFoundError:
                text_indx.set("Nie znaleziono pliku" + extension_split[0] + ".atr")
                record.clear()
                return
        text_record.set("Wczytano pliki: "+str.join(".sav, ", record_names_short)+".sav")
        window.update()

    def exploitation(record, train=False, classifier_choice=0, feature_choice=None):
        if len(record) > 0:
            text_indx.set("")
            record = np.array(record)
            text_count.set("Proszę czekać")
            window.update()
            if not train:
                record_number = str.rsplit(text_record.get(), '/', maxsplit=1)[1]
                record_number = str.rsplit(record_number, '.', maxsplit=1)[0]
            else:
                record_number = str.split(text_record.get(), ',')
                record_number[0] = str.rsplit(record_number[0], ' ', maxsplit=1)[1]
                for i, r in enumerate(record_number):
                    record_number[i] = r.strip(' ')
            predictions = program_exploitation(record, record_number, train, model_paths=models,
                                               classifier=classifier_choice, feature_choice=feature_choice)
            if train:
                text_count.set("Zakończono trening.")
                return
            try:
                predictions_str = "Sumy pobudzeń zliczone z całego rekodu dla danych klas:\n" \
                                  "Ilość pobudzeń normalnych: " + str(predictions.count(0)) \
                                  + "\nIlość pobudzeń z wykrytym ektoskopowym pobudzeniem przedsionkowym: " \
                                  + str(predictions.count(1)) \
                                  + "\nIlość pobudzeń z wykrytym ektoskopowe pobudzeniem komorowym: " \
                                  + str(predictions.count(2)) \
                                  + "\nIlość pobudzeń z wykrytą fuzją pobudzenia normalnego i arytmicznego: " \
                                  + str(predictions.count(3))
            except:
                text_count.set("Za mało danych")
                return
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
                    indexes += "\n\nNumery pobudzeń z wykrytym ektoskopowym pobudzeniem przedsionkowym:"
                elif i == 1 and len(a):
                    indexes += "\n\nNumery pobudzeń z wykrytym ektoskopowe pobudzeniem komorowym"
                elif i == 2 and len(a):
                    indexes += "\n\nNumery pobudzeń z wykrytą fuzją pobudzenia normalnego i arytmicznego"
                for it, i in enumerate(a):
                    if it % 26 == 0:
                        indexes += "\n"
                    indexes += str(i + 1) + " "
                indexes += "\n"
            text_indx.set(indexes)
            text_count.set(predictions_str)
        else:
            text_indx.set("Nie wczytano żadnego rekordu")

    def general_help():
        help_window = Tk()
        help_window.title("Instrukcja obsługi")
        Label(help_window, text="Program oferuje dwie funkcjonalności związane z wykrywaniem"
                                "arytmii na podstawie sygnału EKG.\n"
                                "Pierwsza to wykrywanie arytmii, korzystając "
                                "z modeli wytrenowanych na zbiorze MIT-BIH.\n"
                                "Druga funkcjonalność umożliwia użytkownikowi samodzielne "
                                "trenowanie własnych modeli. Można ich potem użyć w tym programie "
                                "w ramach do wykrywania arytmii w innych rekordach.\n\nPrzyciski:\n"
                                "1. Rozpocznij ewaluację. Zacznij wykrywać arytmię w wybranych sygnałach EKG\n"
                                "2. Trenuj model. Wybierz jeden lub więcej sygnałów EKG i zaczynij uczyć modele")\
            .pack(padx=20, pady=20)
        Button(help_window, text="Ok", command=lambda: help_window.destroy()).pack(pady=20)

    def evaluate():
        window.withdraw()
        sub_window = Toplevel()
        label_text = Label(sub_window, textvariable=text_record)
        label_text.pack(padx=20)
        label_indx = Label(sub_window, textvariable=text_indx)
        label_indx.pack()
        Label(sub_window, textvariable=text_count).pack()
        Button(sub_window, text="Wczytaj rekord", command=lambda: eval_read_record(record)).pack()
        Button(sub_window, text="Wybierz własne modele", command=lambda: choose_models()).pack()
        Button(sub_window, text="Rozpocznij wykrywanie", command=lambda: exploitation(record)).pack()
        Button(sub_window, text="Objaśnienie wyników", command=lambda: detection_help()).pack()
        Button(sub_window, text="Pomoc: wczytywanie danych", command=lambda: evaluate_help()).pack()
        Button(sub_window, text="Menu główne", command=lambda: terminate()).pack(pady=30)

        def detection_help():
            help_window = Tk()
            help_window.title("Objaśnienie wyników")
            Label(help_window,
                  text="Na początku wyświetlone są numery kolejności udzerzeń serca we wczytanym rekordzie, "
                       "które odpowiadają jednej z trzech kategorii pobudzeń "
                       "arytmicznych:\n    1.Ektoskopowe pobudzenie przedsionkowe\n"
                       "   2.Ektoskopowe pobudzenie komorowe\n"
                       "   3.Fuzja pobudzenia normalnego i arytmicznego\n"
                       "Pobudzenia normalne nie są wyświetlane\n\n "
                       "Niżej znajdują się zsumowane wartości wszystkich "
                       "czterech kategorii").pack(padx=20, pady=20)
            Button(help_window, text="Ok", command=lambda: help_window.destroy()).pack(pady=20)

        def evaluate_help():
            help_window = Tk()
            help_window.title("Instrukcja obsługi")
            Label(help_window, text="Przyciski:\n" +
                                    "1.Wczytaj rekord. Akcpetowane są tylko pliki z rozszerzeniem WFDB (.dat).\n"
                                    "W tym samym katalogu z wybranym plikiem musi znajdować się plik towarzyszący"
                                    " o tej samej nazwie z rozszerzeniem .hea.\n"
                                    "2.Wybierz własne modele. Domyślnie są używane modele wytrenowane na zbiorze"
                                    " MIT-BIH. Zamiast nich można użyć własnych modeli dostarczonych z zewnątrz lub\n"
                                    "wytrenowanych w tym programie za pomocą drugiej opcji z menu głównego.\nPliki "
                                    "modeli musza być z rozszerzeniem (.sav)\n"
                                    "3.Rozpocznij wykrywanie. Przycisk uruchamiający zadanie wykrywania arytmii.\n"
                                    "Wymaga wczytanego pliku i ewentualnych wczytanych własnych modeli.\n"
                                    "Pomoc dotycząca odczytywania wygenerowanych wyników jest w przycisku: "
                                    "Objaśnienie wyników\n"
                                    "4.Objaśnienie wyników, pomoc dotycząca czytania danych wygenerowanych za pomocą"
                                    "przycisku powyżej").pack(padx=20, pady=20)
            Button(help_window, text="Ok", command=lambda: help_window.destroy()).pack(pady=20)

        def terminate():
            text_indx.set("")
            text_count.set("")
            text_indx.set("")
            window.deiconify()
            sub_window.destroy()

    def choose_models():
        record_names = askopenfilenames(title="Proszę wybrać pliki z modelami",
                                        filetypes=[("SAV", ".sav")])
        for record_name in record_names:
            models.append(record_name)
        text_indx.set("Załadowane modele: \n" + str.join("\n", record_names))

    def train():
        window.withdraw()
        sub_window = Toplevel()
        label_text = Label(sub_window, textvariable=text_record)
        label_text.pack(padx=20)
        label_indx = Label(sub_window, textvariable=text_indx)
        label_indx.pack()
        classifier = IntVar()
        f1 = IntVar()
        f2 = IntVar()
        f3 = IntVar()
        f4 = IntVar()
        f5 = IntVar()
        f6 = IntVar()
        f7 = IntVar()
        f8 = IntVar()
        feature_choice = np.array(np.zeros(8))
        Label(sub_window, textvariable=text_count).pack()
        Button(sub_window, text="Wczytaj rekordy treningowe", command=lambda: train_read_records(record)).pack()
        Button(sub_window, text="Rozpocznij trenowanie", command=lambda: do_train()).pack()
        Radiobutton(sub_window, text="Perceptron wielowarswowy", variable=classifier, value=0).pack()
        Radiobutton(sub_window, text="K najbliższych sąsiadów", variable=classifier, value=1).pack()
        Checkbutton(sub_window, text="(Średnia, wariancja, skośność, kurtoza)", variable=f1).pack()
        Checkbutton(sub_window, text="(Wariancja, skośność, kurtoza)", variable=f2).pack()
        Checkbutton(sub_window, text="Falki Daubechies 5", variable=f3).pack()
        Checkbutton(sub_window, text="Falki Daubechies 6", variable=f4).pack()
        Checkbutton(sub_window, text="Interwały R-R", variable=f5).pack()
        Checkbutton(sub_window, text="Bez ekstrakcji 90 próbek", variable=f6).pack()
        Checkbutton(sub_window, text="Bez ekstrakcji 45 próbek", variable=f7).pack()
        Checkbutton(sub_window, text="Widmowa gęstość mocy", variable=f8).pack()

        def do_train():
            cbuttons()
            exploitation(record, True,classifier_choice=classifier.get(),feature_choice=feature_choice)

        def cbuttons():
            feature_choice[0] = f1.get()
            feature_choice[1] = f2.get()
            feature_choice[2] = f3.get()
            feature_choice[3] = f4.get()
            feature_choice[4] = f5.get()
            feature_choice[5] = f6.get()
            feature_choice[6] = f7.get()
            feature_choice[7] = f8.get()
        Button(sub_window, text="Pomoc: trenowanie", command=lambda: training_help()).pack()
        Button(sub_window, text="Menu główne", command=lambda: terminate()).pack(pady=30)

        def training_help():
            help_window = Tk()
            help_window.title("Objaśnienie trenowania")
            Label(help_window,
                  text="Aplikacja oferuje możliwość trenowania modeli na własnych zbiorach danych.\n"
                       "Przyciski:\n"
                       "1.Wczytaj rekordy treningowe. Wymagany format plików z rekordami to WFDB (.dat)\n"
                       "oraz plik towarzyszący, posiadający etykiety, z roszerzeniem (.atr)\n"
                       "2.Rozpocznij trenowanie. Wygenerowane modele są zapisywane w postaci plików (.sav)\n"
                       "do katalogu ./trained_models_gui/\n"
                       "Dodatkowo istnieje możliwość wyboru klasyfikatora (MLP lub kNN)\n"
                       "Z listy należy również wybrać deskryptory cech.").pack(padx=20, pady=20)
            Button(help_window, text="Ok", command=lambda: help_window.destroy()).pack(pady=20)

        def terminate():
            text_record.set("")
            text_count.set("")
            text_indx.set("")
            window.deiconify()
            sub_window.destroy()

    mainloop()


if __name__ == "__main__":
    main()
