import tkinter as tk
from ctypes import windll
from tkinter import ttk
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load

windll.shcore.SetProcessDpiAwareness(1)


def main():
    root = tk.Tk()

    frame = ttk.Frame(root, padding=10)
    frame.grid()

    root.geometry("800x750")


LARGEFONT = ("Verdana", 35)


class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Seizure Detection")
        self.state("zoomed")

        container = ttk.Frame(self)
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for page in (PatientPage, DiagnosePage):
            frame = page(container, self)

            self.frames[page] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.patient_data = {
            "first_name": "",
            "last_name": "",
            "email": "",
            "birth_date": "",
        }
        self.show_frame(PatientPage)

    def show_frame(self, page: ttk.Frame):
        frame = self.frames[page]
        frame.tkraise()


class PatientPage(tk.Frame):
    def __init__(self, parent: ttk.Frame, controller: App):
        super().__init__(parent)
        self.controller = controller
        ttk.Label(self, text="First Name").grid(row=0, column=0)
        ttk.Label(self, text="Last Name").grid(row=1, column=0)
        ttk.Label(self, text="Email").grid(row=2, column=0)
        ttk.Label(self, text="Date of Birth").grid(row=3, column=0)
        self.first_name = ttk.Entry(self)
        self.first_name.grid(row=0, column=1)
        self.last_name = ttk.Entry(self)
        self.last_name.grid(row=1, column=1)
        self.email = ttk.Entry(self)
        self.email.grid(row=2, column=1)
        self.birth_date = ttk.Entry(self)
        self.birth_date.grid(row=3, column=1)

        button = ttk.Button(self, text="Submit", command=self.submit)
        button.grid(row=4, column=0)

        self.error_lab = ttk.Label(self)
        self.error_lab.grid(row=5, column=0)

    def submit(self):
        data = [
            self.first_name.get(),
            self.last_name.get(),
            self.email.get(),
            self.birth_date.get(),
        ]
        for field in data:
            if field is None or field == "":
                self.error_lab.config(text="Fill in all fields")
                return
        # for field in self.contr
        self.controller.patient_data["first_name"] = self.first_name.get()
        self.controller.patient_data["last_name"] = self.last_name.get()
        self.controller.patient_data["email"] = self.email.get()
        self.controller.patient_data["birth_date"] = self.birth_date.get()
        print(self.controller.patient_data)
        self.controller.show_frame(DiagnosePage)


class DiagnosePage(ttk.Frame):
    filename: str = None
    signal: pd.Series = None

    def __init__(self, parent: tk.Frame, _):
        super().__init__(parent)

        select_label = ttk.Label(self, text="Select Data File")
        select_label.grid(row=0, column=0)
        self.file_entry = ttk.Entry(self, text=self.filename)
        self.file_entry.grid(row=0, column=1)
        self.open = ttk.Button(self, text="open", command=self.read_file)
        self.open.grid(row=0, column=2)

        ttk.Button(self, text="Diagnose", command=self.predict).grid(row=1, column=0)
        self.diagnosis = ttk.Label(self, text="")
        self.diagnosis.grid(row=2, column=0)

    def read_file(self):
        self.filename = askopenfilename(title="Select file")
        self.file_entry.delete(0, tk.END)
        self.file_entry.insert(0, self.filename)
        self.signal = pd.read_csv(self.filename, index_col=0).squeeze()
        self.plot(self.signal, "019182")

    @staticmethod
    def plot(signal: pd.Series, idx: str) -> None:
        t = np.arange(0, 23, 23 / len(signal))
        plt.plot(t, signal, color="green")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"Patient {idx} EEG")
        plt.show()

    def predict(self) -> None:
        if self.signal is None:
            self.diagnosis.config(text="No signal selected")
            return

        pcas = [
            np.array([self.signal @ np.load(f".\\pca\\pca{i}.npy")])
            for i in range(2, 6)
        ]

        models = [load(f".\\models\\rf\\rf_1v{i}.joblib") for i in range(2, 6)]

        predictions = [model.predict(pca) for model, pca in zip(models, pcas)]
        # y_pred = sum(predictions) <= len(predictions) // 2
        # diagnosis = "Seizure" if y_pred else "No seizure"
        # diagnosis = "-".join(predictions)
        self.diagnosis.config(text=str(predictions))


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
