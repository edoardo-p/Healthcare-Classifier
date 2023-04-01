import tkinter as tk
from ctypes import windll
from tkinter import ttk
from tkinter.filedialog import askopenfilename

import numpy as np
import pandas as pd
from joblib import load
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

windll.shcore.SetProcessDpiAwareness(1)

DAYS = list(range(1, 32))
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
YEARS = list(range(1900, 2023))


class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Seizure Detection")
        # self.state("zoomed")
        self.geometry("800x750")

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
        # self.show_frame(PatientPage)
        self.show_frame(DiagnosePage)

    def show_frame(self, page):
        frame = self.frames[page]
        frame.tkraise()


class PatientPage(ttk.Frame):
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

        self.day = tk.StringVar(self)
        self.month = tk.StringVar(self)
        self.year = tk.StringVar(self)

        day_combo = ttk.Combobox(self, textvariable=self.day)
        day_combo.grid(row=3, column=1)
        day_combo["values"] = DAYS
        month_combo = ttk.Combobox(self, textvariable=self.month)
        month_combo.grid(row=3, column=2)
        month_combo["values"] = MONTHS
        year_combo = ttk.Combobox(self, textvariable=self.year)
        year_combo.grid(row=3, column=3)
        year_combo["values"] = YEARS

        button = ttk.Button(self, text="Submit", command=self.submit)
        button.grid(row=4, column=0)

        self.error_lab = ttk.Label(self)
        self.error_lab.grid(row=5, column=0)

    def submit(self):
        birth_date = (
            f"{self.day.get()}/{MONTHS.index(self.month.get()) + 1}/{self.year.get()}"
        )
        data = [
            self.first_name.get(),
            self.last_name.get(),
            self.email.get(),
            birth_date,
        ]
        for field in data:
            if field is None or field == "":
                self.error_lab.config(text="Fill in all fields")
                return
        for field, val in zip(self.controller.patient_data, data):
            self.controller.patient_data[field] = val

        print(self.controller.patient_data)
        self.controller.show_frame(DiagnosePage)


class DiagnosePage(ttk.Frame):
    def __init__(self, parent: ttk.Frame, _):
        super().__init__(parent)

        select_label = ttk.Label(self, text="Select signal file")
        select_label.grid(row=0, column=0)
        self.file_entry = ttk.Entry(self)
        self.file_entry.grid(row=0, column=1)
        self.open = ttk.Button(self, text="open", command=self.read_file)
        self.open.grid(row=0, column=2)

        ttk.Button(self, text="Diagnose", command=self.predict).grid(row=3, column=0)
        self.diagnosis = ttk.Label(self, text="")
        self.diagnosis.grid(row=4, column=0)

    def read_file(self):
        self.filename = askopenfilename(title="Select file", initialdir=".")
        if self.filename is None or self.filename == "":
            return
        self.file_entry.delete(0, tk.END)
        self.file_entry.insert(0, self.filename)
        self.signal = pd.read_csv(self.filename, index_col=0).squeeze()
        self.plot(self.signal, "019182")

    def plot(self, signal: pd.Series, idx: str) -> None:
        # fig = Figure(figsize=(5, 4), dpi=100)
        fig = plt.figure(figsize=(5, 4), dpi=100)
        ax: plt.Axes = fig.add_subplot()
        t = np.arange(0, 23, 23.0 / len(signal))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Patient {idx} EEG")
        ax.plot(t, signal, "g-")

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, ipadx=40, ipady=20)

        toolbar_frame = ttk.Frame(self)
        toolbar_frame.grid(row=2, column=0)
        NavigationToolbar2Tk(canvas, toolbar_frame)

    def predict(self) -> None:
        if self.signal is None:
            self.diagnosis.config(text="No signal selected")
            return

        pcas = [
            np.array([self.signal @ np.load(f".\\pca\\pca{i}.npy")])
            for i in range(2, 6)
        ]

        models = [load(f".\\models\\rf\\1v{i}.joblib") for i in range(2, 6)]

        predictions = [model.predict(pca)[0] for model, pca in zip(models, pcas)]
        # y_pred = sum(predictions) <= len(predictions) // 2
        # diagnosis = "Seizure" if y_pred else "No seizure"
        # diagnosis = "-".join(predictions)
        self.diagnosis.config(text=str(predictions))


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
