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

    root.title("Support Vector Machine - Classifier")
    root.geometry("800x750")


LARGEFONT = ("Verdana", 35)


class App(tk.Tk):
    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
        # __init__ function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)

        # creating a container
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # initializing frames to an empty array
        self.frames = {}

        # iterating through a tuple consisting
        # of the different page layouts
        for F in (PatientPage, DiagnosePage):
            frame = F(container, self)

            # initializing frame of that object from
            # startpage, page1, page2 respectively with
            # for loop
            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(DiagnosePage)

    # to display the current frame passed as
    # parameter
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class PatientPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # label of frame Layout 2
        label = ttk.Label(self, text="PatientPage", font=LARGEFONT)

        # putting the grid in its
        # grid
        label.grid(row=0, column=4, padx=10, pady=10)

        button = ttk.Button(
            self,
            text="DiagnosePage",
            command=lambda: controller.show_frame(DiagnosePage),
        )

        # putting the button in its place by
        # using grid
        button.grid(row=1, column=1, padx=10, pady=10)


class DiagnosePage(tk.Frame):
    def __init__(self, parent, controller):
        self.filename: str = None
        self.signal: pd.Series = None

        tk.Frame.__init__(self, parent)

        select_label = ttk.Label(self, text="Select Data File")
        select_label.grid(row=0, column=0)
        self.file_entry = ttk.Entry(self, text=self.filename)
        self.file_entry.grid(row=0, column=1)
        self.open = ttk.Button(self, text="open", command=self.read_file)
        self.open.grid(row=0, column=2)

        tk.Button(self, text="Diagnose", command=self.predict).grid(row=1, column=0)
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
        plt.show(block=False)

    def predict(self) -> int:
        if self.signal is None:
            self.diagnosis.config(text="No signal selected")
            return

        pcas = [
            np.array([self.signal @ np.load(f".\\pca\\pca{i}.npy")])
            for i in range(2, 6)
        ]

        models = [load(f".\\models\\rf\\rf_1v{i}.joblib") for i in range(2, 6)]

        predictions = [model.predict(pca) for model, pca in zip(models, pcas)]
        y_pred = sum(predictions) <= len(predictions) // 2
        diagnosis = "Seizure" if y_pred else "No seizure"
        self.diagnosis.config(text=diagnosis)


if __name__ == "__main__":
    # Driver Code
    app = App()
    app.mainloop()
