from utils.log_parser import parse_log_file
from threading import Event
from threading import Thread
import os
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np


class Plot:
    def __init__(self, log_name) -> None:
        self.id = log_name
        self.session_log_name = f"/home/fedml/fedml-ng/logs/flotilla_{log_name}.log"
        self.colors = [
            [1.0, 0.31994636, 0.11065268, 1.0],
            [0.6627451, 0.8505867, 0.53165947, 1.0],
            [0.07254902, 0.88292761, 0.9005867, 1.0],
        ]
        plot_thread = Thread(target=self.plot)
        plot_thread.start()

    def plot(self) -> None:
        sleep = Event()
        while not sleep.is_set():
            df = parse_log_file(self.session_log_name)
            check_list = df["values"][
                df["message"] == "fedserver_session_finished_running"
            ].to_list()
            if len(check_list) >= 1:
                print("STOPPING PLOTTER")
                return
            sleep.wait(10)
            self.plot_log_vs_accuracy(df)

    def plot_log_vs_accuracy(self, df) -> None:
        results = df["values"][df["message"] == "fedserver.train_callback"].to_list()

        try:
            rounds = int(results[-1][3])
        except IndexError:
            return

        acc = list()
        loss = list()

        for rnd in results:
            acc.append(float(rnd[4]))
            loss.append(float(rnd[5]))

        if len(acc) < 2 or len(loss) < 2:
            return

        round_numbers = np.arange(0, rounds + 1)

        fig, ax1 = plt.subplots()
        ax1.set_xlim(0, rounds)
        ax1.set_ylim(0, 100)
        ax1.set_axisbelow(True)

        plt.grid(which="major", linestyle="-", linewidth=0.5)
        plt.grid(which="minor", linestyle="dotted", linewidth=0.2)
        ax1.tick_params(axis="x", which="minor", bottom=False)
        ax1.tick_params(axis="x", which="major", bottom=False)
        ax1.set_xlabel("Round Number", fontsize=19.65)
        ax1.set_ylabel("Accuracy", fontsize=20.25)
        ax1.plot(round_numbers, acc, "o-", color=self.colors[0])

        ax2 = ax1.twinx()
        ax2.set_ylabel("Loss", fontsize=20.25)
        ax2.minorticks_on()
        ml = mpl.ticker.MultipleLocator(0.25)
        ax2.yaxis.set_minor_locator(ml)
        ax2.set_ylim(0, 10)

        ax2.plot(round_numbers, loss, "o--", color=self.colors[0])
        line_acc = Line2D([0], [0], label="Accuracy", color=self.colors[0])
        line_loss = Line2D([0], [0], linestyle="--", label="Loss", color=self.colors[0])
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.extend(
            [
                line_acc,
                line_loss,
            ]
        )

        plt.legend(handles=handles, prop={"size": 13.85}, loc="best")
        os.makedirs(exist_ok=True, name="plots")
        plt.savefig(f"plots/{self.id}_loss_vs_acc.jpg")
        plt.close()
