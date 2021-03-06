from data_util.data_loader import load_learned_with_terrain
from data_util.data_loader import load_massive_result_data
from logger import Logger
from plot_util import plot_comparison, plot_result, plot_mean_with_std_multiple
from tkinter.filedialog import askopenfilenames
from tkinter import simpledialog
from operator import add
import numpy as np


def compare_results(custom_titles=False):
    learned_list = []
    learned_titles = []

    Logger.input("Which data-files would you like to open?")
    file_names = askopenfilenames()
    if not file_names:
        return
    for file_name in file_names:
        learned, _ = load_learned_with_terrain(file_name)
        learned_list.append(learned)
        if custom_titles:
            custom_title = simpledialog.askstring(title="Enter title", prompt="Enter title for " + file_name)
            if custom_title:
                learned_titles.append(custom_title)
            else:
                learned_titles.append(file_name.split("/")[-1])
        else:
            learned_titles.append(file_name.split("/")[-1])
        Logger.status("Loaded data-file \"" + file_name + "\".")

    plot_comparison(learned_list, learned_titles)


def compare_massive_results(custom_titles=False):
    result_list = []
    result_datas = []
    result_titles = []

    Logger.input("Which data-files would you like to open?")
    file_names = askopenfilenames()
    if not file_names:
        return
    for file_name in file_names:
        learned = load_massive_result_data(file_name)
        result_list.append(learned)
        result_datas.append(learned.result_data)

        if custom_titles:
            custom_title = simpledialog.askstring(title="Enter title", prompt="Enter title for " + file_name)
            if custom_title:
                result_titles.append(custom_title)
            else:
                result_titles.append(file_name.split("/")[-1])
        else:
            result_titles.append(file_name.split("/")[-1])
        Logger.status("Loaded data-file \"" + file_name + "\".")

    plot_mean_with_std_multiple(result_datas, result_titles)


def combine_results(title=""):
    Logger.input("Which data-files would you like to open?")
    file_names = askopenfilenames()
    if not file_names:
        return
    learned_list = []
    for file_name in file_names:
        learned, _ = load_learned_with_terrain(file_name)
        learned_list.append(learned.parameters.rewards_all_episodes)
        # learned_titles.append(file_name.split("/")[-1])
        Logger.status("Loaded data-file \"" + file_name + "\".")

    max_size = len(min(learned_list, key=len))
    sum_list = [0] * max_size

    for learned in learned_list:
        sum_list = list(map(add, sum_list, learned[:max_size]))

    sum_list = [x / len(learned_list) for x in sum_list]

    plot_result(sum_list, title)
