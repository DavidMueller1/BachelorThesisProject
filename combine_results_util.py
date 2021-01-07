from data_util.data_loader import load_learned_with_terrain
from logger import Logger
from plot_util import plot_comparison
from tkinter.filedialog import askopenfilenames
from tkinter import simpledialog


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


def combine_results():
    learned_list = []
    learned_titles = []

    Logger.input("Which data-files would you like to open?")
    file_names = askopenfilenames()
    if not file_names:
        return
    for file_name in file_names:
        learned, _ = load_learned_with_terrain(file_name)
        learned_list.append(learned)
        learned_titles.append(file_name.split("/")[-1])
        Logger.status("Loaded data-file \"" + file_name + "\".")

    plot_comparison(learned_list, learned_titles)
