from data_util.data_loader import load_learned_with_terrain
from logger import Logger
from plot_util import plot_comparison
import plot_util
from visualize_util import visualize_best_path
from visualization_engine_3d.engine import Engine3D
import time

scale = 14
distance = 100

learned_list = []
learned_titles = []

Logger.input("Which data-file would you like to open?")
file_name = input()
learned, _ = load_learned_with_terrain(file_name)
learned_list.append(learned)
learned_titles.append(file_name)

Logger.status("Loaded data-file \"" + file_name + "\".")
Logger.status("You can keep adding files. When you are done press enter.")

while True:
    Logger.input("Which data-file would you like to add? If none, press enter.")
    file_name = input()
    if file_name == "":
        break
    learned, _ = load_learned_with_terrain(file_name)
    learned_list.append(learned)
    learned_titles.append(file_name)

    Logger.status("Loaded data-file \"" + file_name + "\".")

plot_comparison(learned_list, learned_titles)
