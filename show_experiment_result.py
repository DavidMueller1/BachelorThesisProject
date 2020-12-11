from data_util.data_loader import load_learned_with_terrain
from logger import Logger
import plot_util
from visualize_util import visualize_best_path
from visualize_util import visualize_best_path_deep_q
from plot_util import plot_progress
from visualization_engine_3d.engine import Engine3D
from tkinter.filedialog import askopenfilename
from data_util.experiment_data_classes import LearnedDeepQ
from data_util.experiment_data_classes import Learned
import time


Logger.input("Which data-file would you like to open?")
file_name = askopenfilename()
learned, terrain = load_learned_with_terrain(file_name)
scale = 14
distance = 100
Logger.status("Loaded data-file \"" + file_name + "\". Initializing renderer...")

world = Engine3D(terrain, agent_pos=(0, 0), scale=scale, distance=distance, width=800, height=800)
world.render()

Logger.info("Highest point is", terrain.highest_point)

if isinstance(learned, Learned):
    Logger.info("Data is a Q-Learning sample")
    plot_util.plot_q_table(learned.q_table, world)

    while True:
        Logger.status("Showing best learned path...")
        visualize_best_path(world, learned.parameters, learned.q_table)
        Logger.status("Highest point reached. Playing again...")
        # time.sleep(5)
elif isinstance(learned, LearnedDeepQ):
    Logger.info("Data is a Deep-Q-Learning sample")
    Logger.debug("Values:", learned.parameters.rewards_all_episodes)
    plot_progress(learned.parameters.rewards_all_episodes)
    while True:
        Logger.status("Showing best learned path...")
        visualize_best_path_deep_q(world, learned.parameters, learned.network)
        Logger.status("Highest point reached. Playing again...")
        # time.sleep(5)
