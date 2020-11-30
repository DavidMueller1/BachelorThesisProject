from data_util.data_loader import load_learned_with_terrain
from logger import Logger
import plot_util
from visualize_util import visualize_best_path
from visualization_engine_3d.engine import Engine3D
from tkinter.filedialog import askopenfilename
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

plot_util.plot_q_table(learned.q_table, world)

while True:
    Logger.status("Showing best learned path...")
    visualize_best_path(world, learned.parameters, learned.q_table)
    Logger.status("Highest point reached. Playing again...")
    # time.sleep(5)
