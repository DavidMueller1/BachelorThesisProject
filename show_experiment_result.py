from data_util.data_loader import load_learned_with_terrain
from logger import Logger
from visualize_util import visualize_best_path
from visualization_engine_3d.engine import Engine3D
import time


Logger.input("Which data-file would you like to open?")
file_name = input()
learned, terrain = load_learned_with_terrain(file_name)
scale = 14
distance = 100
Logger.status("Loaded data-file \"" + file_name + "\". Initializing renderer...")

world = Engine3D(terrain, agent_pos=(0, 0), scale=scale, distance=distance, width=1400, height=750)
world.render()

while True:
    Logger.status("Showing best learned path...")
    visualize_best_path(world, learned.parameters, learned.q_table)
    Logger.status("Highest point reached. Playing again in 5 seconds...")
    time.sleep(5)
