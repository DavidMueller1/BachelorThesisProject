from visualization_engine_3d.engine import Engine3D
from init_terrain import generate_random_terrain
from data_util.experiment_data_classes import Terrain
from data_util.data_loader import load_terrain
from data_util.data_saver import save_terrain
from logger import Logger


# RENDERER
# scale = 20
scale = 14
distance = 100


# TERRAIN
terrain_file = "test_2"  # Will generate a new terrain if empty

terrain_saved = False  # Do not change
if terrain_file == "":
    Logger.status("Generating new terrain...")
    terrain: Terrain = generate_random_terrain()

    world = Engine3D(terrain, agent_pos=(0, 0), scale=scale, distance=distance, width=800, height=800, manual_control=True)
    world.render()

    Logger.input("Would you like to save the new Terrain? (y/n)")
    if input() == "y":
        Logger.input("Enter a file name: ")
        terrain_file = input()
        save_terrain(terrain_file, terrain)
        terrain_saved = terrain_file
        Logger.status("Terrain saved as \"" + terrain_file + "\"")

else:
    terrain_saved = terrain_file
    Logger.status("Loading terrain from file \"" + terrain_file + "\"...")
    terrain: Terrain = load_terrain(terrain_file)
    world = Engine3D(terrain, agent_pos=(0, 0), scale=scale, distance=distance, width=800, height=800, manual_control=True)
    world.render()

Logger.status("Terrain ready. Highest point is", terrain.highest_point)