from logger import Logger
from init_terrain import generate_random_terrain
from visualization_engine_3d.engine import Engine3D
from data_util.data_saver import save_terrain
from data_util.data_loader import load_terrain
from visualization_engine_3d.engine import Rewards


def get_environment(terrain_file, scale=14, distance=100, random_spawn=True, render_world=True):

    # TERRAIN
    if terrain_file == "":
        Logger.status("Generating new terrain...")
        terrain = generate_random_terrain()

        world = Engine3D(terrain, agent_pos=(0, 0), scale=scale, distance=distance, width=800, height=800, random_spawn=random_spawn, reward_val=Rewards.Spiral)
        if render_world:
            world.render()

        Logger.input("Would you like to save the new Terrain? (y/n)")
        if input() == "y":
            Logger.input("Enter a file name: ")
            terrain_file = input()
            save_terrain(terrain_file, terrain)
            Logger.status("Terrain saved as \"" + terrain_file + "\"")

    else:
        Logger.status("Loading terrain from file \"" + terrain_file + "\"...")
        terrain = load_terrain(terrain_file)
        world = Engine3D(terrain, agent_pos=(0, 0), scale=scale, distance=distance, width=800, height=800,
                         random_spawn=random_spawn, reward_val=Rewards.Spiral)
        if render_world:
            world.render()

    Logger.status("Terrain ready. Highest point is", terrain.highest_point)

    return world
