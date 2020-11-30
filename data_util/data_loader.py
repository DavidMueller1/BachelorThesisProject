import dill
from data_util.experiment_data_classes import LearnedForSpecificTerrain
import visualization_engine_3d.engine as render_engine
from terrain_generator import random_terrain_generator
import random
import numpy as np
import time


def load_terrain(file: str):
    with open("data/terrain/" + file, 'rb') as file:
        return dill.load(file)


def load_learned_with_terrain(file: str):
    # with open("data/learned/" + file, 'rb') as file:
    with open(file, 'rb') as file:
        data: LearnedForSpecificTerrain = dill.load(file)

    with open("data/terrain/" + data.terrainFile, 'rb') as terrain_file:
        return data.learned, dill.load(terrain_file)


# print("Which file would you like to open?")
# file_name = input()
# with open("data/" + file_name, 'rb') as f:
#     save_data = dill.load(f)
#
# max_steps_per_episode = save_data.max_steps_per_episode
# q_table = save_data.q_table
# points = save_data.points
# width = save_data.width
# length = save_data.length
#
# scale = 12
# distance = 100
#
# triangles = random_terrain_generator.RandomTerrain.calculate_triangles(points, width, length)
#
# world = render_engine.Engine3D(points, triangles, grid_width=width, grid_height=length, agent_pos=(0, 0), scale=scale, distance=distance, width=1400, height=750)
# world.render()
#
# while True:
#     reward_demo = 0
#     max_reward_demo = 0
#     state = world.reset_agent()
#     for step in range(max_steps_per_episode):
#         exploration_rate_threshold = random.uniform(0, 1)
#         action = np.argmax(q_table[state, :])
#         new_state, reward = world.agent_perform_action(action)
#         state = new_state
#         reward_demo += reward
#         if max_reward_demo < reward:
#             max_reward_demo = reward
#
#         world.clear()
#         world.render()
#         time.sleep(0.1)
#
#     print("********REWARD DEMO********\n")
#     print(reward_demo)
#     print("********MAX REWARD DEMO********\n")
#     print(max_reward_demo)
#     time.sleep(4)
