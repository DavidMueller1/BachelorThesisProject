from visualization_engine_3d.engine import Engine3D
from init_terrain import generate_random_terrain
from data_util.experiment_data_classes import Terrain
from data_util.experiment_data_classes import Parameters
from data_util.experiment_data_classes import LearnedForSpecificTerrain
from data_util.experiment_data_classes import Learned
from data_util.data_loader import load_terrain
from data_util.data_saver import save_terrain
from data_util.data_saver import save_learned_for_terrain
from visualize_util import visualize_best_path
from rl_algorithms import simple
from rl_algorithms import buffer
from rl_algorithms import exploration_as_reward
from logger import Logger
import numpy as np
import time


plot_training_progress = True  # If True, the training will take longer
plot_interval = 10  # Only refresh plot every n episodes. Will speed up training
plot_moving_average_period = 100  # The period in which the average is computed

visualize_training = False  # If True, the training will obviously take much much longer


# RENDERER
# scale = 20
scale = 14
distance = 100


# TERRAIN
# terrain_file = "test_2"  # Will generate a new terrain if empty
terrain_file = ""  # Will generate a new terrain if empty

terrain_saved = False  # Do not change
if terrain_file == "":
    Logger.status("Generating new terrain...")
    # terrain: Terrain = generate_random_terrain()
    terrain: Terrain = generate_random_terrain(width=50, length=50, n1div=15)

    world = Engine3D(terrain, agent_pos=(0, 0), scale=scale, distance=distance, width=800, height=800, manual_control=False)
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
    world = Engine3D(terrain, agent_pos=(0, 0), scale=scale, distance=distance, width=800, height=800)
    world.render()

Logger.status("Terrain ready. Highest point is", terrain.highest_point)


# TRAINING
Logger.status("Beginning training...")

params = Parameters(
    num_episodes=10000,
    max_steps_per_episode=500,

    learning_rate=0.5,
    discount_rate=0.99,

    start_exploration_rate=1,
    max_exploration_rate=1,
    min_exploration_rate=0.01,
    exploration_decay_rate=0.0001,

    rewards_all_episodes=[],
    max_rewards_all_episodes=[],
)

q_table, params = buffer.train(width=terrain.width, length=terrain.length, params=params, environment=world,
                        visualize=visualize_training, plot=plot_training_progress, plot_interval=plot_interval,
                        plot_moving_avg_period=plot_moving_average_period)
Logger.status("Training done.")


# SHOW RESULT
Logger.info("Zeros in q_table: ", q_table.size - np.count_nonzero(q_table), "/", q_table.size)
Logger.info("Non-Zeros in q_table: ", np.count_nonzero(q_table))

Logger.status("Showing best learned path...")
time.sleep(3)
visualize_best_path(world, params, q_table)


Logger.input("Would you like to save the Q-Table and the Params? (y/n)")
if input() == "y":
    if not terrain_saved:
        Logger.input("You need to save the terrain as well. Please enter a terrain file name: ")
        terrain_file = input()
        save_terrain(terrain_file, terrain)
        terrain_saved = terrain_file

    Logger.input("Enter a file name for the learned data: ")
    file_name = input()
    save_learned_for_terrain(file_name, LearnedForSpecificTerrain(Learned(q_table, params), terrain_saved))
    Logger.status("Data saved as \"" + file_name + "\"")


while True:
    Logger.status("Showing best learned path...")
    visualize_best_path(world, params, q_table)
    Logger.status("Highest point reached. Playing again in 5 seconds...")
    time.sleep(5)
