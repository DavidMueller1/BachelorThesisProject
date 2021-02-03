from visualization_engine_3d.engine import Engine3D
from init_terrain import generate_random_terrain
from data_util.experiment_data_classes import Terrain
from data_util.experiment_data_classes import DeepQParameters
from data_util.experiment_data_classes import LearnedDeepQForSpecificTerrain
from data_util.experiment_data_classes import LearnedDeepQ
from data_util.data_loader import load_terrain
from data_util.data_saver import save_terrain
from data_util.data_saver import save_learned_for_terrain_deep_q_absolute_path
from visualize_util import visualize_best_path_deep_q
from tkinter.filedialog import asksaveasfilename
from rl_algorithms import simple
from rl_algorithms import buffer
# from rl_algorithms import deep_q
from rl_algorithms import deep_q_2
from rl_algorithms import deep_q_2_changing_reward
from rl_algorithms import exploration_as_reward
from logger import Logger
import numpy as np
import time
import torch
from visualization_engine_3d.engine import Rewards


plot_training_progress = True  # If True, the training will take longer
plot_interval = 5  # Only refresh plot every n episodes. Will speed up training
plot_moving_average_period = 100  # The period in which the average is computed

show_path_interval = 10  # The period in which the current episode's path is shown. 0 to never show

visualize_training = False  # If True, the training will obviously take much much longer


# RENDERER
# scale = 20
scale = 14
distance = 100

# Reward
reward = Rewards.Spiral


# TERRAIN
terrain_file = "test_2"  # Will generate a new terrain if empty

terrain_saved = False  # Do not change
if terrain_file == "":
    Logger.status("Generating new terrain...")
    terrain: Terrain = generate_random_terrain()

    world = Engine3D(terrain, agent_pos=(0, 0), scale=scale, distance=distance, width=800, height=800, reward_val=reward)
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
    world = Engine3D(terrain, agent_pos=(0, 0), scale=scale, distance=distance, width=800, height=800, random_spawn=True, reward_val=reward)
    world.render()

Logger.status("Terrain ready. Highest point is", terrain.highest_point)


# TRAINING
Logger.status("Beginning training...")

# params = DeepQParameters(
#     num_episodes=10000,
#     max_steps_per_episode=100,
#     replay_buffer_size=200,
#     batch_size=5,
#
#     # learning_rate=0.001,
#     learning_rate=0.001,
#     discount_rate=0.999,
#     target_update=10,
#
#     start_exploration_rate=1,
#     max_exploration_rate=1,
#     min_exploration_rate=0.01,
#     exploration_decay_rate=0.0002,
#
#     rewards_all_episodes=[],
#     max_rewards_all_episodes=[],
# )

params = DeepQParameters(
    num_episodes=2000,
    max_steps_per_episode=80,
    # replay_buffer_size=60000,
    replay_buffer_size=20000,
    batch_size=64,
    # batch_size=32,
    # batch_size=8,

    # learning_rate=0.001,
    learning_rate=0.001,
    discount_rate=0.999,
    target_update=25,

    # start_exploration_rate=0.2,
    start_exploration_rate=1,
    max_exploration_rate=1,
    # min_exploration_rate=0.2,
    min_exploration_rate=0.001,
    # exploration_decay_rate=0,
    exploration_decay_rate=0.005,

    rewards_all_episodes=[],
    max_rewards_all_episodes=[],
    max_reward_average=0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_net, params = deep_q_2.train(width=terrain.width, length=terrain.length, params=params, environment=world,
                        visualize=visualize_training, show_path_interval=show_path_interval, plot=plot_training_progress, plot_interval=plot_interval,
                        plot_moving_avg_period=plot_moving_average_period)

# target_net, params = deep_q_2_changing_reward.train(width=terrain.width, length=terrain.length, params=params, environment=world,
#                         visualize=visualize_training, show_path_interval=show_path_interval, plot=plot_training_progress, plot_interval=plot_interval,
#                         plot_moving_avg_period=plot_moving_average_period)
Logger.status("Training done.")


# SHOW RESULT
# Logger.info("Zeros in q_table: ", q_table.size - np.count_nonzero(q_table), "/", q_table.size)
# Logger.info("Non-Zeros in q_table: ", np.count_nonzero(q_table))

# Logger.status("Showing best learned path...")
# while True:
#     time.sleep(3)
#     world.clear_path()
#     visualize_best_path_deep_q(world, params, target_net)
#     Logger.input("Show again? (y/n)")
#     if input() == "n":
#         break


Logger.input("Would you like to save the Target-Net and the Params? (y/n)")
if input() == "y":
    if not terrain_saved:
        Logger.input("You need to save the terrain as well. Please enter a terrain file name: ")
        terrain_file = input()
        save_terrain(terrain_file, terrain)
        terrain_saved = terrain_file

    Logger.input("Enter a file name for the learned data: ")
    # file_name = input()
    file_name = asksaveasfilename()
    save_learned_for_terrain_deep_q_absolute_path(file_name, LearnedDeepQForSpecificTerrain(LearnedDeepQ(target_net, params), terrain_saved))
    Logger.status("Data saved as \"" + file_name + "\"")


while True:
    Logger.input("Show again? (y/n)")
    if input() == "n":
        break
    # time.sleep(3)
    visualize_best_path_deep_q(world, params, target_net)

