from data_util.data_loader import load_learned_with_terrain, load_massive_result_data, load_massive_single_result_data, \
    load_terrain
from logger import Logger
import plot_util
from visualize_util import visualize_best_path
from visualize_util import visualize_best_path_deep_q
from plot_util import plot_progress
from visualization_engine_3d.engine import Engine3D
from tkinter.filedialog import askopenfilename
from data_util.experiment_data_classes import LearnedDeepQ
from data_util.experiment_data_classes import Learned
from tkinter import *
import numpy as np
import time
from visualization_engine_3d.engine import Rewards, States

terrain_file = "test_2"


def show_experiment_result(random_spawn=False, repeat=True):
    Logger.input("Which data-file would you like to open?")
    file_name = askopenfilename()
    data = load_massive_single_result_data(file_name)
    scale = 14
    distance = 100
    Logger.status("Loaded data-file \"" + file_name + "\". Initializing renderer...")

    terrain = load_terrain(terrain_file)
    # world = Engine3D(terrain, agent_pos=(0, 0), scale=scale, distance=distance, width=800, height=800, random_spawn=random_spawn)
    world = Engine3D(terrain, agent_pos=(0, 0), scale=scale, distance=distance, width=800, height=800, random_spawn=random_spawn, reward_val=Rewards.Delta, state_val=States.Default)
    world.render()

    if hasattr(data, 'q_table'):
        Logger.info("Data is a Q-Learning sample")
        Logger.info("Zeros in q_table: ", data.q_table.size - np.count_nonzero(data.q_table), "/", data.q_table.size)
        # plot_util.plot_q_table(data.q_table, world)

        eps_history = []
        exploration_rate = data.params.start_exploration_rate
        for episode in range(data.params.num_episodes):
            eps_history.append(exploration_rate)
            exploration_rate = data.params.min_exploration_rate if data.params.min_exploration_rate >= exploration_rate else data.params.min_exploration_rate + (
                    data.params.max_exploration_rate - data.params.min_exploration_rate) * np.exp(
                -data.params.exploration_decay_rate * episode)

        plot_progress(data.params.rewards_all_episodes, epsilon=eps_history, title=False, latex_size='HALF')

        while True:
            Logger.status("Showing best learned path...")
            visualize_best_path(world, data.params, data.q_table)
            Logger.status("Highest point reached. Playing again...")
            if not repeat:
                break
            time.sleep(0.5)
    elif hasattr(data, 'trained_net'):
        Logger.info("Data is a Deep-Q-Learning sample")
        # Logger.debug("Values:", data.parameters.rewards_all_episodes)
        eps_history = []
        exploration_rate = data.params.start_exploration_rate
        for episode in range(data.params.num_episodes):
            eps_history.append(exploration_rate)
            exploration_rate = data.params.min_exploration_rate if data.params.min_exploration_rate >= exploration_rate else data.params.min_exploration_rate + (
                    data.params.max_exploration_rate - data.params.min_exploration_rate) * np.exp(
                -data.params.exploration_decay_rate * episode)
        plot_progress(data.params.rewards_all_episodes, epsilon=eps_history, title=False)
        while True:
            Logger.status("Showing best learned path...")
            visualize_best_path_deep_q(world, data.params, data.trained_net)
            Logger.status("Highest point reached. Playing again...")
            if not repeat:
                break
            time.sleep(1)


def show_experiment_result_with_terrain(random_spawn=False):
    Logger.input("Which data-file would you like to open?")
    file_name = askopenfilename()
    learned, terrain = load_learned_with_terrain(file_name)
    scale = 14
    distance = 100
    Logger.status("Loaded data-file \"" + file_name + "\". Initializing renderer...")

    world = Engine3D(terrain, agent_pos=(0, 0), scale=scale, distance=distance, width=800, height=800, random_spawn=random_spawn)
    world.render()

    Logger.info("Highest point is", terrain.highest_point)

    if isinstance(learned, Learned):
        Logger.info("Data is a Q-Learning sample")
        plot_util.plot_q_table(learned.q_table, world)
        plot_progress(learned.parameters.rewards_all_episodes)

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


def show_params():
    Logger.input("Which data-file would you like to open?")
    file_name = askopenfilename()
    data = load_massive_result_data(file_name)
    data.params.rewards_all_episodes = []
    data.params.max_rewards_all_episodes = []
    Logger.info(data.params)


def show_experiment_result_luna_lander(repeat=True):
    Logger.input("Which data-file would you like to open?")
    file_name = askopenfilename()
    data = load_massive_single_result_data(file_name)
    Logger.status("Loaded data-file \"" + file_name + "\". Initializing renderer...")
    plot_progress(data.params.rewards_all_episodes)

        # Logger.debug("Values:", data.parameters.rewards_all_episodes)
        # while True:
        #     Logger.status("Showing best learned path...")
        #     visualize_best_path_deep_q(world, data.params, data.trained_net)
        #     Logger.status("Highest point reached. Playing again...")
        #     time.sleep(0.5)
