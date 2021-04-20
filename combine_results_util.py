from data_util.data_loader import load_learned_with_terrain, load_terrain
from data_util.data_loader import load_massive_result_data
from logger import Logger
from plot_util import plot_comparison, plot_result, plot_mean_with_std_multiple, plot_boxplots
from tkinter.filedialog import askopenfilenames
from tkinter import simpledialog
from operator import add
from visualization_engine_3d.engine import Rewards, States
import gym

from visualization_engine_3d.engine import Engine3D
from visualize_util import get_reward_from_trained_net_on_environment, get_reward_from_trained_net_on_environment_lunar_lander
import numpy as np

terrain_file = "test_2"

def compare_results(custom_titles=False):
    learned_list = []
    learned_titles = []

    Logger.input("Which data-files would you like to open?")
    file_names = askopenfilenames()
    if not file_names:
        return
    for file_name in file_names:
        learned, _ = load_learned_with_terrain(file_name)
        learned_list.append(learned)
        if custom_titles:
            custom_title = simpledialog.askstring(title="Enter title", prompt="Enter title for " + file_name)
            if custom_title:
                learned_titles.append(custom_title)
            else:
                learned_titles.append(file_name.split("/")[-1])
        else:
            learned_titles.append(file_name.split("/")[-1])
        Logger.status("Loaded data-file \"" + file_name + "\".")

    plot_comparison(learned_list, learned_titles)


def compare_massive_results(custom_titles=False):
    result_list = []
    result_datas = []
    result_titles = []

    Logger.input("Which data-files would you like to open?")
    file_names = askopenfilenames()
    if not file_names:
        return
    for file_name in file_names:
        learned = load_massive_result_data(file_name)
        result_list.append(learned)
        result_datas.append(learned.result_data)

        if custom_titles:
            custom_title = simpledialog.askstring(title="Enter title", prompt="Enter title for " + file_name)
            if custom_title:
                result_titles.append(custom_title)
            else:
                result_titles.append(file_name.split("/")[-1])
        else:
            result_titles.append(file_name.split("/")[-1])
        Logger.status("Loaded data-file \"" + file_name + "\".")

    plot_mean_with_std_multiple(result_datas, result_titles)


def compare_massive_single_results(custom_titles=False):
    result_list = []
    result_datas = []
    result_titles = []

    Logger.input("How many data-files would you like to open?")

    count = simpledialog.askinteger(title="Enter count", prompt="How many experiments would you like to open?")

    Logger.input("Which data-files would you like to open?")

    for n in range(count):
        q_table_zeros = 0
        q_table_zeros_count = 0

        file_names = askopenfilenames()
        experiment_results = []
        if not file_names:
            return
        for file_name in file_names:
            learned = load_massive_result_data(file_name)
            experiment_results.append(learned.single_result_data)
            if hasattr(learned, 'q_table'):
                q_table_zeros += learned.q_table.size - np.count_nonzero(learned.q_table)
                q_table_zeros_count += 1

        # result_list.append(learned)
        result_datas.append(experiment_results)

        if q_table_zeros_count > 0:
            Logger.info("Average Zeros in q_table: ", q_table_zeros / q_table_zeros_count)

        if custom_titles:
            custom_title = simpledialog.askstring(title="Enter title", prompt="Enter title for this experiment")
            if custom_title:
                result_titles.append(custom_title)
            else:
                result_titles.append(file_names[0].split("/")[-2])
        else:
            result_titles.append(file_names[0].split("/")[-2])
        Logger.status("Loaded data-files.")



    plot_mean_with_std_multiple(result_datas, result_titles)


def boxplot_massive_single_results(random_spawn, repeat_per_net=5):
    result_list = []
    result_titles = []

    custom_experiment_title = simpledialog.askstring(title="Enter title", prompt="Enter title for this experiment")
    if custom_experiment_title == "":
        custom_experiment_title = "Experiment"

    scale = 14
    distance = 100
    terrain = load_terrain(terrain_file)

    Logger.input("How many data-files would you like to open?")

    count = simpledialog.askinteger(title="Enter count", prompt="How many experiments would you like to open?")

    Logger.input("Which data-files would you like to open?")

    for n in range(count):
        file_names = askopenfilenames()
        experiment_results = []
        if not file_names:
            return
        custom_title = simpledialog.askstring(title="Enter title", prompt="Enter title for this experiment")
        if custom_title:
            result_titles.append(custom_title)
        else:
            result_titles.append(file_names[0].split("/")[-2])
        world = Engine3D(terrain, agent_pos=(0, 0), scale=scale, distance=distance, width=800, height=800,
                         random_spawn=random_spawn, reward_val=Rewards.Spiral, state_val=States.Default)
        Logger.status("Performing tests on " + terrain_file + "...")
        for file_name in file_names:
            learned = load_massive_result_data(file_name)
            if hasattr(learned, 'q_table'):
                Logger.debug("Has q_table")
            elif hasattr(learned, 'trained_net'):
                params = learned.params
                trained_net = learned.trained_net
                for _ in range(repeat_per_net):
                    experiment_results.append(get_reward_from_trained_net_on_environment(world, params, trained_net))

        result_list.append(experiment_results)
    Logger.debug(result_list)
    plot_boxplots(result_list, result_titles, custom_experiment_title)


def boxplot_massive_single_results_lunar_lander(repeat_per_net=10):
    result_list = []
    result_titles = []

    custom_experiment_title = simpledialog.askstring(title="Enter title", prompt="Enter title for this experiment")
    if custom_experiment_title == "":
        custom_experiment_title = "Experiment"

    scale = 14
    distance = 100
    terrain = load_terrain(terrain_file)

    Logger.input("How many data-files would you like to open?")

    count = simpledialog.askinteger(title="Enter count", prompt="How many experiments would you like to open?")

    Logger.input("Which data-files would you like to open?")

    for n in range(count):
        file_names = askopenfilenames()
        experiment_results = []
        if not file_names:
            return
        custom_title = simpledialog.askstring(title="Enter title", prompt="Enter title for this experiment")
        if custom_title:
            result_titles.append(custom_title)
        else:
            result_titles.append(file_names[0].split("/")[-2])

        env = gym.make('LunarLander-v2')
        Logger.status("Performing tests on " + terrain_file + "...")
        for file_name in file_names:
            learned = load_massive_result_data(file_name)
            params = learned.params
            trained_net = learned.trained_net
            for _ in range(repeat_per_net):
                experiment_results.append(get_reward_from_trained_net_on_environment_lunar_lander(env, params, trained_net))

        result_list.append(experiment_results)
    Logger.debug(result_list)
    plot_boxplots(result_list, result_titles, custom_experiment_title)


def combine_results(title=""):
    Logger.input("Which data-files would you like to open?")
    file_names = askopenfilenames()
    if not file_names:
        return
    learned_list = []
    for file_name in file_names:
        learned, _ = load_learned_with_terrain(file_name)
        learned_list.append(learned.parameters.rewards_all_episodes)
        # learned_titles.append(file_name.split("/")[-1])
        Logger.status("Loaded data-file \"" + file_name + "\".")

    max_size = len(min(learned_list, key=len))
    sum_list = [0] * max_size

    for learned in learned_list:
        sum_list = list(map(add, sum_list, learned[:max_size]))

    sum_list = [x / len(learned_list) for x in sum_list]

    plot_result(sum_list, title)
