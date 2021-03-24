import sys
from logger import Logger
import os
import dill
import numpy as np
import random
import math
import json
from matplotlib import pyplot as plt
from plot_util import plot_mean_with_std
from data_util.data_loader import load_massive_result_data
from tkinter.filedialog import askopenfilename
from visualize_util import visualize_best_path_deep_q
from terrain_generator.terrain_util import get_environment

# Logger.debug([-1, 1][random.randint(0, 1)])
# a = random.uniform(0, 1)
# Logger.debug(a)
# Logger.debug(math.sqrt(1 - a ** 2))

file_name = askopenfilename()
data = load_massive_result_data(file_name)
Logger.debug(data.params)

# terrain_file = "test_2"
# environment = get_environment(terrain_file=terrain_file)
#
# # os.chdir(os.path.dirname(__file__))
# # with open("data/learned/2021_02_02_massive_different_epsilon/decaying_epsilon", 'rb') as file:
# #     result_data = dill.load(file)
# #
# # plot_mean_with_std(result_data.result_data)
#
# os.chdir(os.path.dirname(__file__))
# with open("data/learned/new_experiments_path/2021_03_05_decaying_epsilon_start_1/decaying_epsilon_start_1_exp01", 'rb') as file:
#     single_result_data = dill.load(file)
# visualize_best_path_deep_q(environment, single_result_data.params, single_result_data.trained_net)

# data = [
#     [1, 2, 3],
#     [2, 2, 3],
#     [3, 2, 3],
#     [4, 2, 3],
#     [5, 2, 3],
#     [6, 2, 3],
#     [7, 2, 3]
#     ]
#
# data_np = np.array(data)
# mean = data_np.mean(0)
# std = data_np.std(0)
# std_upper = mean + std / 2
# std_lower = mean - std / 2
# print(data_np)
# print(mean)
# print(std)
# print("-----")
# print(std_upper)
# print(std_lower)
#
# # plt.figure(2)
# # plt.clf()
# plt.title("Training...")
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.plot(mean, label="Reward in episode x")
# x = np.arange(0, mean.size, 1)
# print(x)
# plt.fill_between(x, std_upper, std_lower, alpha=0.3)
# plt.show()


# plt.plot(get_average(values, average_period), label="Average reward per " + str(average_period) + " episodes")
# if epsilon:
#     plt.plot(np.asarray(epsilon) * epsilon_fac, label="Epsilon with factor %d in episode x" % epsilon_fac)
# plt.legend(loc='lower right')
# plt.subplots_adjust(bottom=0.2)
# # plt.gcf().text(0.02, -0.1, "Exploration rate: " + str(exploration_rate), fontsize=12)
# # plt.annotate("Test", [0, -20])
# text = ""
# if exploration_rate:
#     text += "Exploration rate: %.2f" % exploration_rate
# if time_left:
#     text += "\nTime left: " + str(time_left).split(".")[0]
# if reward_val:
#     text += "\nReward: " + reward_val.name
# plt.text(0.02, 0.025, text, fontsize=10, transform=plt.gcf().transFigure)
# # plt.autoscale()
# # plt.show()
# plt.pause(0.00001)