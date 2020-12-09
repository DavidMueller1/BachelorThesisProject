from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from rl_algorithms.deep_q import DQN
import numpy as np
import time
from matplotlib import cm
from logger import Logger


def plot_network(network):
    # Logger.debug("Test:", network[0].weight.data.numpy())
    Logger.debug("Test:", network.fc1.weight.cpu().data.numpy())


def plot_network_layer(figure_num, title, layer_values, current_episode):
    if current_episode == 0:
        return
    layer_values = np.transpose(np.array(layer_values))[0]
    step_size = current_episode / (layer_values[0].size - 1)
    x = np.arange(0, current_episode + 1, step_size)
    plt.figure(figure_num)
    plt.clf()
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Weight")
    for index, weights in enumerate(layer_values):
        plt.plot(x, weights)
    # plt.legend(loc='lower right')
    plt.pause(0.00001)


def plot_progress(values, exploration_rate, average_period=100, time_left=False):
    plt.figure(2)
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(values, label="Reward in episode x")
    plt.plot(get_average(values, average_period), label="Average reward per " + str(average_period) + " episodes")
    plt.legend(loc='lower right')
    plt.subplots_adjust(bottom=0.2)
    # plt.gcf().text(0.02, -0.1, "Exploration rate: " + str(exploration_rate), fontsize=12)
    # plt.annotate("Test", [0, -20])
    text = "Exploration rate: %.2f" % exploration_rate
    if time_left:
        text += "\nTime left: " + str(time_left).split(".")[0]
    plt.text(0.02, 0.025, text, fontsize=10, transform=plt.gcf().transFigure)
    # plt.autoscale()
    # plt.show()
    plt.pause(0.00001)


def get_average(values, period):
    if len(values) >= period:
        cumsum = np.cumsum(np.insert(values, 0, 0))
        return np.concatenate((np.zeros(period), (cumsum[period:] - cumsum[:-period]) / float(period)))
    else:
        return np.zeros(len(values))


def plot_comparison(values, titles, average_period=100):
    plt.figure(3)
    plt.clf()
    plt.title("Average reward per " + str(average_period) + " episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    for index, learned in enumerate(values):
        plt.plot(get_average(learned.parameters.rewards_all_episodes, average_period), label=titles[index])
    plt.legend(loc='lower right')
    # plt.subplots_adjust(bottom=0.2)
    # plt.gcf().text(0.02, -0.1, "Exploration rate: " + str(exploration_rate), fontsize=12)
    # plt.annotate("Test", [0, -20])
    # plt.text(0.02, 0.025, "Exploration rate: %.2f" % exploration_rate, fontsize=10, transform=plt.gcf().transFigure)
    # plt.autoscale()
    # plt.show()
    plt.show()


def plot_q_table(q_table, world):
    # lin_x = np.arange(0, len(q_table[0]), 1)
    # lin_y = np.arange(0, 10, 1)
    # x, y = np.meshgrid(lin_x, lin_y)

    plot_north_q_table(q_table, world)
    # plot_east_q_table(q_table, world)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(x, y, q_table[:10])
    # plt.show()


def plot_north_q_table(q_table, world):
    lin_x = np.arange(0, 50, 1)
    lin_y = np.arange(0, 50, 1)
    x, y = np.meshgrid(lin_x, lin_y)
    z = np.zeros((50, 50))

    for state, val in enumerate(q_table):
        pos_x, pos_y = world.state_to_pos(state)
        z[pos_x + 25, pos_y + 25] = val[1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='bone')
    plt.pause(0.0001)
    # plt.show()


def plot_east_q_table(q_table, world):
    lin_x = np.arange(0, 50, 1)
    lin_y = np.arange(0, 50, 1)
    x, y = np.meshgrid(lin_x, lin_y)
    z = np.zeros((50, 50))

    for state, val in enumerate(q_table):
        pos_x, pos_y = world.state_to_pos(state)
        z[pos_x + 25, pos_y + 25] = val[2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='bone')
    plt.show()
