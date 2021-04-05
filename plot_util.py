import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from rl_algorithms.deep_q import DQN
import numpy as np
import time
from matplotlib import cm
from logger import Logger

WIDTH = 422
HEIGHT = 250

DPI = 85

FULL_SIZE = 10
HALF_SIZE = 15
SIZE_MODE = 'FULL'


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


def plot_progress(values, exploration_rate=False, average_period=100, time_left=False, reward_val=False, epsilon=False,
                  epsilon_fac=1, width=False, height=False, show=False, title="Training...", latex_size=SIZE_MODE):
    first = False
    if not plt.fignum_exists(2):
        first = True
        # params = {'text.usetex': True,
        #           'font.size': 11
        #           }
        # plt.rcParams.update(params)
    if width and height:
        plt.figure(2, figsize=(width / DPI, height / DPI), dpi=DPI)
    else:
        plt.figure(2)

    if latex_size == 'FULL':
        matplotlib.rc('font', size=FULL_SIZE)
        matplotlib.rc('axes', titlesize=FULL_SIZE)
        matplotlib.rc('axes', labelsize=FULL_SIZE)
        matplotlib.rc('xtick', labelsize=FULL_SIZE)
        matplotlib.rc('ytick', labelsize=FULL_SIZE)
        matplotlib.rc('legend', fontsize=FULL_SIZE)
        matplotlib.rc('figure', titlesize=FULL_SIZE)
    elif latex_size == 'HALF':
        matplotlib.rc('font', size=HALF_SIZE)
        matplotlib.rc('axes', titlesize=HALF_SIZE)
        matplotlib.rc('axes', labelsize=HALF_SIZE)
        matplotlib.rc('xtick', labelsize=HALF_SIZE)
        matplotlib.rc('ytick', labelsize=HALF_SIZE)
        matplotlib.rc('legend', fontsize=HALF_SIZE)
        matplotlib.rc('figure', titlesize=HALF_SIZE)
    plt.clf()
    if title:
        plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Belohnung")
    plt.plot(values, label="Belohnung in episode x")
    averages = get_average(values, average_period)
    # Logger.info("Best Average:", np.amax(averages))
    # Logger.info("Last Average:", averages[-1])
    plt.plot(averages, label="Ø Belohnung pro " + str(average_period) + " Episoden")
    if latex_size is not False:
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
    else:
        plt.legend(loc='lower right')
    # plt.legend(bbox_to_anchor=(1, 0.5), loc='center left')
    if epsilon:
        sec_color = 'tab:cyan'
        ax2 = plt.twinx()
        ax2.set_ylabel("Epsilon", color=sec_color)
        ax2.plot(np.asarray(epsilon), label="Epsilon in Episode x", color=sec_color)
        ax2.tick_params(axis='y', labelcolor=sec_color)
    plt.subplots_adjust(bottom=0.2)
    # plt.gcf().text(0.02, -0.1, "Exploration rate: " + str(exploration_rate), fontsize=12)
    # plt.annotate("Test", [0, -20])
    text = ""
    if exploration_rate:
        text += "Exploration rate: %.2f" % exploration_rate
    if time_left:
        text += "\nTime left: " + str(time_left).split(".")[0]
    if reward_val:
        text += "\nReward: " + reward_val.name
    plt.text(0.02, 0.025, text, fontsize=10, transform=plt.gcf().transFigure)
    if first:
        plt.tight_layout()
    # plt.autoscale()
    if show:
        plt.show()
    else:
        plt.pause(0.001)
        # plt.pause(0.1)
        # plt.show()


# def plot_progress(values, exploration_rate=False, average_period=100, time_left=False, reward_val=False, epsilon=False, epsilon_fac=1):
#     plt.figure(2)
#     plt.clf()
#     plt.title("Training...")
#     plt.xlabel("Episode")
#     plt.ylabel("Reward")
#     plt.plot(values, label="Reward in episode x")
#     plt.plot(get_average(values, average_period), label="Average reward per " + str(average_period) + " episodes")
#     if epsilon:
#
#         plt.plot(np.asarray(epsilon) * epsilon_fac, label="Epsilon with factor %d in episode x" % epsilon_fac)
#     plt.legend(loc='lower right')
#     plt.subplots_adjust(bottom=0.2)
#     # plt.gcf().text(0.02, -0.1, "Exploration rate: " + str(exploration_rate), fontsize=12)
#     # plt.annotate("Test", [0, -20])
#     text = ""
#     if exploration_rate:
#         text += "Exploration rate: %.2f" % exploration_rate
#     if time_left:
#         text += "\nTime left: " + str(time_left).split(".")[0]
#     if reward_val:
#         text += "\nReward: " + reward_val.name
#     plt.text(0.02, 0.025, text, fontsize=10, transform=plt.gcf().transFigure)
#     # plt.autoscale()
#     # plt.show()
#     plt.pause(0.00001)


def plot_mean_with_std(result_data, title="Mean with Std", period=100, latex_size=SIZE_MODE):
    mean, std = get_moving_average_mean_and_std(result_data, period)
    std_upper = mean + std / 2
    std_lower = mean - std / 2

    if latex_size == 'FULL':
        matplotlib.rc('font', size=FULL_SIZE)
        matplotlib.rc('axes', titlesize=FULL_SIZE)
        matplotlib.rc('axes', labelsize=FULL_SIZE)
        matplotlib.rc('xtick', labelsize=FULL_SIZE)
        matplotlib.rc('ytick', labelsize=FULL_SIZE)
        matplotlib.rc('legend', fontsize=FULL_SIZE)
        matplotlib.rc('figure', titlesize=FULL_SIZE)
    elif latex_size == 'HALF':
        matplotlib.rc('font', size=HALF_SIZE)
        matplotlib.rc('axes', titlesize=HALF_SIZE)
        matplotlib.rc('axes', labelsize=HALF_SIZE)
        matplotlib.rc('xtick', labelsize=HALF_SIZE)
        matplotlib.rc('ytick', labelsize=HALF_SIZE)
        matplotlib.rc('legend', fontsize=HALF_SIZE)
        matplotlib.rc('figure', titlesize=HALF_SIZE)

    # plt.figure(2)
    # plt.clf()
    # plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Belohnung")
    plt.plot(mean, label="Average Reward in episode x")
    x = np.arange(0, mean.size, 1)
    print(x)
    plt.fill_between(x, std_upper, std_lower, alpha=0.3)
    plt.show()


def plot_mean_with_std_multiple(result_datas, titles, title="Mean with Std", period=100, latex_size=SIZE_MODE):
    plt.figure(7, figsize=(WIDTH / DPI, HEIGHT / DPI))

    if latex_size == 'FULL':
        matplotlib.rc('font', size=FULL_SIZE)
        matplotlib.rc('axes', titlesize=FULL_SIZE)
        matplotlib.rc('axes', labelsize=FULL_SIZE)
        matplotlib.rc('xtick', labelsize=FULL_SIZE)
        matplotlib.rc('ytick', labelsize=FULL_SIZE)
        matplotlib.rc('legend', fontsize=FULL_SIZE)
        matplotlib.rc('figure', titlesize=FULL_SIZE)
    elif latex_size == 'HALF':
        matplotlib.rc('font', size=HALF_SIZE)
        matplotlib.rc('axes', titlesize=HALF_SIZE)
        matplotlib.rc('axes', labelsize=HALF_SIZE)
        matplotlib.rc('xtick', labelsize=HALF_SIZE)
        matplotlib.rc('ytick', labelsize=HALF_SIZE)
        matplotlib.rc('legend', fontsize=HALF_SIZE)
        matplotlib.rc('figure', titlesize=HALF_SIZE)
    # fig = plt.figure(7)
    # dpi = fig.get_dpi()
    # fig.set_size_inches(422.5 / float(DPI), 300.0 / float(dpi))
    # fig.set_size_inches(WIDTH, HEIGHT)
    # plt.clf()
    # plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Belohnung")
    for index, result_data in enumerate(result_datas):
        mean, std = get_moving_average_mean_and_std(result_data, period)
        std_upper = mean + std / 2
        std_lower = mean - std / 2
        plt.plot(mean, label=titles[index])
        x = np.arange(0, mean.size, 1)
        plt.fill_between(x, std_upper, std_lower, alpha=0.3)
    # plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def plot_boxplots(values, titles):
    plt.figure(3)
    plt.xlabel("Experiment")
    plt.ylabel("Belohnung")
    plt.boxplot(values)
    plt.xticks(np.arange(1, len(values) + 1), titles)
    plt.tight_layout()
    plt.show()


def get_average(values, period):
    if len(values) >= period:
        cumsum = np.cumsum(np.insert(values, 0, 0))
        return np.concatenate((np.zeros(period), (cumsum[period:] - cumsum[:-period]) / float(period)))
    else:
        return np.zeros(len(values))


def get_moving_average_mean_and_std(values, period=100):
    moving_avg_vals = []
    for experiment in values:
        moving_avg_vals.append(get_average(experiment, period))

    data_np = np.array(moving_avg_vals)
    mean = data_np.mean(0)
    std = data_np.std(0)
    return mean, std


def get_current_average(values, period=100):
    if len(values) >= period:
        return sum(values[-period:]) / float(period)
    else:
        return 0


def plot_result(values, title, average_period=100):
    plt.figure(3)
    plt.clf()
    plt.title("Average reward per " + str(average_period) + " episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(get_average(values, average_period), label=title)
    plt.legend(loc='lower right')
    # plt.subplots_adjust(bottom=0.2)
    # plt.gcf().text(0.02, -0.1, "Exploration rate: " + str(exploration_rate), fontsize=12)
    # plt.annotate("Test", [0, -20])
    # plt.text(0.02, 0.025, "Exploration rate: %.2f" % exploration_rate, fontsize=10, transform=plt.gcf().transFigure)
    # plt.autoscale()
    # plt.show()
    plt.show()


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
